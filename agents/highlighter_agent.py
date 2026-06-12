from agents.base_agent import BaseAgent
from config import FOCUS_POSITIONS, MAX_HIGHLIGHT_DURATION, ZOOM_MAX_LEVEL


class HighlighterAgent(BaseAgent):
    """
    Selects the best highlight segments by reasoning about content, emotion,
    and energy — not just filling a time target mechanically.
    """

    SYSTEM = (
        "You are a professional video editor. "
        "You watch videos and find the moments that best match what the viewer asked for. "
        "You think about what is actually happening in each scene — the emotion, the energy, "
        "the visual action — and pick accordingly. "
        "Always respond with ONLY a valid JSON object — no prose, no markdown fences."
    )

    def run(
        self,
        explained_data: dict,
        max_duration: int = None,
        style: str = None,
        feedback: str = "",
    ) -> dict:
        max_dur = max_duration or MAX_HIGHLIGHT_DURATION
        video_duration = float(explained_data.get("duration", 1e9))

        # Build a rich per-segment description so the LLM sees full context:
        # importance score, the reason it was scored, and the actual content
        segments_text = _build_segments_text(explained_data.get("segments", []))

        style_block = (
            f"\nWHAT THE USER WANTS: \"{style}\"\n"
            "This is your PRIMARY goal. Read every segment below and find the ones that "
            "best match this. Forget structural rules — find the RIGHT moments.\n"
            if style
            else "\nNo style specified — pick the highest-importance moments that together tell a coherent story.\n"
        )

        feedback_block = (
            f"\nPREVIOUS ATTEMPT FEEDBACK — fix these specific issues:\n{feedback}\n"
            if feedback else ""
        )

        prompt = f"""You are selecting clips for a {max_dur}-second highlight reel.

VIDEO INFO:
  Duration : {video_duration}s
  Summary  : {explained_data.get('summary', 'unknown')}
  Tone     : {explained_data.get('tone', 'unknown')}
  Pacing   : {explained_data.get('pacing', 'unknown')}
{style_block}
AVAILABLE SEGMENTS (read every one carefully before deciding):
{segments_text}

SELECTION RULES:
1. CONTENT FIRST: Pick segments where what is happening matches what the user asked for.
   Read the CONTENT and REASON fields — they tell you what is in the scene.
2. DO NOT CUT MID-SENTENCE: If a speech segment is selected, use its full start-to-end
   time. Never trim a spoken thought mid-way.
3. MERGE ADJACENT CLIPS: If two consecutive segments you want are within 1 second of each
   other, merge them into one continuous clip (use the earlier start and later end).
4. STYLE BEATS IMPORTANCE: A segment with importance=0.4 that perfectly matches the style
   is better than a 0.9 segment that does not match.
5. DURATION CONSTRAINT: Total must not exceed {max_dur}s. Minimum is {int(max_dur * 0.7)}s.
   If you cannot fill the minimum with matching content, extend your best segments slightly.
6. All start/end values must be plain numbers within 0–{video_duration}s.
7. ZOOM: Each segment's FOCUS field tells you where the main subject/action sits in the
   frame. For each highlight, decide whether punching in on that area would make the clip
   more impactful — e.g. a single speaker, a small but important object, a face reacting.
   - If yes, set "zoom": {{"position": "<that FOCUS value>", "level": a number between
     1.1 and {ZOOM_MAX_LEVEL} (higher = tighter zoom)}}.
   - If no — wide shots, group/crowd scenes, fast action, or FOCUS is "center" with
     nothing specific to emphasize — set "zoom": {{"position": "center", "level": 1.0}}.
{feedback_block}
Return a JSON object with:
- "highlights": list of selected clips, each with:
    - "id": the original segment id (or "merged" for merged clips)
    - "start": start time in seconds (plain number)
    - "end": end time in seconds (plain number)
    - "reason": one sentence explaining exactly WHY this clip matches the request
    - "zoom": {{"position": one of [{", ".join(FOCUS_POSITIONS)}], "level": 1.0-{ZOOM_MAX_LEVEL}}}
- "total_duration": sum of clip durations (plain number)
- "narrative": 1-2 sentences on what the highlight reel shows and why it matches the request

Return ONLY a JSON object."""

        response = self.call_llm(prompt, system=self.SYSTEM)
        result = self.extract_json(response)

        # LLM sometimes returns the highlights array directly instead of
        # wrapping it in {"highlights": [...]} — normalise either form.
        if isinstance(result, list):
            result = {"highlights": result}

        highlights = result.get("highlights", [])

        # Normalise zoom decisions in code — never trust the LLM to stick to
        # the position vocabulary or level bounds.
        highlights = [_normalise_zoom(h) for h in highlights]

        # Clamp to video bounds
        highlights = _clamp_highlights(highlights, video_duration)

        # Merge clips that are <= 1s apart — avoids micro-cuts
        highlights = _merge_adjacent(highlights, gap_threshold=1.0)

        # Enforce duration limit in code — never rely on the LLM to do arithmetic.
        # Sort chronologically first, then greedily keep clips until budget is full.
        highlights = _enforce_duration_limit(highlights, max_dur)

        total = sum(h["end"] - h["start"] for h in highlights)
        result["highlights"] = highlights
        result["total_duration"] = round(total, 2)

        return result

    def validate(self, output: dict, max_duration: int = None, **kwargs) -> tuple:
        max_dur = max_duration or MAX_HIGHLIGHT_DURATION

        if "highlights" not in output:
            return False, "Missing 'highlights' key"
        if not output["highlights"]:
            return False, "No highlights selected"

        for h in output["highlights"]:
            if not all(k in h for k in ("start", "end")):
                return False, f"Highlight missing start/end: {h}"
            try:
                start, end = float(h["start"]), float(h["end"])
            except (TypeError, ValueError):
                return False, f"Non-numeric start/end: {h}"
            if end <= start:
                return False, f"Invalid time range [{start}, {end}]"
            if start < 0:
                return False, f"Negative start time: {start}"

        total = sum(float(h["end"]) - float(h["start"]) for h in output["highlights"])
        if total > max_dur:
            return False, f"Total {total:.1f}s exceeds max {max_dur}s"

        return True, f"{len(output['highlights'])} highlights, {total:.1f}s total"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_segments_text(segments: list) -> str:
    """
    Build a rich text block so the LLM sees full content context per segment,
    not just an importance number.
    """
    lines = []
    for s in segments:
        imp = _safe_float(s.get("importance", 0))
        sid = s.get("id", "?")
        start = s.get("start", 0)
        end = s.get("end", 0)
        text = s.get("text", "")
        reason = s.get("reason", "")
        focus = s.get("focus_position", "center")

        # Summarise the content: trim long descriptions but keep key signals
        content = str(text).strip()[:200]

        line = (
            f"[id={sid} | {start}s-{end}s | importance={imp:.2f} | focus={focus}]\n"
            f"  CONTENT: {content if content else '(no description available)'}\n"
        )
        if reason:
            line += f"  REASON : {reason[:150]}\n"
        lines.append(line)
    return "\n".join(lines)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_zoom(highlight: dict) -> dict:
    """
    Ensure every highlight has a well-formed "zoom" dict:
      {"position": one of FOCUS_POSITIONS, "level": 1.0-ZOOM_MAX_LEVEL}

    Falls back to a no-op zoom (center, 1.0) when the LLM omits the field,
    returns the wrong type, or picks an unrecognised position/level.
    """
    zoom = highlight.get("zoom")
    if not isinstance(zoom, dict):
        return {**highlight, "zoom": {"position": "center", "level": 1.0}}

    position = str(zoom.get("position", "center")).strip().lower()
    if position not in FOCUS_POSITIONS:
        position = "center"

    level = _safe_float(zoom.get("level", 1.0), default=1.0)
    level = max(1.0, min(level, ZOOM_MAX_LEVEL))

    return {**highlight, "zoom": {"position": position, "level": round(level, 2)}}


def _clamp_highlights(highlights: list, video_duration: float) -> list:
    clamped = []
    for h in highlights:
        try:
            start = max(0.0, float(h["start"]))
            end = min(float(h["end"]), video_duration)
        except (TypeError, ValueError):
            continue
        if end - start >= 0.5:
            clamped.append({**h, "start": round(start, 2), "end": round(end, 2)})
    return clamped


def _enforce_duration_limit(highlights: list, max_dur: float) -> list:
    """
    Greedily keep clips in chronological order until the duration budget is full.
    If a clip would push the total over the limit, trim it to fit the remaining
    budget (as long as the trimmed portion is >= 0.5s).

    This runs in code so validation never fails due to LLM arithmetic errors.
    """
    result = []
    total = 0.0
    for h in sorted(highlights, key=lambda x: float(x["start"])):
        clip_dur = float(h["end"]) - float(h["start"])
        remaining = max_dur - total
        if clip_dur <= remaining:
            result.append(h)
            total += clip_dur
        else:
            # Trim this clip to exactly fill the remaining budget
            if remaining >= 0.5:
                trimmed = {**h, "end": round(float(h["start"]) + remaining, 2)}
                result.append(trimmed)
            break
    return result


def _merge_adjacent(highlights: list, gap_threshold: float = 1.0) -> list:
    """
    Merge clips that are within gap_threshold seconds of each other.
    Prevents jarring micro-cuts when the LLM picks consecutive segments.
    """
    if not highlights:
        return highlights

    # Sort chronologically first
    sorted_h = sorted(highlights, key=lambda h: float(h["start"]))
    merged = [dict(sorted_h[0])]

    for curr in sorted_h[1:]:
        prev = merged[-1]
        gap = float(curr["start"]) - float(prev["end"])
        if gap <= gap_threshold:
            # Extend the previous clip to cover this one too
            prev["end"] = max(float(prev["end"]), float(curr["end"]))
            prev["id"] = "merged"
            prev["reason"] = prev.get("reason", "") + " | " + curr.get("reason", "")
        else:
            merged.append(dict(curr))

    return merged
