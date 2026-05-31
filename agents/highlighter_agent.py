from agents.base_agent import BaseAgent
from config import MAX_HIGHLIGHT_DURATION


class HighlighterAgent(BaseAgent):
    """Uses Ollama to select the best highlight segments that form a coherent reel."""

    SYSTEM = (
        "You are an expert video editor specializing in highlight reels. "
        "You select segments that are engaging, meaningful, and tell a complete story. "
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

        # Cast importance to float safely so :.2f never crashes on string values
        segments_text = "\n".join(
            f"[id={s['id']} | {s['start']}s-{s['end']}s | importance={_safe_float(s.get('importance', 0)):.2f}]: "
            f"{s.get('text', '[Visual]')[:120]}"
            for s in explained_data.get("segments", [])
        )

        style_block = (
            f"\nUSER STYLE PREFERENCE: \"{style}\"\n"
            "Let this preference guide which segments you pick and why. "
            "Interpret it freely — match the mood, pacing, and focus the user described.\n"
            if style else ""
        )

        feedback_block = (
            f"\nPREVIOUS ATTEMPT FEEDBACK (fix these issues):\n{feedback}\n"
            if feedback else ""
        )

        prompt = f"""Select the best highlights to create a {max_dur}-second reel.
TARGET: fill at least {int(max_dur * 0.8)}s and no more than {max_dur}s.
VIDEO DURATION: {video_duration}s — do NOT use start/end times beyond this.

Video summary: {explained_data.get('summary', 'unknown')}
Tone   : {explained_data.get('tone', 'unknown')}
Pacing : {explained_data.get('pacing', 'unknown')}
{style_block}
Available segments:
{segments_text}

Rules:
1. Only use segment ids, start, and end times exactly as listed above.
2. start and end MUST be within 0–{video_duration}s.
3. Pick segments from the beginning, middle, AND end of the video.
4. If a style preference is given, it overrides importance scores.
5. Total duration must be between {int(max_dur * 0.8)}s and {max_dur}s.
{feedback_block}
Return a JSON object with:
- "highlights": list of selected segments, each with:
    - "id": the original segment id (copy exactly from the list above)
    - "start": start time in seconds
    - "end": end time in seconds
    - "reason": one sentence on why this segment was chosen
- "total_duration": sum of durations in seconds
- "narrative": 1-2 sentences describing what the highlight reel shows

Return ONLY a JSON object."""

        response = self.call_llm(prompt, system=self.SYSTEM)
        result = self.extract_json(response)

        # Clamp any hallucinated timestamps to actual video bounds
        result["highlights"] = _clamp_highlights(
            result.get("highlights", []), video_duration
        )

        # Recalculate total_duration — never trust LLM arithmetic
        total = sum(h["end"] - h["start"] for h in result["highlights"])
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
                return False, f"Non-numeric start/end in highlight: {h}"
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

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_highlights(highlights: list, video_duration: float) -> list:
    """Clamp start/end to [0, video_duration] and drop zero-length clips."""
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
