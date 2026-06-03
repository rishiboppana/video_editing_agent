from agents.base_agent import BaseAgent


class ExplainerAgent(BaseAgent):
    """
    Uses Ollama to build semantic understanding of the video.

    Key enhancement: for every speech segment the prompt now includes the
    visual description of what was concurrently on screen.  This lets the LLM
    reason about *what was said* and *what was seen* at the same moment rather
    than treating audio and visual as two separate streams.

    Timeline format sent to the LLM:
      [SPEECH 3.2s-7.8s | visual: person on stage gesturing to crowd]: "welcome everyone"
      [VISUAL 8.0s-14.5s]: Close-up of instruments being set up on stage.
    """

    SYSTEM = (
        "You are an expert video content analyst. "
        "You analyze synchronized speech and visual scene data to understand video content. "
        "Always respond with ONLY a valid JSON object — no prose, no markdown fences."
    )

    def run(self, transcript: dict, video_type: str = None, feedback: str = "") -> dict:
        audio_segs = transcript.get("segments", [])
        visual_segs = transcript.get("visual_segments", [])

        # Build a fast lookup: for any point in time, which visual description applies?
        # This lets us annotate every speech segment with what was on screen.
        def visual_desc_at(start: float, end: float) -> str:
            overlapping = [
                v.get("description", "")
                for v in visual_segs
                if v["start"] < end and v["end"] > start
                and v.get("description", "").strip()  # skip empty — no real content
            ]
            return overlapping[0] if overlapping else ""

        all_entries = []

        # Speech segments — annotated with the concurrent visual description
        for s in audio_segs:
            concurrent = visual_desc_at(s["start"], s["end"])
            if concurrent:
                label = (
                    f"[SPEECH {s['start']}s-{s['end']}s | visual: {concurrent}]: "
                    f"\"{s['text']}\""
                )
            else:
                label = f"[SPEECH {s['start']}s-{s['end']}s]: \"{s['text']}\""
            all_entries.append({"start": float(s["start"]), "label": label})

        # Visual-only segments (stand-alone scenes with no concurrent speech shown above)
        speech_times = {(s["start"], s["end"]) for s in audio_segs}
        for v in visual_segs:
            # Only show visual as a standalone line if it doesn't overlap with any speech
            has_speech = any(
                s_start < v["end"] and s_end > v["start"]
                for s_start, s_end in speech_times
            )
            if not has_speech:
                desc = v.get("description", "").strip()
                if not desc:  # skip segments with no real visual content
                    continue
                all_entries.append({
                    "start": float(v["start"]),
                    "label": f"[VISUAL {v['start']}s-{v['end']}s]: {desc}",
                })

        all_entries.sort(key=lambda e: e["start"])
        content_block = "\n".join(e["label"] for e in all_entries)

        feedback_block = f"\nPREVIOUS ATTEMPT FEEDBACK:\n{feedback}\n" if feedback else ""

        video_type_block = (
            f"\nVideo type (user-specified): {video_type}\n"
            f"Interpret and score all segments through the lens of this video type.\n"
            if video_type else ""
        )

        prompt = f"""Analyze this video's synchronized speech and visual content.
Each SPEECH line shows what was said and (when available) what was visually on screen at that moment.
Each VISUAL line shows a scene with no concurrent speech.
{video_type_block}
IMPORTANT: All "start" and "end" values must be plain numbers only (e.g. 4.5 not "4.5s").
Importance must be a float between 0.0 and 1.0 (e.g. 0.8 not "high").

Return a JSON object with:
- "summary": 2-3 sentences synthesizing what was said AND what was seen
- "topics": list of main topics (strings)
- "tone": overall tone (e.g. comedic, musical, action, educational, emotional)
- "pacing": visual rhythm (e.g. fast cuts, slow cinematic, static talking-head)
- "key_moments": most impactful moments, each with "start", "end", "description"
- "segments": ALL speech and standalone visual entries listed below, each with:
    - "id": original id (integer for speech, "v0"/"v1"/etc for visual)
    - "start": start seconds
    - "end": end seconds
    - "text": the speech quote or visual description
    - "importance": 0.0–1.0 (1.0 = must be in the highlight reel)
    - "reason": one sentence on why this score

Video overview:
  Duration   : {transcript.get('duration', '?')}s
  Speech     : {len(audio_segs)} segments
  Visual     : {len(visual_segs)} scenes
  Video type : {video_type or 'not specified'}

Synchronized timeline:
{content_block}
{feedback_block}
Return ONLY a JSON object."""

        # Guard: if there is truly nothing to analyse, return a minimal result
        # rather than sending an empty prompt to the LLM.
        if not content_block.strip():
            return {
                "summary": "No speech or visual content could be extracted from this video.",
                "topics": [],
                "tone": "unknown",
                "pacing": "unknown",
                "key_moments": [],
                "segments": [],
                "duration": transcript.get("duration", 0),
            }

        response = self.call_llm(prompt, system=self.SYSTEM)
        result = self.extract_json(response)

        # Always carry duration forward so the highlighter can clamp timestamps
        result["duration"] = transcript.get("duration", 0)

        result["segments"] = self._merge_all_segments(
            audio_segs, visual_segs, result.get("segments", [])
        )
        return result

    def validate(self, output: dict, **kwargs) -> tuple:
        for field in ("summary", "topics", "tone", "segments"):
            if field not in output:
                return False, f"Missing required field: {field}"
        if not output["summary"].strip():
            return False, "summary is empty"
        if not output["segments"]:
            return False, "segments list is empty"

        # Auto-heal bad importance values rather than rejecting the whole output.
        # The LLM sometimes returns "high"/"low"/null — normalise them in place.
        for seg in output["segments"]:
            seg["importance"] = _normalise_importance(seg.get("importance"))

        return True, f"{len(output['segments'])} segments analyzed"

    def _merge_all_segments(self, audio: list, visual: list, scored: list) -> list:
        scored_map = {str(s.get("id")): s for s in scored}
        merged = []

        for seg in audio:
            key = str(seg["id"])
            if key in scored_map:
                entry = dict(scored_map[key])
                entry["importance"] = _normalise_importance(entry.get("importance"))
                merged.append(entry)
            else:
                merged.append({**seg, "importance": 0.4, "reason": "Not individually analyzed"})

        audio_duration = sum(s["end"] - s["start"] for s in audio)
        is_sparse_audio = audio_duration < 5.0

        for seg in visual:
            key = str(seg["id"])
            if key in scored_map:
                entry = dict(scored_map[key])
                entry["importance"] = _normalise_importance(entry.get("importance"))
                if is_sparse_audio or entry["importance"] >= 0.5:
                    merged.append(entry)
            elif is_sparse_audio:
                desc = seg.get("description", "").strip()
                if desc:  # only include if there is actual visual content
                    merged.append({
                        **seg,
                        "text": desc,
                        "importance": 0.4,
                        "reason": "Visual segment in sparse-audio video",
                    })

        return sorted(merged, key=lambda x: float(x.get("start", 0)))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_TEXT_TO_IMPORTANCE = {
    "low": 0.2, "medium": 0.5, "moderate": 0.5,
    "high": 0.8, "very high": 0.95, "critical": 1.0,
}

def _normalise_importance(value) -> float:
    """
    Convert whatever the LLM returned for importance into a valid float 0–1.
    Handles: float, int, numeric string, text labels, None.
    """
    if value is None:
        return 0.4
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    s = str(value).strip().lower()
    if s in _TEXT_TO_IMPORTANCE:
        return _TEXT_TO_IMPORTANCE[s]
    try:
        return max(0.0, min(1.0, float(s)))
    except ValueError:
        return 0.4
