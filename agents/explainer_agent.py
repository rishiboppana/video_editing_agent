from agents.base_agent import BaseAgent


class ExplainerAgent(BaseAgent):
    """
    Uses Ollama to build a semantic understanding of the video by combining:
      - speech transcription (what was said)
      - visual descriptions (what was seen, per scene)
    Produces a summary, topics, tone, pacing, and per-segment importance scores.
    """

    SYSTEM = (
        "You are an expert video content analyst. "
        "You analyze both speech transcripts and visual scene descriptions to understand video content. "
        "Always respond with ONLY a valid JSON object — no prose, no markdown fences."
    )

    def run(self, transcript: dict, feedback: str = "") -> dict:
        audio_segs = transcript.get("segments", [])
        visual_segs = transcript.get("visual_segments", [])

        # Combine audio + visual entries and sort by start time (float) BEFORE
        # formatting into strings — avoids any fragile string-parsing of timestamps.
        all_entries = []
        for s in audio_segs:
            all_entries.append({
                "start": float(s["start"]),
                "label": f"[SPEECH {s['start']}s-{s['end']}s]: {s['text']}",
            })
        for v in visual_segs:
            desc = v.get("description", "[no description]")
            all_entries.append({
                "start": float(v["start"]),
                "label": f"[VISUAL {v['start']}s-{v['end']}s]: {desc}",
            })

        all_entries.sort(key=lambda e: e["start"])
        content_block = "\n".join(e["label"] for e in all_entries)

        feedback_block = f"\nPREVIOUS ATTEMPT FEEDBACK:\n{feedback}\n" if feedback else ""

        prompt = f"""You have both the speech transcript and visual scene descriptions of a video.
Analyze the combined audio-visual content and return a JSON object with these exact keys:

- "summary": 2-3 sentence description synthesizing what was said AND what was seen
- "topics": list of main topics (strings)
- "tone": overall tone (e.g. comedic, musical, action, educational, emotional, informal)
- "pacing": description of visual rhythm (e.g. fast-paced cuts, slow cinematic, talking-head)
- "key_moments": list of the most impactful moments, each with "start", "end", "description"
- "segments": list covering ALL entries below, each with:
    - "id": the original segment id (integer for speech, "v0"/"v1" etc for visual)
    - "start": start time in seconds
    - "end": end time in seconds
    - "text": the speech text, or the visual description
    - "importance": float 0.0–1.0 (1.0 = most important for a highlight reel)
    - "reason": one sentence explaining why this segment has this importance score

Video overview:
  Duration  : {transcript.get('duration', '?')}s
  Speech    : {len(audio_segs)} segments
  Visual    : {len(visual_segs)} scene segments

Timeline (chronological):
{content_block}
{feedback_block}
Return ONLY a JSON object."""

        response = self.call_llm(prompt, system=self.SYSTEM)
        result = self.extract_json(response)

        # Guarantee all original segments survive even if the LLM skipped them
        result["segments"] = self._merge_all_segments(audio_segs, visual_segs, result.get("segments", []))
        return result

    def validate(self, output: dict, **kwargs) -> tuple:
        for field in ("summary", "topics", "tone", "segments"):
            if field not in output:
                return False, f"Missing required field: {field}"
        if not output["summary"].strip():
            return False, "summary is empty"
        if not output["segments"]:
            return False, "segments list is empty"
        for seg in output["segments"]:
            if "importance" not in seg:
                return False, f"Segment {seg.get('id')} missing importance"
            try:
                score = float(seg["importance"])
            except (TypeError, ValueError):
                return False, f"Segment {seg.get('id')} has non-numeric importance"
            if not 0.0 <= score <= 1.0:
                return False, f"Segment {seg.get('id')} importance {score} out of [0,1]"
        return True, f"{len(output['segments'])} segments analyzed"

    def _merge_all_segments(self, audio: list, visual: list, scored: list) -> list:
        scored_map = {str(s.get("id")): s for s in scored}
        merged = []

        for seg in audio:
            key = str(seg["id"])
            merged.append(scored_map.get(key, {**seg, "importance": 0.4, "reason": "Not individually analyzed"}))

        audio_duration = sum(s["end"] - s["start"] for s in audio)
        is_sparse_audio = audio_duration < 5.0  # less than 5s of speech → visual-dominant video

        for seg in visual:
            key = str(seg["id"])
            if key in scored_map:
                scored_seg = scored_map[key]
                # Always include visual segments for sparse-audio videos;
                # otherwise only include if the LLM rated them important enough
                if is_sparse_audio or float(scored_seg.get("importance", 0)) >= 0.5:
                    merged.append(scored_seg)
            elif is_sparse_audio:
                desc = seg.get("description", "[visual scene]")
                merged.append({**seg, "text": desc, "importance": 0.4, "reason": "Visual-dominant video filler"})

        return sorted(merged, key=lambda x: float(x.get("start", 0)))
