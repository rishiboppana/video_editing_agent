from agents.base_agent import BaseAgent


class ExplainerAgent(BaseAgent):
    """Uses Ollama to semantically analyze the transcript and score every segment."""

    SYSTEM = (
        "You are an expert video content analyst. "
        "You analyze transcripts and return structured JSON. "
        "Always respond with ONLY a valid JSON object — no prose, no markdown fences."
    )

    def run(self, transcript: dict, feedback: str = "") -> dict:
        audio_segs = transcript.get("segments", [])
        visual_segs = transcript.get("visual_segments", [])
        
        # Combine information for the LLM
        segs_info = []
        for s in audio_segs:
            segs_info.append(f"[AUDIO] {s['start']}s-{s['end']}s: {s['text']}")
        for v in visual_segs:
            segs_info.append(f"[VISUAL Scene Cut] {v['start']}s-{v['end']}s")
            
        segments_text = "\n".join(segs_info)

        feedback_block = ""
        if feedback:
            feedback_block = f"\n\nPREVIOUS ATTEMPT FEEDBACK (fix these issues):\n{feedback}\n"

        prompt = f"""Analyze the video content information below and return a JSON object with these exact keys:

- "summary": 2-3 sentence description of the content context (synthesis of audio and visual structure)
- "topics": list of main topics (strings)
- "tone": overall tone (e.g. comedic, musical, action, technical, informal)
- "pacing": description of the visual rhythm (e.g. fast-paced cuts, slow cinematic shots)
- "key_moments": list of potential highlights, each with "start", "end", "description"
- "segments": list covering ALL AUDIO segments plus key VISUAL segments, each with:
    - "id": segment id (e.g. integer for audio, "v0" etc for visual)
    - "start": start time in seconds
    - "end": end time in seconds
    - "text": description or transcription text
    - "importance": float 0.0–1.0 (1.0 = most important)
    - "reason": one sentence explaining the importance score

Overview:
Video Duration: {transcript.get('duration')}s
Number of speech segments: {len(audio_segs)}
Number of visual scene changes: {len(visual_segs)}

Content Details:
{segments_text}

{feedback_block}
Return ONLY a JSON object."""

        response = self.call_llm(prompt, system=self.SYSTEM)
        result = self.extract_json(response)

        # Merge segments to ensure original audio segments are preserved
        result["segments"] = self._merge_all_segments(audio_segs, visual_segs, result.get("segments", []))
        return result

    def validate(self, output: dict, **kwargs) -> tuple:
        for field in ("summary", "topics", "tone", "segments"):
            if field not in output:
                return False, f"Missing required field: {field}"
        return True, f"{len(output['segments'])} segments processed"

    def _merge_all_segments(self, audio: list, visual: list, scored: list) -> list:
        scored_map = {str(s.get("id")): s for s in scored}
        merged = []
        
        # Ensure all audio segments are included
        for a in audio:
            aid = str(a["id"])
            if aid in scored_map:
                merged.append(scored_map[aid])
            else:
                merged.append({**a, "importance": 0.5, "reason": "Default audio importance"})
                
        # Include high-importance visual segments or all if audio is sparse
        audio_coverage = sum(a["end"] - a["start"] for a in audio)
        is_sparse = audio_coverage < 5.0
        
        for v in visual:
            vid = str(v["id"])
            if vid in scored_map:
                # Only include visual if LLM scored it high or if audio is sparse
                if is_sparse or scored_map[vid].get("importance", 0) > 0.6:
                    merged.append(scored_map[vid])
            elif is_sparse:
                merged.append({**v, "text": "[Visual Scene]", "importance": 0.4, "reason": "Synthetic visual filler"})
                
        return sorted(merged, key=lambda x: x["start"])

    def _merge_segments(self, originals: list, scored: list) -> list:
        scored_map = {s["id"]: s for s in scored if "id" in s}
        merged = []
        for orig in originals:
            if orig["id"] in scored_map:
                merged.append(scored_map[orig["id"]])
            else:
                merged.append({**orig, "importance": 0.3, "reason": "Not individually analyzed"})
        return merged
