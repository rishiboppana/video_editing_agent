from agents.base_agent import BaseAgent
from config import MAX_HIGHLIGHT_DURATION


class HighlighterAgent(BaseAgent):
    """Uses Ollama to select the best highlight segments that form a coherent reel."""

    SYSTEM = (
        "You are an expert video editor specializing in highlight reels. "
        "You select segments that are engaging, meaningful, and tell a complete story. "
        "Always respond with ONLY a valid JSON object — no prose, no markdown fences."
    )

    def run(self, explained_data: dict, max_duration: int = None, feedback: str = "") -> dict:
        max_dur = max_duration or MAX_HIGHLIGHT_DURATION

        segments_text = "\n".join(
            f"[id={s['id']} | {s['start']}s-{s['end']}s | importance={s['importance']:.2f}]: {s.get('text', '[Visual]')}"
            for s in explained_data["segments"]
        )

        feedback_block = ""
        if feedback:
            feedback_block = f"\n\nPREVIOUS ATTEMPT FEEDBACK (fix these issues):\n{feedback}\n"

        prompt = f"""Select the best highlights to create a {max_dur}-second reel.
TARGET DURATION: {max_dur} seconds. (Fill at least {int(max_dur * 0.8)}s)

Video summary: {explained_data['summary']}
Tone: {explained_data.get('tone', 'unknown')}
Pacing: {explained_data.get('pacing', 'unknown')}

Available segments (Audio and Visual):
{segments_text}

Rules:
1. SELECT ENOUGH SEGMENTS TO FILL {max_dur} SECONDS (or as close as possible).
2. If total duration of high-importance segments is less than {max_dur}s, you MUST select lower-importance segments to fill the time.
3. PICK AT LEAST ONE SEGMENT FROM EACH THIRD OF THE VIDEO (Beginning, Middle, End).
4. If this is a visual-centric video (few audio segments), prioritize visually distinct [VISUAL] segments.
5. Do not exceed {max_dur}s.
{feedback_block}
Return a JSON object with:
- "highlights": list of selected segments, each with:
    - "id": the original segment id
    - "start": start time in seconds
    - "end": end time in seconds
    - "reason": why this segment contributes to the {max_dur}s target
- "total_duration": sum of durations (float, in seconds)
- "narrative": summary of the highlight reel narrative

Return ONLY a JSON object."""

        response = self.call_llm(prompt, system=self.SYSTEM)
        result = self.extract_json(response)

        # Recalculate total_duration from actual segment times (don't trust LLM math)
        highlights = result.get("highlights", [])
        total = sum(h["end"] - h["start"] for h in highlights)
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
            if h["end"] <= h["start"]:
                return False, f"Invalid time range [{h['start']}, {h['end']}]"

        total = sum(h["end"] - h["start"] for h in output["highlights"])
        if total > max_dur:
            return False, f"Total duration {total:.1f}s exceeds max {max_dur}s"

        return True, f"{len(output['highlights'])} highlights, {total:.1f}s total"
