from agents.base_agent import BaseAgent


class MusicRecommenderAgent(BaseAgent):
    """
    Recommends what kind of background music would fit the final highlight reel.

    This agent does NOT source, download, or attach any actual audio file —
    music licensing is outside its scope. It returns a structured
    recommendation (genre, mood, tempo, instrumentation) that the user can
    act on when picking a track from their own licensed music library.
    """

    SYSTEM = (
        "You are a professional music supervisor for short-form video content. "
        "You recommend background music styles based on a video's tone, pacing, and content. "
        "Always respond with ONLY a valid JSON object — no prose, no markdown fences."
    )

    def run(
        self,
        explained_data: dict,
        highlights_data: dict,
        style: str = None,
        feedback: str = "",
    ) -> dict:
        summary = explained_data.get("summary", "unknown")
        tone = explained_data.get("tone", "unknown")
        pacing = explained_data.get("pacing", "unknown")
        topics = explained_data.get("topics", [])
        narrative = highlights_data.get("narrative", "")
        total_duration = highlights_data.get("total_duration", 0)

        clip_lines = "\n".join(
            f"- [{h.get('start')}s-{h.get('end')}s]: {h.get('reason', '')}"
            for h in highlights_data.get("highlights", [])
        )

        style_block = (
            f"\nUser's requested style/mood for the reel: \"{style}\"\n"
            "Let this strongly influence the music genre and mood you recommend.\n"
            if style else ""
        )

        feedback_block = (
            f"\nPREVIOUS ATTEMPT FEEDBACK:\n{feedback}\n" if feedback else ""
        )

        prompt = f"""Recommend background music for this {total_duration}-second highlight reel.

VIDEO CONTEXT:
  Summary  : {summary}
  Tone     : {tone}
  Pacing   : {pacing}
  Topics   : {', '.join(topics) if topics else 'none'}
  Narrative: {narrative}
{style_block}
CLIPS IN THE REEL (in order):
{clip_lines or '(no clip detail available)'}

Return a JSON object with:
- "primary_genre": single best-fit music genre/style (e.g. "Upbeat Indie Pop", "Epic Orchestral", "Lo-fi Acoustic")
- "mood": 2-4 words describing the emotional feel (e.g. "joyful, triumphant")
- "tempo_bpm": suggested tempo range as a string (e.g. "100-120 BPM")
- "instrumentation": key instruments/sounds that would fit (e.g. "acoustic guitar, light percussion, strings")
- "energy_curve": one sentence on how the music should build or change across the reel's duration
- "alternative_genres": list of 2 other genres that would also work
- "reference_style_examples": list of 2-3 well-known songs/artists ONLY as a STYLE reference
  (the user must source their own licensed track — these are not to be downloaded or implied as included)
- "reasoning": one or two sentences on why this music style fits the video's content and tone

Return ONLY a JSON object."""

        response = self.call_llm(prompt, system=self.SYSTEM)
        return self.extract_json(response)

    def validate(self, output: dict, **kwargs) -> tuple:
        if not isinstance(output, dict):
            return False, "Output must be a dict"

        required = ("primary_genre", "mood", "tempo_bpm", "instrumentation", "reasoning")
        for field in required:
            if field not in output or not str(output[field]).strip():
                return False, f"Missing or empty field: {field}"

        return True, f"Music recommendation: {output.get('primary_genre')}"
