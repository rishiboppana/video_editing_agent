import json
import logging

from agents.base_agent import BaseAgent
from agents.editor_agent import EditorAgent
from agents.explainer_agent import ExplainerAgent
from agents.highlighter_agent import HighlighterAgent
from agents.transcriber_agent import TranscriberAgent
from config import MAX_HIGHLIGHT_DURATION, MAX_RETRIES, OLLAMA_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-12s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")

DIVIDER = "=" * 64


class OrchestratorAgent(BaseAgent):
    """
    Master controller.  Runs each specialist agent in sequence, validates
    every output both structurally and semantically, and re-runs an agent
    with targeted feedback whenever the output is not satisfactory.
    """

    REVIEW_SYSTEM = (
        "You are a strict quality-control reviewer for AI video-editing agents. "
        "Respond with ONLY a valid JSON object — no prose, no markdown fences."
    )

    def __init__(self):
        super().__init__(OLLAMA_MODEL)
        self.transcriber = TranscriberAgent()
        self.explainer = ExplainerAgent()
        self.highlighter = HighlighterAgent()
        self.editor = EditorAgent()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        output_path: str,
        max_duration: int = None,
        style: str = None,
        video_type: str = None,
    ) -> dict:
        max_dur = max_duration or MAX_HIGHLIGHT_DURATION
        logger.info(DIVIDER)
        logger.info(f"ORCHESTRATOR  video={video_path}  out={output_path}  max={max_dur}s")
        if video_type:
            logger.info(f"  video_type : {video_type}")
        if style:
            logger.info(f"  style      : {style}")
        logger.info(DIVIDER)

        # ── Step 1: Transcription ───────────────────────────────────────
        logger.info("\n[STEP 1/4]  TRANSCRIPTION")
        transcript = self._run_with_retry(
            agent=self.transcriber,
            run_kwargs={"video_path": video_path},
            review_context=f"Transcribing video '{video_path}' with Whisper.",
        )
        logger.info(
            f"  → {len(transcript['segments'])} segments | lang={transcript.get('language')} | "
            f"preview: {transcript['full_text'][:120]}…"
        )

        # ── Step 2: Explanation ─────────────────────────────────────────
        logger.info("\n[STEP 2/4]  EXPLANATION")
        explanation = self._run_with_retry(
            agent=self.explainer,
            run_kwargs={"transcript": transcript, "video_type": video_type, "style": style},
            review_context=(
                f"Analyzing {len(transcript['segments'])} segments "
                f"of a '{video_type or 'unknown'}' video. Style: '{style or 'none'}'."
            ),
        )
        logger.info(f"  → summary: {explanation.get('summary', '')[:160]}")
        logger.info(f"  → topics : {explanation.get('topics', [])}")
        logger.info(f"  → tone   : {explanation.get('tone', 'unknown')}")

        # ── Step 3: Highlight selection ─────────────────────────────────
        logger.info("\n[STEP 3/4]  HIGHLIGHT SELECTION")
        highlights_data = self._run_with_retry(
            agent=self.highlighter,
            run_kwargs={
                "explained_data": explanation,
                "max_duration": max_dur,
                "style": style,
            },
            validate_kwargs={"max_duration": max_dur},
            review_context=(
                f"Selecting highlights from {len(explanation['segments'])} segments. "
                f"Max {max_dur}s. Style: '{style or 'none'}'. "
                f"Summary: {explanation.get('summary', '')[:200]}"
            ),
        )
        logger.info(
            f"  → {len(highlights_data['highlights'])} clips  "
            f"| {highlights_data['total_duration']}s total"
        )
        logger.info(f"  → narrative: {highlights_data.get('narrative', '')[:160]}")

        # ── Step 4: Video editing ───────────────────────────────────────
        ordered_highlights = sorted(
            highlights_data["highlights"], key=lambda h: h["start"]
        )

        logger.info("\n[STEP 4/4]  VIDEO EDITING")
        edit_result = self._run_with_retry(
            agent=self.editor,
            run_kwargs={
                "video_path": video_path,
                "highlights": ordered_highlights,
                "output_path": output_path,
            },
            review_context=(
                f"Cutting {len(ordered_highlights)} clips and joining into {output_path}."
            ),
        )
        logger.info(f"  → output  : {edit_result['output_path']}")
        logger.info(f"  → duration: {edit_result['duration']}s")

        logger.info(f"\n{DIVIDER}")
        logger.info("ORCHESTRATOR  pipeline complete")
        logger.info(DIVIDER)

        return {
            "status": "success",
            "output_path": edit_result["output_path"],
            "duration": edit_result["duration"],
            "clips_count": len(ordered_highlights),
            "summary": explanation.get("summary"),
            "topics": explanation.get("topics", []),
            "tone": explanation.get("tone"),
            "style": style,
            "video_type": video_type,
            "narrative": highlights_data.get("narrative"),
            "pipeline": {
                "transcript": transcript,
                "explanation": explanation,
                "highlights": highlights_data,
            },
        }

    # ------------------------------------------------------------------
    # Core retry loop
    # ------------------------------------------------------------------

    def _run_with_retry(
        self,
        agent: BaseAgent,
        run_kwargs: dict,
        validate_kwargs: dict = None,
        review_context: str = "",
    ) -> dict:
        validate_kwargs = validate_kwargs or {}
        feedback = ""

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(f"  [{agent.name}] attempt {attempt}/{MAX_RETRIES}")

            # Inject orchestrator feedback into the agent's prompt on retries
            if feedback:
                run_kwargs = {**run_kwargs, "feedback": feedback}

            try:
                output = agent.run(**run_kwargs)
            except Exception as exc:
                feedback = f"The agent raised an exception: {exc}. Ensure valid output."
                logger.warning(f"  [{agent.name}] exception: {exc}")
                continue

            # ── 1. Structural validation ────────────────────────────────
            ok, msg = agent.validate(output, **validate_kwargs)
            if not ok:
                feedback = f"Structural validation failed: {msg}. Fix all issues."
                logger.warning(f"  [{agent.name}] structural FAIL: {msg}")
                continue
            logger.info(f"  [{agent.name}] structural OK  — {msg}")

            # ── 2. Semantic review by orchestrator ──────────────────────
            satisfied, review_feedback = self._review_output(
                agent.name, output, review_context
            )
            if not satisfied:
                logger.warning(f"  [{agent.name}] review FAIL: {review_feedback}")
                feedback = review_feedback
                if attempt < MAX_RETRIES:
                    continue
                # On final attempt: accept best-effort result with a warning
                logger.warning(
                    f"  [{agent.name}] max retries reached — accepting best-effort output"
                )

            logger.info(f"  [{agent.name}] accepted on attempt {attempt}")
            return output

        raise RuntimeError(
            f"[{agent.name}] failed to produce satisfactory output after "
            f"{MAX_RETRIES} attempts. Last feedback: {feedback}"
        )

    # ------------------------------------------------------------------
    # LLM-powered semantic review
    # ------------------------------------------------------------------

    def _review_output(self, agent_name: str, output: dict, context: str) -> tuple:
        # Truncate large outputs to stay within token budget
        output_preview = json.dumps(output, indent=2)
        if len(output_preview) > 3000:
            output_preview = output_preview[:3000] + "\n... (truncated)"

        prompt = f"""Review the output of the "{agent_name}" step in a video highlight pipeline.

Pipeline context:
{context}

Agent output:
{output_preview}

Evaluate:
1. Is the output complete and internally consistent?
2. Does it make sense given the context?
3. Are there any obvious errors, missing data, or quality problems?

Return a JSON object with:
- "satisfactory": true or false
- "score": quality score 0.0–1.0
- "issues": list of specific problems found (empty list if none)
- "feedback": actionable improvement instructions (empty string if satisfactory)

Return ONLY a JSON object."""

        try:
            response = self.call_llm(prompt, system=self.REVIEW_SYSTEM)
            evaluation = self.extract_json(response)
            # LLM occasionally returns a JSON array instead of object — treat as satisfied
            if not isinstance(evaluation, dict):
                return True, ""
            satisfied = bool(evaluation.get("satisfactory", True))
            feedback = evaluation.get("feedback", "")
            score = evaluation.get("score", 1.0)
            issues = evaluation.get("issues", [])

            if issues:
                logger.info(
                    f"  [review/{agent_name}] score={score:.2f} "
                    f"issues={issues[:2]}"
                )
            return satisfied, feedback

        except Exception as exc:
            # If the reviewer itself fails, don't block the pipeline
            logger.warning(f"  [review/{agent_name}] reviewer error ({exc}) — defaulting to satisfied")
            return True, ""
