import logging

import numpy as np
import requests

from agents.base_agent import BaseAgent
from config import EMBED_MODEL, EMBED_WEIGHT, OLLAMA_BASE_URL

logger = logging.getLogger("orchestrator")


class ExplainerAgent(BaseAgent):
    """
    Builds semantic understanding of the video by:
      1. Sending the synchronized speech + visual timeline to the LLM for
         importance scoring and summary.
      2. Embedding each scene's rich description with nomic-embed-text.
      3. Embedding the user's style query (or a default "best highlights" query).
      4. Computing cosine similarity between each scene and the style embedding.
      5. Blending the LLM importance score with the embedding similarity to
         produce a final importance score grounded in actual content matching.

    If the embedding model is unavailable, step 2-5 are silently skipped and
    the pipeline continues with LLM-only scores.
    """

    SYSTEM = (
        "You are an expert video content analyst. "
        "You analyze synchronized speech and visual scene data to understand video content. "
        "Always respond with ONLY a valid JSON object — no prose, no markdown fences."
    )

    def run(
        self,
        transcript: dict,
        video_type: str = None,
        style: str = None,
        feedback: str = "",
    ) -> dict:
        audio_segs = transcript.get("segments", [])
        visual_segs = transcript.get("visual_segments", [])

        # ------------------------------------------------------------------
        # Build synchronized timeline for the LLM
        # ------------------------------------------------------------------

        def visual_desc_at(start: float, end: float) -> str:
            overlapping = [
                v.get("description", "")
                for v in visual_segs
                if v["start"] < end and v["end"] > start
                and v.get("description", "").strip()
            ]
            return overlapping[0] if overlapping else ""

        all_entries = []

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

        speech_times = {(s["start"], s["end"]) for s in audio_segs}
        for v in visual_segs:
            has_speech = any(
                s_start < v["end"] and s_end > v["start"]
                for s_start, s_end in speech_times
            )
            if not has_speech:
                desc = v.get("description", "").strip()
                if not desc:
                    continue
                all_entries.append({
                    "start": float(v["start"]),
                    "label": f"[VISUAL {v['start']}s-{v['end']}s]: {desc}",
                })

        all_entries.sort(key=lambda e: e["start"])
        content_block = "\n".join(e["label"] for e in all_entries)

        # Visual-dominant video with no real visual descriptions (vision disabled)
        # — fall back to time-proportional importance so the pipeline can still
        # produce a reasonable highlight rather than analyzing garbage transcript.
        is_visual_dominant = transcript.get("is_visual_dominant", False)
        if not content_block.strip() or (is_visual_dominant and not any(
            v.get("description", "").strip() for v in visual_segs
        )):
            logger.info(
                "  [ExplainerAgent] no analyzable content — "
                "using time-proportional segment scoring"
            )
            return self._time_proportional_result(transcript, style)

        feedback_block = f"\nPREVIOUS ATTEMPT FEEDBACK:\n{feedback}\n" if feedback else ""
        video_type_block = (
            f"\nVideo type (user-specified): {video_type}\n"
            f"Interpret and score all segments through the lens of this video type.\n"
            if video_type else ""
        )

        prompt = f"""Analyze this video's synchronized speech and visual content.
Each SPEECH line shows what was said and (when available) what was visually on screen.
Each VISUAL line shows a scene with no concurrent speech.
{video_type_block}
IMPORTANT: All "start" and "end" values must be plain numbers (e.g. 4.5 not "4.5s").
Importance must be a float 0.0-1.0 (e.g. 0.8 not "high").

Return a JSON object with:
- "summary": 2-3 sentences synthesizing what was said AND what was seen
- "topics": list of main topics (strings)
- "tone": overall tone (e.g. comedic, musical, action, educational, emotional)
- "pacing": visual rhythm (e.g. fast cuts, slow cinematic, static talking-head)
- "key_moments": most impactful moments, each with "start", "end", "description"
- "segments": ALL speech and standalone visual entries below, each with:
    - "id": original id (integer for speech, "v0"/"v1"/etc for visual)
    - "start": start seconds
    - "end": end seconds
    - "text": the speech quote or visual description
    - "importance": 0.0-1.0 (1.0 = must be in the highlight reel)
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

        response = self.call_llm(prompt, system=self.SYSTEM)
        result = self.extract_json(response)

        result["duration"] = transcript.get("duration", 0)
        result["segments"] = self._merge_all_segments(
            audio_segs, visual_segs, result.get("segments", [])
        )

        # ------------------------------------------------------------------
        # Embedding-based similarity scoring
        # Embed each segment's rich description and the style/query, then
        # blend cosine similarity into the final importance score.
        # ------------------------------------------------------------------
        result["segments"] = self._enrich_with_embeddings(
            segments=result["segments"],
            style=style,
            summary=result.get("summary", ""),
        )

        return result

    # ------------------------------------------------------------------
    # Embedding methods
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list:
        """Get a dense embedding vector from Ollama (nomic-embed-text)."""
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            logger.debug(f"  [ExplainerAgent] embedding failed: {e}")
            return []

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        if not a or not b:
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def _enrich_with_embeddings(
        self, segments: list, style: str, summary: str
    ) -> list:
        """
        For each segment embed its full description and compute cosine
        similarity against the style query.  Blend with the LLM score:

            final_importance = (1 - EMBED_WEIGHT) * llm_score
                             + EMBED_WEIGHT * embedding_similarity

        Silently returns the original list if embedding is unavailable.
        """
        # Build the query the user cares about
        if style:
            query = f"Video highlight preference: {style}"
        else:
            query = (
                f"The most important, representative, and engaging moments "
                f"from this video: {summary}"
            )

        logger.info(f"  [ExplainerAgent] embedding query: {query[:120]}")
        query_vec = self._embed(query)

        if not query_vec:
            logger.info("  [ExplainerAgent] embedding unavailable — using LLM scores only")
            return segments

        enriched = []
        for seg in segments:
            # Build a rich description string: text + reason + any visual detail
            description_parts = [
                str(seg.get("text", "")),
                str(seg.get("reason", "")),
            ]
            description = " ".join(p for p in description_parts if p.strip())

            seg_vec = self._embed(description) if description.strip() else []

            if seg_vec:
                sim = self._cosine_similarity(query_vec, seg_vec)
                llm_score = float(seg.get("importance", 0.5))
                blended = round(
                    (1 - EMBED_WEIGHT) * llm_score + EMBED_WEIGHT * sim, 3
                )
                enriched.append({
                    **seg,
                    "importance": blended,
                    "llm_score": round(llm_score, 3),
                    "embedding_score": round(sim, 3),
                    "reason": (
                        f"{seg.get('reason', '')} "
                        f"[embedding similarity to style: {sim:.2f}]"
                    ).strip(),
                })
            else:
                enriched.append(seg)

        scored = sum(1 for s in enriched if "embedding_score" in s)
        logger.info(
            f"  [ExplainerAgent] embedding scores computed for "
            f"{scored}/{len(enriched)} segments"
        )
        return enriched

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, output: dict, **kwargs) -> tuple:
        for field in ("summary", "topics", "tone", "segments"):
            if field not in output:
                return False, f"Missing required field: {field}"
        if not output["summary"].strip():
            return False, "summary is empty"
        if not output["segments"]:
            return False, "segments list is empty"

        for seg in output["segments"]:
            seg["importance"] = _normalise_importance(seg.get("importance"))

        return True, f"{len(output['segments'])} segments analyzed"

    # ------------------------------------------------------------------
    # Time-proportional fallback (visual-dominant, no vision)
    # ------------------------------------------------------------------

    def _time_proportional_result(self, transcript: dict, style: str = None) -> dict:
        """
        When the video has no usable speech or visual descriptions, divide it
        into segments and score them by timeline position.

        Importance curve — weighted toward the middle and end of the video
        where key moments (kisses, reactions, climax) typically occur:
          first quarter  : 0.50  (setup/context)
          second quarter : 0.70  (rising action)
          third quarter  : 0.90  (peak / climax)
          fourth quarter : 0.80  (resolution)
        """
        duration = float(transcript.get("duration", 0))
        visual_segs = transcript.get("visual_segments", [])

        def _position_importance(start: float, end: float) -> float:
            mid = (start + end) / 2
            ratio = mid / duration if duration > 0 else 0.5
            if ratio < 0.25:
                return 0.50
            elif ratio < 0.50:
                return 0.70
            elif ratio < 0.75:
                return 0.90
            else:
                return 0.80

        segments = []
        for v in visual_segs:
            imp = _position_importance(v["start"], v["end"])
            segments.append({
                "id": v["id"],
                "start": v["start"],
                "end": v["end"],
                "text": "",
                "importance": imp,
                "reason": (
                    f"Time-proportional score — no visual analysis available. "
                    f"Position: {v['start']:.1f}s-{v['end']:.1f}s of {duration:.1f}s total."
                ),
            })

        # Apply embedding enrichment if style is provided
        if style and segments:
            segments = self._enrich_with_embeddings(
                segments=segments,
                style=style,
                summary=f"video of {duration:.0f} seconds",
            )

        return {
            "summary": (
                f"Visual-dominant video ({duration:.0f}s). "
                "No speech detected and visual analysis is disabled. "
                "Segments are scored by timeline position."
            ),
            "topics": [],
            "tone": "unknown",
            "pacing": "unknown",
            "key_moments": [],
            "segments": sorted(segments, key=lambda x: float(x.get("start", 0))),
            "duration": duration,
            "is_visual_dominant": True,
        }

    # ------------------------------------------------------------------
    # Segment merging
    # ------------------------------------------------------------------

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
                if desc:
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
