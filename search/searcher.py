"""
Semantic search over the indexed scene embeddings.

Embeds the user's query with nomic-embed-text, computes cosine similarity
against every stored scene embedding, and returns the top-K matches with
their video path, exact timeframe, description, and similarity score.
"""
import logging
from typing import Optional

import numpy as np
import requests

from config import EMBED_MODEL, OLLAMA_BASE_URL
from search.store import load_index, DEFAULT_INDEX_PATH

logger = logging.getLogger("search.searcher")


def search(
    query: str,
    index_path: str = DEFAULT_INDEX_PATH,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list:
    """
    Search indexed scenes for the given query.

    Returns a list of dicts sorted by similarity (highest first):
    {
      "rank"         : 1,
      "score"        : 0.87,
      "video"        : "videos/Concert.mp4",
      "start"        : 8.3,
      "end"          : 14.5,
      "frame_time"   : 11.4,
      "description"  : "Performer on stage ...",
      "video_duration": 37.5,
    }
    """
    index = load_index(index_path)
    scenes = [s for s in index["scenes"] if s.get("embedding")]

    if not scenes:
        logger.warning("No embedded scenes found in index. Run 'index' first.")
        return []

    # Embed the query
    query_vec = _embed(query)
    if not query_vec:
        logger.error("Failed to embed query — is Ollama running with nomic-embed-text?")
        return []

    # Stack all scene embeddings into a matrix for batch cosine similarity
    query_arr = np.array(query_vec, dtype=np.float32)
    scene_vecs = np.array([s["embedding"] for s in scenes], dtype=np.float32)

    # Cosine similarity: (A · B) / (|A| |B|)
    query_norm = np.linalg.norm(query_arr)
    scene_norms = np.linalg.norm(scene_vecs, axis=1)

    valid = scene_norms > 0
    scores = np.zeros(len(scenes))
    scores[valid] = (scene_vecs[valid] @ query_arr) / (scene_norms[valid] * query_norm)

    # Rank and filter
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score < min_score:
            break
        if len(results) >= top_k:
            break
        scene = scenes[idx]
        results.append({
            "rank": len(results) + 1,
            "score": round(score, 4),
            "video": scene["video"],
            "start": scene["start"],
            "end": scene["end"],
            "frame_time": scene.get("frame_time"),
            "description": scene.get("description", ""),
            "video_duration": scene.get("video_duration", 0),
            "phash": scene.get("phash", ""),
        })

    return results


def _embed(text: str) -> list:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("embedding", [])
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []


def format_results(results: list, query: str) -> str:
    """Pretty-print search results."""
    if not results:
        return f'No results found for: "{query}"'

    lines = [f'\nSearch results for: "{query}"\n' + "=" * 60]
    for r in results:
        duration = r["end"] - r["start"]
        lines.append(
            f"\n#{r['rank']}  score={r['score']:.3f}\n"
            f"  Video     : {r['video']}\n"
            f"  Timeframe : {r['start']}s – {r['end']}s  ({duration:.1f}s)\n"
            f"  Description: {r['description'][:200] if r['description'] else '(no description)'}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)
