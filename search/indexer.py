"""
Video indexer — walks a folder, detects scenes in every video, extracts
unique frames via pHash, describes them with LLaVA, and embeds the
descriptions with nomic-embed-text.
"""
import base64
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import requests

from config import (
    ENABLE_VISION,
    EMBED_MODEL,
    FFMPEG_PATH,
    FFPROBE_PATH,
    OLLAMA_BASE_URL,
    VISION_MODEL,
    VISION_TIMEOUT,
)
from search.phash_utils import compute_phash, deduplicate_scenes
from search.store import (
    add_scene,
    is_video_indexed,
    load_index,
    mark_video_indexed,
    remove_video_scenes,
    save_index,
    DEFAULT_INDEX_PATH,
)

logger = logging.getLogger("search.indexer")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "0.15"))


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def index_folder(
    folder: str,
    index_path: str = DEFAULT_INDEX_PATH,
    force: bool = False,
    vision: bool = None,
) -> dict:
    """
    Walk folder (recursively) and index every video found.
    Skips already-indexed videos unless force=True.
    Returns the final index dict.
    """
    use_vision = vision if vision is not None else ENABLE_VISION
    index = load_index(index_path)
    index["embed_model"] = EMBED_MODEL
    index["vision_model"] = VISION_MODEL if use_vision else "none"

    video_paths = _find_videos(folder)
    logger.info(f"Found {len(video_paths)} video(s) in '{folder}'")

    for video_path in video_paths:
        if not force and is_video_indexed(index, video_path):
            logger.info(f"  SKIP (already indexed): {video_path}")
            continue

        logger.info(f"  INDEXING: {video_path}")
        if force:
            remove_video_scenes(index, video_path)

        try:
            scenes_added = _index_video(video_path, index, use_vision)
            mark_video_indexed(index, video_path)
            save_index(index, index_path)
            logger.info(f"  → {scenes_added} unique scenes indexed")
        except Exception as e:
            logger.error(f"  ERROR indexing {video_path}: {e}")

    return index


def index_single_video(
    video_path: str,
    index_path: str = DEFAULT_INDEX_PATH,
    force: bool = False,
    vision: bool = None,
) -> int:
    """Index a single video file. Returns the number of scenes added."""
    use_vision = vision if vision is not None else ENABLE_VISION
    index = load_index(index_path)

    if not force and is_video_indexed(index, video_path):
        logger.info(f"Already indexed: {video_path}")
        return 0

    if force:
        remove_video_scenes(index, video_path)

    count = _index_video(video_path, index, use_vision)
    mark_video_indexed(index, video_path)
    save_index(index, index_path)
    return count


# ------------------------------------------------------------------
# Core indexing logic
# ------------------------------------------------------------------

def _index_video(video_path: str, index: dict, use_vision: bool) -> int:
    duration = _get_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"Could not determine duration of {video_path}")

    # 1. Detect scene boundaries
    scenes = _detect_scenes(video_path, duration)
    logger.info(f"    {len(scenes)} scenes detected")

    # 2. Extract one representative frame per scene + compute pHash
    frames_and_hashes = []
    temp_files = []
    try:
        for scene in scenes:
            mid = round((scene["start"] + scene["end"]) / 2, 2)
            frame_path = tempfile.mktemp(suffix=".jpg")
            temp_files.append(frame_path)
            try:
                _extract_frame(video_path, mid, frame_path)
                phash = compute_phash(frame_path)
                frames_and_hashes.append((scene, mid, frame_path, phash))
            except Exception as e:
                logger.warning(f"    Frame extraction failed at {mid}s: {e}")

        # 3. Deduplicate using pHash
        scenes_list = [x[0] for x in frames_and_hashes]
        hashes_list = [x[3] for x in frames_and_hashes]
        unique_pairs = deduplicate_scenes(scenes_list, hashes_list)
        unique_ids = {s["id"] for s, _ in unique_pairs}
        logger.info(f"    {len(unique_pairs)} unique scenes after pHash dedup "
                    f"(removed {len(scenes) - len(unique_pairs)} near-duplicates)")

        count = 0
        for scene, mid, frame_path, phash in frames_and_hashes:
            if scene["id"] not in unique_ids:
                continue

            # 4. Describe the frame with LLaVA (if vision enabled)
            description = ""
            if use_vision and os.path.exists(frame_path):
                description = _describe_frame(frame_path, scene["start"], scene["end"])

            # 5. Embed the description (or a scene-position fallback)
            embed_text = description if description.strip() else (
                f"Video scene from {scene['start']}s to {scene['end']}s"
            )
            embedding = _embed(embed_text)

            # 6. Store in index
            add_scene(index, {
                "video": video_path,
                "video_duration": duration,
                "scene_id": scene["id"],
                "start": scene["start"],
                "end": scene["end"],
                "frame_time": mid,
                "phash": phash,
                "description": description,
                "embedding": embedding,
            })
            count += 1

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

    return count


# ------------------------------------------------------------------
# ffmpeg helpers (self-contained, no agent imports)
# ------------------------------------------------------------------

def _get_duration(video_path: str) -> float:
    cmd = [
        FFPROBE_PATH, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _detect_scenes(video_path: str, duration: float) -> list:
    """Detect scene cuts using ffmpeg showinfo. Falls back to adaptive chunks."""
    for threshold in [SCENE_THRESHOLD, 0.08]:
        try:
            segments = _run_scene_detection(video_path, duration, threshold)
            if len(segments) > 1:
                return segments
        except Exception:
            pass
    return _adaptive_chunks(duration)


def _run_scene_detection(video_path: str, duration: float, threshold: float) -> list:
    cmd = [
        FFMPEG_PATH, "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "0", "-an", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    scene_times = [0.0]
    for line in result.stderr.splitlines():
        if "pts_time:" in line and "iskey:" in line:
            m = re.search(r"pts_time:(\d+\.?\d*)", line)
            if m:
                t = float(m.group(1))
                if t > 0.1:
                    scene_times.append(t)
    points = sorted(set(scene_times + [duration]))
    segments = []
    for i in range(len(points) - 1):
        s, e = points[i], points[i + 1]
        if e - s >= 1.0:
            segments.append({"id": f"v{i}", "start": round(s, 2), "end": round(e, 2)})
    return segments


def _adaptive_chunks(duration: float) -> list:
    chunk = max(3.0, duration / 8.0)
    segments, idx, t = [], 0, 0.0
    while t < duration:
        end = min(round(t + chunk, 2), round(duration, 2))
        if end - t >= 1.0:
            segments.append({"id": f"v{idx}", "start": round(t, 2), "end": end})
            idx += 1
        t += chunk
    return segments


def _extract_frame(video_path: str, timestamp: float, out_path: str):
    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "3",
        "-vf", "scale=640:-1",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"Frame extraction failed at {timestamp}s")


# ------------------------------------------------------------------
# LLaVA description
# ------------------------------------------------------------------

def _describe_frame(frame_path: str, start: float, end: float) -> str:
    with open(frame_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        f"Describe this video frame (scene: {start}s-{end}s) in detail. Cover:\n"
        "PEOPLE: Who is present? Their appearance, expression, action.\n"
        "OBJECTS: Notable objects, props, text visible.\n"
        "SETTING: Location, background, lighting.\n"
        "ACTIONS: What is happening?\n"
        "EMOTIONS: Overall mood and energy level.\n"
        "Be specific and factual."
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": VISION_MODEL,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False,
            },
            timeout=VISION_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"    LLaVA description failed: {e}")
        return ""


# ------------------------------------------------------------------
# Embedding
# ------------------------------------------------------------------

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
        logger.warning(f"    Embedding failed: {e}")
        return []


# ------------------------------------------------------------------
# Folder walk
# ------------------------------------------------------------------

def _find_videos(folder: str) -> list:
    videos = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, f))
    return videos
