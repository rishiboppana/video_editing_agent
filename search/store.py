"""
Index persistence — read and write the scene index to disk.

Format (search/index.json):
{
  "version": 1,
  "embed_model": "nomic-embed-text",
  "vision_model": "llava",
  "indexed_videos": {
    "videos/Concert.mp4": "2026-07-02T10:00:00"
  },
  "scenes": [
    {
      "video"         : "videos/Concert.mp4",
      "video_duration": 37.5,
      "scene_id"      : "v0",
      "start"         : 0.0,
      "end"           : 8.3,
      "frame_time"    : 4.15,
      "phash"         : "f8c8a8b0d0e0f0f0",
      "description"   : "PEOPLE: Performer on stage...",
      "embedding"     : [0.123, ...]
    }
  ]
}
"""
import json
import os
from datetime import datetime, timezone

DEFAULT_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "search_index.json"
)


def load_index(path: str = DEFAULT_INDEX_PATH) -> dict:
    if not os.path.exists(path):
        return {
            "version": 1,
            "embed_model": "",
            "vision_model": "",
            "indexed_videos": {},
            "scenes": [],
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index: dict, path: str = DEFAULT_INDEX_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def mark_video_indexed(index: dict, video_path: str):
    index["indexed_videos"][video_path] = datetime.now(timezone.utc).isoformat()


def is_video_indexed(index: dict, video_path: str) -> bool:
    return video_path in index["indexed_videos"]


def add_scene(index: dict, scene_record: dict):
    """Append a single scene record to the index."""
    index["scenes"].append(scene_record)


def remove_video_scenes(index: dict, video_path: str):
    """Remove all scenes belonging to a video (for re-indexing)."""
    index["scenes"] = [s for s in index["scenes"] if s["video"] != video_path]
    index["indexed_videos"].pop(video_path, None)


def index_stats(index: dict) -> dict:
    n_videos = len(index["indexed_videos"])
    n_scenes = len(index["scenes"])
    n_described = sum(1 for s in index["scenes"] if s.get("description", "").strip())
    n_embedded = sum(1 for s in index["scenes"] if s.get("embedding"))
    return {
        "videos": n_videos,
        "scenes": n_scenes,
        "described": n_described,
        "embedded": n_embedded,
    }
