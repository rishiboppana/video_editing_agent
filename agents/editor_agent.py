import os
import shutil
import subprocess
import tempfile

from agents.base_agent import BaseAgent
from config import FFMPEG_PATH, FFPROBE_PATH


class EditorAgent(BaseAgent):
    """Cuts highlight clips from the source video and joins them using ffmpeg. No LLM needed."""

    def run(self, video_path: str, highlights: list, output_path: str, feedback: str = "") -> dict:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        temp_dir = tempfile.mkdtemp(prefix="video_editor_")
        clip_paths = []
        dimensions = self._get_dimensions(video_path)

        try:
            for i, h in enumerate(highlights):
                clip_path = os.path.join(temp_dir, f"clip_{i:03d}.mp4")
                self._cut_clip(video_path, h["start"], h["end"], clip_path, h.get("zoom"), dimensions)
                clip_paths.append(clip_path)

            self._join_clips(clip_paths, output_path)
            duration = self._get_duration(output_path)

            return {
                "output_path": output_path,
                "duration": duration,
                "clips_count": len(clip_paths),
            }
        finally:
            for cp in clip_paths:
                if os.path.exists(cp):
                    os.remove(cp)
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def validate(self, output: dict, **kwargs) -> tuple:
        if "output_path" not in output:
            return False, "Missing output_path"
        if not os.path.exists(output["output_path"]):
            return False, f"Output file not found: {output['output_path']}"
        if output.get("duration", 0) <= 0:
            return False, "Output video has zero or negative duration"
        return True, f"Video created: {output['output_path']} ({output['duration']}s)"

    def _cut_clip(
        self,
        video_path: str,
        start: float,
        end: float,
        out: str,
        zoom: dict = None,
        dimensions: tuple = (0, 0),
    ):
        cmd = [
            FFMPEG_PATH, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(round(end - start, 3)),
        ]

        vf = _zoom_filter(zoom, *dimensions)
        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            out,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Clip cut failed [{start}-{end}]:\n{result.stderr[-500:]}")

    def _get_dimensions(self, video_path: str) -> tuple:
        cmd = [
            FFPROBE_PATH, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            width, height = result.stdout.strip().split("x")
            return int(width), int(height)
        except (ValueError, AttributeError):
            return 0, 0

    def _join_clips(self, clip_paths: list, output_path: str):
        if len(clip_paths) == 1:
            shutil.copy(clip_paths[0], output_path)
            return

        list_file = tempfile.mktemp(suffix=".txt")
        try:
            with open(list_file, "w") as f:
                for cp in clip_paths:
                    f.write(f"file '{os.path.abspath(cp)}'\n")

            cmd = [
                FFMPEG_PATH, "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Clip join failed:\n{result.stderr[-500:]}")
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

    def _get_duration(self, path: str) -> float:
        cmd = [
            FFPROBE_PATH, "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return round(float(result.stdout.strip()), 2)
        return 0.0


# ------------------------------------------------------------------
# Zoom helpers
# ------------------------------------------------------------------

# Maps each FOCUS_POSITIONS grid cell to a (horizontal, vertical) anchor
# fraction (0.0-1.0) used to position the crop window within the frame.
_POSITION_ANCHORS = {
    "top-left": (0.0, 0.0), "top-center": (0.5, 0.0), "top-right": (1.0, 0.0),
    "middle-left": (0.0, 0.5), "center": (0.5, 0.5), "middle-right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0), "bottom-center": (0.5, 1.0), "bottom-right": (1.0, 1.0),
}


def _zoom_filter(zoom: dict, width: int, height: int) -> str:
    """
    Build an ffmpeg crop+scale filter that "punches in" on the grid cell
    named by zoom["position"], at zoom["level"]x magnification, then scales
    back to the original resolution.

    Returns None when no zoom is requested (level <= 1.0) or the source
    dimensions are unknown.
    """
    if not zoom or width <= 0 or height <= 0:
        return None

    level = float(zoom.get("level", 1.0))
    if level <= 1.0:
        return None

    anchor_x, anchor_y = _POSITION_ANCHORS.get(zoom.get("position", "center"), (0.5, 0.5))

    # Crop window size for the requested magnification, rounded to even
    # pixels (required by yuv420p encoding).
    crop_w = max(2, int(width / level) & ~1)
    crop_h = max(2, int(height / level) & ~1)

    x = int((width - crop_w) * anchor_x) & ~1
    y = int((height - crop_h) * anchor_y) & ~1
    x = max(0, min(x, width - crop_w))
    y = max(0, min(y, height - crop_h))

    return f"crop={crop_w}:{crop_h}:{x}:{y},scale={width}:{height}"
