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

        try:
            for i, h in enumerate(highlights):
                clip_path = os.path.join(temp_dir, f"clip_{i:03d}.mp4")
                self._cut_clip(video_path, h["start"], h["end"], clip_path)
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

    def _cut_clip(self, video_path: str, start: float, end: float, out: str):
        cmd = [
            FFMPEG_PATH, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(round(end - start, 3)),
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            out,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Clip cut failed [{start}-{end}]:\n{result.stderr[-500:]}")

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
