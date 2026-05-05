import os
import shutil
import subprocess
import tempfile

import whisper

from agents.base_agent import BaseAgent
from config import WHISPER_MODEL, FFMPEG_PATH, FFPROBE_PATH


class TranscriberAgent(BaseAgent):
    """Extracts audio from video and transcribes it using Whisper. No LLM needed."""

    def __init__(self):
        super().__init__()
        self._whisper = None

    @property
    def whisper(self):
        if self._whisper is None:
            self._whisper = whisper.load_model(WHISPER_MODEL)
        return self._whisper

    def run(self, video_path: str, feedback: str = "") -> dict:
        duration = self._get_duration(video_path)
        visual_segments = self._get_visual_segments(video_path, duration)
        audio_path = self._extract_audio(video_path)
        try:
            result = self.whisper.transcribe(audio_path)
            segments = [
                {
                    "id": i,
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip(),
                }
                for i, seg in enumerate(result["segments"])
                if seg["text"].strip()
            ]
            return {
                "segments": segments,
                "visual_segments": visual_segments,
                "full_text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "duration": duration,
            }
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _get_visual_segments(self, video_path: str, duration: float) -> list:
        """Detect scene changes and return them as visual-only segments."""
        # Use ffprobe to find scene changes (scene > 0.3 is fairly sensitive)
        cmd = [
            FFPROBE_PATH, "-v", "error", "-show_entries", "frame=pkt_pts_time,pict_type",
            "-select_streams", "v", "-filter:v", "select='gt(scene,0.3)'",
            "-of", "csv=p=0", video_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            scene_starts = []
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split(",")
                    if parts:
                        try:
                            scene_starts.append(float(parts[0]))
                        except ValueError:
                            continue
            
            # Create segments between scene changes
            visual_segments = []
            points = sorted(list(set([0.0] + scene_starts + [duration])))
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i+1]
                if end - start < 0.5: continue # Skip too-short scenes
                visual_segments.append({
                    "id": f"v{i}",
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "type": "visual"
                })
            return visual_segments
        except Exception:
            # Fallback to fixed 5s chunks if scene detection fails or times out
            chunks = []
            for i in range(0, int(duration), 5):
                end = min(i + 5, duration)
                if end - i < 1.0: break
                chunks.append({"id": f"v{i//5}", "start": float(i), "end": float(end), "type": "visual"})
            return chunks

    def _get_duration(self, video_path: str) -> float:
        cmd = [
            FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 0.0
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    def validate(self, output: dict, **kwargs) -> tuple:
        if not isinstance(output, dict):
            return False, "Output must be a dict"
        for field in ("segments", "full_text"):
            if field not in output:
                return False, f"Missing field: {field}"
        
        full_text = (output.get("full_text") or "").strip()
        
        # If no text at all, that's fine (silent video), but segments must exist or be explicitly empty
        if not full_text:
            return True, "No speech detected (silent video)"

        # Detect common Whisper hallucinations on silent/music videos
        hallucinations = [
            "thank you for watching", "thanks for watching",
            "please subscribe", "subscribing",
            "brought to you by", "subtitle by",
            "amara.org", "english subtitles"
        ]
        text_lower = full_text.lower()
        if any(h in text_lower for h in hallucinations) and len(full_text) < 100:
             # If it's short and contains hallucination markers, treat as "no speech"
             output["full_text"] = ""
             output["segments"] = []
             return True, "Hallucination caught; treating as silent video"

        # Detect high repetition (another sign of hallucination)
        words = text_lower.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.2: # Very high repetition
                output["full_text"] = ""
                output["segments"] = []
                return True, "Repetitive noise caught; treating as silent video"

        # If we have segments, validate them
        if output["segments"]:
            for seg in output["segments"]:
                if not all(k in seg for k in ("id", "start", "end", "text")):
                    return False, f"Segment missing required keys: {seg}"
                if seg["end"] <= seg["start"]:
                    return False, f"Segment has invalid time range: {seg}"
        
        return True, f"{len(output.get('segments', []))} segments transcribed"

    def _extract_audio(self, video_path: str) -> str:
        audio_path = tempfile.mktemp(suffix=".wav")
        cmd = [
            FFMPEG_PATH, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio extraction failed:\n{result.stderr}")
        return audio_path
