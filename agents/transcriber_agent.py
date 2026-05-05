import base64
import logging
import os
import subprocess
import tempfile

import requests
import whisper

from agents.base_agent import BaseAgent
from config import FFMPEG_PATH, FFPROBE_PATH, MAX_VISUAL_DESCRIPTIONS, OLLAMA_BASE_URL, VISION_MODEL, WHISPER_MODEL

logger = logging.getLogger("orchestrator")


class TranscriberAgent(BaseAgent):
    """
    Extracts three kinds of understanding from a video:
      1. Speech  — Whisper transcription with timestamps
      2. Scenes  — ffprobe scene-cut detection (where the visual content changes)
      3. Vision  — LLaVA frame description (what is actually visible in each scene)
    """

    def __init__(self):
        super().__init__()
        self._whisper = None

    @property
    def whisper(self):
        if self._whisper is None:
            self._whisper = whisper.load_model(WHISPER_MODEL)
        return self._whisper

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, video_path: str, feedback: str = "") -> dict:
        duration = self._get_duration(video_path)
        logger.info(f"  [TranscriberAgent] video duration: {duration:.1f}s")

        # 1. Detect scene cuts → time boundaries
        scene_segments = self._detect_scenes(video_path, duration)
        logger.info(f"  [TranscriberAgent] {len(scene_segments)} visual scenes detected")

        # 2. Describe each scene using LLaVA (extract frame → ask vision LLM)
        visual_segments = self._describe_scenes(video_path, scene_segments)
        logger.info(f"  [TranscriberAgent] {len(visual_segments)} scenes described")

        # 3. Transcribe speech using Whisper
        audio_path = self._extract_audio(video_path)
        try:
            result = self.whisper.transcribe(audio_path)
            audio_segments = [
                {
                    "id": i,
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip(),
                }
                for i, seg in enumerate(result["segments"])
                if seg["text"].strip()
            ]
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        logger.info(f"  [TranscriberAgent] {len(audio_segments)} speech segments transcribed")

        return {
            "segments": audio_segments,
            "visual_segments": visual_segments,
            "full_text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "duration": duration,
        }

    # ------------------------------------------------------------------
    # Scene detection
    # ------------------------------------------------------------------

    def _detect_scenes(self, video_path: str, duration: float) -> list:
        """Use ffprobe to find timestamps where the visual content changes significantly."""
        cmd = [
            FFPROBE_PATH, "-v", "error",
            "-show_entries", "frame=pkt_pts_time",
            "-select_streams", "v",
            "-vf", "select='gt(scene,0.3)'",
            "-of", "csv=p=0",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            scene_starts = []
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            scene_starts.append(float(line.split(",")[0]))
                        except ValueError:
                            continue

            # Build segments between consecutive scene-cut boundaries
            points = sorted(set([0.0] + scene_starts + [duration]))
            segments = []
            for i in range(len(points) - 1):
                start, end = points[i], points[i + 1]
                if end - start < 0.5:
                    continue
                segments.append({"id": f"v{i}", "start": round(start, 2), "end": round(end, 2)})
            return segments

        except Exception as e:
            logger.warning(f"  [TranscriberAgent] scene detection failed ({e}), using 5s chunks")
            return [
                {"id": f"v{i}", "start": float(i), "end": min(float(i + 5), duration)}
                for i in range(0, int(duration), 5)
                if min(float(i + 5), duration) - float(i) >= 1.0
            ]

    # ------------------------------------------------------------------
    # Visual description with LLaVA
    # ------------------------------------------------------------------

    def _describe_scenes(self, video_path: str, scenes: list) -> list:
        """
        For each scene segment, extract a frame at the midpoint and ask LLaVA
        to describe what is visually happening. This is the perception layer.
        """
        # Sort scenes by duration (longest first) and cap the number we describe
        # to keep processing time reasonable on long videos
        scenes_to_describe = sorted(scenes, key=lambda s: s["end"] - s["start"], reverse=True)
        scenes_to_describe = scenes_to_describe[:MAX_VISUAL_DESCRIPTIONS]
        described_ids = {s["id"] for s in scenes_to_describe}

        described = []
        for scene in scenes:
            if scene["id"] not in described_ids:
                # Short/less important scene — mark as undescribed rather than skip entirely
                described.append({**scene, "description": "[brief scene — not individually analyzed]", "type": "visual"})
                continue

            midpoint = round((scene["start"] + scene["end"]) / 2, 2)
            frame_path = tempfile.mktemp(suffix=".jpg")

            try:
                self._extract_frame(video_path, midpoint, frame_path)
                description = self._ask_vision_llm(frame_path, scene["start"], scene["end"])
            except Exception as e:
                logger.warning(f"  [TranscriberAgent] vision failed for scene {scene['id']} ({e})")
                description = "[visual description unavailable]"
            finally:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

            described.append({
                **scene,
                "description": description,
                "type": "visual",
                "sample_time": midpoint,
            })

        return described

    def _extract_frame(self, video_path: str, timestamp: float, out_path: str):
        """Extract a single JPEG frame at a given timestamp."""
        cmd = [
            FFMPEG_PATH, "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "3",          # JPEG quality (2=best, 31=worst)
            "-vf", "scale=640:-1", # Resize width to 640px (keeps aspect ratio, reduces tokens)
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"Frame extraction failed at {timestamp}s:\n{result.stderr[-300:]}")

    def _ask_vision_llm(self, frame_path: str, start: float, end: float) -> str:
        """
        Send a frame to LLaVA (via Ollama /api/chat) and get a visual description.
        LLaVA accepts base64-encoded images in the 'images' field of a message.
        """
        with open(frame_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            f"This is a frame from a video segment between {start}s and {end}s. "
            "In 1-2 sentences, describe: what is visually happening, "
            "who or what is on screen, and the setting or mood. "
            "Be specific and factual. Do not speculate beyond what you see."
        )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_duration(self, video_path: str) -> float:
        cmd = [
            FFPROBE_PATH, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 0.0
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, output: dict, **kwargs) -> tuple:
        if not isinstance(output, dict):
            return False, "Output must be a dict"
        for field in ("segments", "full_text", "visual_segments"):
            if field not in output:
                return False, f"Missing field: {field}"

        full_text = (output.get("full_text") or "").strip()

        # Hallucination guard — Whisper sometimes generates these on silent/music videos
        hallucinations = [
            "thank you for watching", "thanks for watching",
            "please subscribe", "amara.org", "subtitle by", "brought to you by",
        ]
        if full_text and any(h in full_text.lower() for h in hallucinations) and len(full_text) < 100:
            output["full_text"] = ""
            output["segments"] = []
            return True, "Hallucination caught; treating as silent video"

        # High repetition guard
        words = full_text.lower().split()
        if len(words) > 10 and len(set(words)) / len(words) < 0.2:
            output["full_text"] = ""
            output["segments"] = []
            return True, "Repetitive noise caught; treating as silent video"

        if output["segments"]:
            for seg in output["segments"]:
                if not all(k in seg for k in ("id", "start", "end", "text")):
                    return False, f"Segment missing required keys: {seg}"
                if seg["end"] <= seg["start"]:
                    return False, f"Segment has invalid time range: {seg}"

        return True, (
            f"{len(output['segments'])} speech segments, "
            f"{len(output['visual_segments'])} visual scenes"
        )
