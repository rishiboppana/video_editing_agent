import base64
import logging
import os
import re
import subprocess
import tempfile

import requests
import torch
import whisper

from agents.base_agent import BaseAgent
from config import (
    ENABLE_VISION,
    FFMPEG_PATH, FFPROBE_PATH,
    MAX_VISUAL_DESCRIPTIONS, OLLAMA_BASE_URL,
    VISION_MODEL, VISION_TIMEOUT,
    VISION_MULTI_FRAME_MIN, VISION_MAX_CONSECUTIVE_FAILURES,
    WHISPER_MODEL,
)

logger = logging.getLogger("orchestrator")

SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "0.25"))


def _best_device() -> str:
    """
    Pick the fastest available compute device for Whisper.

    Priority:
      1. CUDA  — NVIDIA GPU on Linux/Windows
      2. MPS   — Apple Silicon (M1/M2/M3/M4) Metal GPU
      3. CPU   — fallback

    The result is logged once so the user can see which device was selected.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"  [TranscriberAgent] Whisper device: {device}")
    return device


class TranscriberAgent(BaseAgent):
    """
    Extracts three kinds of understanding from a video:
      1. Speech  — Whisper transcription with timestamps
      2. Scenes  — ffmpeg scene-cut detection (where the visual content changes)
      3. Vision  — LLaVA multi-frame description (what is actually visible in each scene)
    """

    def __init__(self):
        super().__init__()
        self._whisper = None
        self._device = _best_device()

    @property
    def whisper(self):
        if self._whisper is None:
            self._whisper = whisper.load_model(WHISPER_MODEL, device=self._device)
        return self._whisper

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, video_path: str, feedback: str = "") -> dict:
        duration = self._get_duration(video_path)
        logger.info(f"  [TranscriberAgent] video duration: {duration:.1f}s")

        scene_segments = self._detect_scenes(video_path, duration)
        logger.info(f"  [TranscriberAgent] {len(scene_segments)} visual scenes detected")

        visual_segments = self._describe_scenes(video_path, scene_segments)
        logger.info(f"  [TranscriberAgent] {len(visual_segments)} scenes described")

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
    # Scene detection  (ffmpeg, not ffprobe)
    # ------------------------------------------------------------------
    # ffprobe does NOT compute scene scores via -vf select — it only works
    # with ffmpeg's full filter pipeline.  We write scene-change metadata
    # to a temp file, parse the pts_time values, then fall back to adaptive
    # fixed chunks if nothing is found.
    # ------------------------------------------------------------------

    def _detect_scenes(self, video_path: str, duration: float) -> list:
        try:
            segments = self._detect_with_ffmpeg(video_path, duration)
            if len(segments) > 1:
                return segments
            logger.info(
                "  [TranscriberAgent] no scene cuts at threshold "
                f"{SCENE_THRESHOLD} — using adaptive chunks"
            )
        except Exception as e:
            logger.warning(f"  [TranscriberAgent] scene detection error: {e}")

        return self._adaptive_chunks(duration)

    def _detect_with_ffmpeg(self, video_path: str, duration: float) -> list:
        """Write scene-change metadata to a temp file via ffmpeg select filter."""
        meta_file = tempfile.mktemp(suffix="_scenes.txt")
        try:
            cmd = [
                FFMPEG_PATH, "-y", "-i", video_path,
                "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',metadata=mode=print:file={meta_file}",
                "-vsync", "0", "-an", "-f", "null", "-",
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

            scene_times = [0.0]
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    content = f.read()
                for line in content.splitlines():
                    # Lines look like: "frame:12  pts:12012  pts_time:0.400400"
                    if "pts_time:" in line:
                        m = re.search(r"pts_time:(\d+\.?\d*)", line)
                        if m:
                            t = float(m.group(1))
                            if t > 0.2:
                                scene_times.append(t)

            points = sorted(set(scene_times + [duration]))
            segments = []
            for i in range(len(points) - 1):
                s, e = points[i], points[i + 1]
                if e - s >= 1.0:
                    segments.append({
                        "id": f"v{i}",
                        "start": round(s, 2),
                        "end": round(e, 2),
                    })
            return segments
        finally:
            if os.path.exists(meta_file):
                os.remove(meta_file)

    def _adaptive_chunks(self, duration: float) -> list:
        """Divide the video into ~8 equal chunks when scene detection finds nothing."""
        chunk = max(3.0, duration / 8.0)
        segments, idx, t = [], 0, 0.0
        while t < duration:
            end = min(round(t + chunk, 2), round(duration, 2))
            if end - t >= 1.0:
                segments.append({"id": f"v{idx}", "start": round(t, 2), "end": end})
                idx += 1
            t += chunk
        return segments

    # ------------------------------------------------------------------
    # Visual description with LLaVA  (multi-frame for long scenes)
    # ------------------------------------------------------------------

    def _describe_scenes(self, video_path: str, scenes: list) -> list:
        if not ENABLE_VISION:
            logger.info("  [TranscriberAgent] vision disabled (ENABLE_VISION=false) — skipping LLaVA")
            return [{**s, "description": "[vision disabled]", "type": "visual"} for s in scenes]

        # Prioritise longest scenes for LLaVA (they carry the most visual information)
        priority = sorted(scenes, key=lambda s: s["end"] - s["start"], reverse=True)
        describe_ids = {s["id"] for s in priority[:MAX_VISUAL_DESCRIPTIONS]}

        described = []
        consecutive_failures = 0

        for scene in scenes:
            if scene["id"] not in describe_ids:
                described.append({
                    **scene,
                    "description": "[brief scene — not individually analyzed]",
                    "type": "visual",
                })
                continue

            # If vision model is consistently timing out, stop trying
            if consecutive_failures >= VISION_MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"  [TranscriberAgent] {consecutive_failures} consecutive vision "
                    "timeouts — skipping remaining vision calls"
                )
                described.append({
                    **scene,
                    "description": "[vision skipped — model too slow for this hardware]",
                    "type": "visual",
                })
                continue

            seg_duration = scene["end"] - scene["start"]

            # Multi-frame sampling only for genuinely long scenes (default >= 15s).
            # On CPU, even 1 frame can take 2–3 min; 3 frames triples that.
            if seg_duration >= VISION_MULTI_FRAME_MIN:
                sample_times = [
                    round(scene["start"] + seg_duration * 0.20, 2),
                    round(scene["start"] + seg_duration * 0.50, 2),
                    round(scene["start"] + seg_duration * 0.80, 2),
                ]
            else:
                sample_times = [round((scene["start"] + scene["end"]) / 2, 2)]

            frame_paths = []
            try:
                for t in sample_times:
                    fp = tempfile.mktemp(suffix=".jpg")
                    self._extract_frame(video_path, t, fp)
                    frame_paths.append(fp)
                description = self._ask_vision_llm(
                    frame_paths, scene["start"], scene["end"], sample_times
                )
                consecutive_failures = 0  # reset on success
            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    f"  [TranscriberAgent] vision failed for {scene['id']} "
                    f"({e}) [{consecutive_failures}/{VISION_MAX_CONSECUTIVE_FAILURES}]"
                )
                description = "[visual description unavailable]"
            finally:
                for fp in frame_paths:
                    if os.path.exists(fp):
                        os.remove(fp)

            described.append({
                **scene,
                "description": description,
                "type": "visual",
                "sample_times": sample_times,
            })

        return described

    def _extract_frame(self, video_path: str, timestamp: float, out_path: str):
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
            raise RuntimeError(
                f"Frame extraction failed at {timestamp}s:\n{result.stderr[-300:]}"
            )

    def _ask_vision_llm(
        self,
        frame_paths: list,
        start: float,
        end: float,
        sample_times: list,
    ) -> str:
        images_b64 = []
        for fp in frame_paths:
            with open(fp, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode("utf-8"))

        if len(images_b64) == 1:
            prompt = (
                f"This frame is from a video at {sample_times[0]}s "
                f"(scene: {start}s–{end}s). "
                "In 1–2 sentences describe: who/what is on screen, "
                "what action is happening, and the setting or mood."
            )
        else:
            times_str = ", ".join(f"{t}s" for t in sample_times)
            prompt = (
                f"These {len(images_b64)} frames are sampled at {times_str} "
                f"from a video segment {start}s–{end}s. "
                "In 2–3 sentences describe: what is happening throughout this segment, "
                "who/what is on screen, how the scene changes, and the overall mood."
            )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": VISION_MODEL,
                "messages": [
                    {"role": "user", "content": prompt, "images": images_b64}
                ],
                "stream": False,
            },
            timeout=VISION_TIMEOUT,
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

        hallucinations = [
            "thank you for watching", "thanks for watching",
            "please subscribe", "amara.org", "subtitle by", "brought to you by",
        ]
        if full_text and any(h in full_text.lower() for h in hallucinations) and len(full_text) < 100:
            output["full_text"] = ""
            output["segments"] = []
            return True, "Hallucination caught; treating as silent video"

        words = full_text.lower().split()
        if len(words) > 10 and len(set(words)) / len(words) < 0.2:
            output["full_text"] = ""
            output["segments"] = []
            return True, "Repetitive noise caught; treating as silent video"

        for seg in output["segments"]:
            if not all(k in seg for k in ("id", "start", "end", "text")):
                return False, f"Segment missing required keys: {seg}"
            if seg["end"] <= seg["start"]:
                return False, f"Segment has invalid time range: {seg}"

        return True, (
            f"{len(output['segments'])} speech segments, "
            f"{len(output['visual_segments'])} visual scenes"
        )
