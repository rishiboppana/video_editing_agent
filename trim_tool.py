import subprocess


def trim_video(video_path: str, start_time: float, end_time: float, output_path: str) -> str:
    """Trim video_path from start_time to end_time (seconds) and write to output_path."""
    duration = round(end_time - start_time, 3)
    if duration <= 0:
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"trim_video failed:\n{result.stderr[-500:]}")
    return output_path
