import os
import subprocess
import tempfile


def get_video_info(video_path: str) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
            "-of", "json",
            video_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}:\n{result.stderr}")
    import json
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    return streams[0] if streams else {}


def videos_are_compatible(video_list: list) -> bool:
    base = get_video_info(video_list[0])
    for video in video_list[1:]:
        info = get_video_info(video)
        if (info.get("width"), info.get("height"), info.get("r_frame_rate"), info.get("codec_name")) != \
           (base.get("width"), base.get("height"), base.get("r_frame_rate"), base.get("codec_name")):
            return False
    return True


def normalize_video(input_path: str, output_path: str) -> str:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-r", "30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"normalize_video failed on {input_path}:\n{result.stderr[-500:]}")
    return output_path


def _concat_videos(video_list: list, output_path: str) -> str:
    list_file = tempfile.mktemp(suffix=".txt")
    try:
        with open(list_file, "w") as f:
            for v in video_list:
                f.write(f"file '{os.path.abspath(v)}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"concat failed:\n{result.stderr[-500:]}")
        return output_path
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)


def merge_videos(video_list: list, output_path: str) -> str:
    """Merge a list of video files into output_path. Normalizes if formats differ."""
    if not isinstance(video_list, list) or len(video_list) < 2:
        raise ValueError("Need at least 2 videos to merge")
    for v in video_list:
        if not os.path.exists(v):
            raise FileNotFoundError(f"Video not found: {v}")

    if videos_are_compatible(video_list):
        return _concat_videos(video_list, output_path)

    temp_dir = tempfile.mkdtemp(prefix="merge_norm_")
    normalized = []
    try:
        for i, v in enumerate(video_list):
            norm_path = os.path.join(temp_dir, f"norm_{i}.mp4")
            normalize_video(v, norm_path)
            normalized.append(norm_path)
        return _concat_videos(normalized, output_path)
    finally:
        for f in normalized:
            if os.path.exists(f):
                os.remove(f)
        if os.path.isdir(temp_dir):
            os.rmdir(temp_dir)
