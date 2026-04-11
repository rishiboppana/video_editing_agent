import subprocess
import os

# ---------- Get video metadata ----------
def get_video_info(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,codec_name",
         "-of", "default=noprint_wrappers=1",
         video],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()


# ---------- Check compatibility ----------
def videos_are_compatible(video_list):
    base_info = get_video_info(video_list[0])

    for video in video_list[1:]:
        if get_video_info(video) != base_info:
            return False

    return True


# ---------- Normalize video ----------
def normalize_video(input_path, output_path):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-r", "30",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        output_path
    ]

    subprocess.run(command, check=True)
    return output_path


# ---------- Fast merge ----------
def fast_merge(video_list, output_path):

    list_file = "videos/file_list.txt"

    with open(list_file, "w") as f:
        for v in video_list:
            f.write(f"file '{os.path.abspath(v)}'\n")

    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        "-y",
        output_path
    ]

    subprocess.run(command, check=True)
    return output_path


# ---------- MAIN FUNCTION ----------
def merge_videos(video_list, output_path="videos/merged.mp4"):

    # ---------- Validation ----------
    if not isinstance(video_list, list) or len(video_list) < 2:
        raise ValueError("Need at least 2 videos to merge")

    for video in video_list:
        if not os.path.exists(video):
            raise FileNotFoundError(f"File not found: {video}")

    print("🔍 Checking compatibility...")

    # ---------- CASE 1: Same format ----------
    if videos_are_compatible(video_list):
        print("⚡ Same format detected → fast merge")
        return fast_merge(video_list, output_path)

    # ---------- CASE 2: Different format ----------
    else:
        print("⚠️ Different formats → normalizing...")

        normalized_videos = []

        for i, video in enumerate(video_list):
            norm_path = f"videos/normalized_{i}.mp4"
            normalize_video(video, norm_path)
            normalized_videos.append(norm_path)

        print("✅ Normalization done → merging safely")

        output = fast_merge(normalized_videos, output_path)

        # ---------- Cleanup temp files ----------
        for f in normalized_videos:
            if os.path.exists(f):
                os.remove(f)

        return output