import subprocess

def trim_video(video_path, start_time, end_time):

    output_path = "videos/trimmed.mp4"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c", "copy",
        output_path
    ]

    subprocess.run(command)

    return output_path