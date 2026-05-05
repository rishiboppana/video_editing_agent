import whisper
import subprocess
import json
import os

VIDEO_PATH = "videos/Dequila.mp4"
OUTPUT_PATH = "Dequila_highlight2.mp4"
MAX_DURATION = 15  # seconds

# -----------------------------
# 1. Transcribe using Whisper
# -----------------------------
def transcribe(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })
    return segments

# -----------------------------
# 2. Simple Text Importance Score
# -----------------------------
def text_score(text):
    # simple heuristic: longer + strong words
    keywords = ["important", "key", "must", "best", "trick"]
    score = len(text.split()) / 10
    
    for k in keywords:
        if k in text.lower():
            score += 1.0
    return score

# -----------------------------
# 3. Score Segments
# -----------------------------
def score_segments(segments):
    for seg in segments:
        seg["score"] = text_score(seg["text"])
    return segments

# -----------------------------
# 4. Select Top Segments (15 sec budget)
# -----------------------------
def select_segments(segments, max_duration=15):
    segments = sorted(segments, key=lambda x: x["score"], reverse=True)
    
    selected = []
    total = 0
    
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if total + duration <= max_duration:
            selected.append(seg)
            total += duration
    
    # keep chronological order
    selected = sorted(selected, key=lambda x: x["start"])
    return selected

# -----------------------------
# 5. Cut Clips using ffmpeg
# -----------------------------
def cut_clips(video_path, segments):
    clip_files = []
    
    for i, seg in enumerate(segments):
        out_file = f"clip_{i}.mp4"
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ss", str(seg["start"]),
            "-to", str(seg["end"]),
            "-c", "copy",
            out_file
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_files.append(out_file)
    
    return clip_files

# -----------------------------
# 6. Stitch Clips
# -----------------------------
def stitch_clips(clip_files, output_path):
    with open("clips.txt", "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "clips.txt",
        "-c", "copy",
        output_path
    ]
    
    subprocess.run(cmd)
    
    # cleanup
    for clip in clip_files:
        os.remove(clip)
    os.remove("clips.txt")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("Transcribing...")
    segments = transcribe(VIDEO_PATH)
    
    print("Scoring...")
    segments = score_segments(segments)
    
    print("Selecting highlights...")
    selected = select_segments(segments, MAX_DURATION)
    
    print("Cutting clips...")
    clips = cut_clips(VIDEO_PATH, selected)
    
    print("Stitching final video...")
    stitch_clips(clips, OUTPUT_PATH)
    
    print(f"Done! Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()