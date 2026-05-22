"""
Same behavior as agent.py but always uses Ollama model llama3:latest (no OLLAMA_MODEL env).
Use on hosts with enough RAM (~8+ GiB free recommended). For small instances use agent.py
with OLLAMA_MODEL=llama3.2:3b instead.
"""
import json
import os
import subprocess
import time

import requests

from merge_tool import merge_videos
from speech_to_text import speech_to_text
from trim_tool import trim_video

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_FIXED = "llama3:latest"
DEFAULT_TRIM_INPUT = "videos/1_feb5.mp4"


def get_video_duration_seconds(video_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def ask_llm(prompt, trim_input_path=None):

    system_prompt = """
You are a video editing agent. 
You have access to the following tools:

1. trim_video: Trims a video between two timestamps.
   - start_time (number): Start time in seconds.
   - end_time (number): End time in seconds (segment end time passed to ffmpeg -to).

2. merge_videos: Merges a list of video files into one.
   - video_list (list of strings): Absolute or relative paths to the videos to merge, in order.
   - output_path (string): Path for the merged output file. Default: "videos/merged.mp4"

When the user chooses trim_video, map natural language to start_time/end_time using video_duration_seconds:

- "first N seconds" / "beginning N seconds": start_time=0, end_time=min(N, video_duration_seconds)
- "last N seconds" / "final N seconds": start_time=max(0, video_duration_seconds - N), end_time=video_duration_seconds
- "from A to B" (seconds): start_time=A, end_time=B (stay within 0..video_duration_seconds)

Analyze the user's instruction and decide which tool to use. If no tool is appropriate, set "tool" to null.

Return ONLY JSON in this exact format:

For trim_video:
{
  "tool": "trim_video",
  "parameters": {
    "start_time": number,
    "end_time": number
  }
}

For merge_videos:
{
  "tool": "merge_videos",
  "parameters": {
    "video_list": ["path/to/video1.mp4", "path/to/video2.mp4"],
    "output_path": "videos/merged.mp4"
  }
}

For no tool:
{
  "tool": null,
  "parameters": {}
}
"""

    path = trim_input_path or DEFAULT_TRIM_INPUT
    extra = ""
    try:
        duration = get_video_duration_seconds(path)
        extra = (
            f"\nContext for trim_video:\n"
            f"- trim_input_path: {path}\n"
            f"- video_duration_seconds: {duration:.6f}\n"
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        extra = f"\nContext for trim_video: could not read duration for {path}: {e}\n"

    full_prompt = system_prompt + extra + "\nInstruction: " + prompt

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL_FIXED,
            "prompt": full_prompt,
            "stream": False
        },
        timeout=600,
    )

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Ollama returned non-JSON (HTTP {response.status_code}): {response.text[:2000]!r}"
        ) from e

    if data.get("error"):
        raise RuntimeError(
            f"Ollama error: {data['error']}\n"
            "llama3:latest needs substantial RAM; use a larger instance or run agent.py "
            "with a smaller OLLAMA_MODEL on small hosts."
        )

    if "response" not in data:
        raise RuntimeError(
            f"Ollama JSON missing 'response' (HTTP {response.status_code}). "
            f"Keys: {list(data.keys())}. Body: {str(data)[:1500]}"
        )

    text = data["response"]

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end != 0:
        json_str = text[start:end]
        return json.loads(json_str)
    else:
        raise ValueError(f"Could not find JSON in response: {text}")


def execute_agent_decision(result):
    """Runs trim or merge (FFmpeg) from parsed LLM JSON."""
    tool_name = result.get("tool")

    if tool_name == "trim_video":
        params = result.get("parameters", {})
        start = params.get("start_time")
        end = params.get("end_time")

        if start is not None and end is not None:
            trim_in = DEFAULT_TRIM_INPUT
            try:
                dur = get_video_duration_seconds(trim_in)
                s = float(start)
                e = float(end)
                if s < 0:
                    s = 0.0
                if e < 0:
                    e = 0.0
                if s > dur:
                    s = dur
                if e > dur:
                    e = dur
                if e < s:
                    s, e = e, s
                if abs(e - s) < 1e-6:
                    e = min(dur, s + 0.001)
                start, end = s, e
            except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
                print(f"Warning: could not clamp trim range: {exc}")

            print(f"Agent decided to trim {trim_in} from {start}s to {end}s.")
            output = trim_video(trim_in, start, end)
            return output
        else:
            return "Error: Missing parameters for trim_video."

    elif tool_name == "merge_videos":
        params = result.get("parameters", {})
        video_list = params.get("video_list")
        output_path = params.get("output_path", "videos/merged.mp4")

        if video_list and len(video_list) >= 2:
            print(f"Agent decided to merge {len(video_list)} videos → {output_path}")
            output = merge_videos(video_list, output_path)
            return output
        else:
            return "Error: merge_videos requires at least 2 videos in video_list."

    elif tool_name is None:
        return "Agent could not find an appropriate tool to fulfill the request."
    else:
        return f"Error: Unknown tool {tool_name} requested."


def run_agent(user_prompt):
    result = ask_llm(user_prompt, trim_input_path=DEFAULT_TRIM_INPUT)
    return execute_agent_decision(result)


if __name__ == "__main__":
    print(f"Ollama model (fixed): {OLLAMA_MODEL_FIXED}")
    t0 = time.perf_counter()
    instruction = speech_to_text("audio/first4seconds.wav")
    t_stt = time.perf_counter()
    print("User prompt:", instruction)
    print(f"Timing — speech-to-text: {t_stt - t0:.2f} s")

    t_llm_start = time.perf_counter()
    llm_result = ask_llm(instruction, trim_input_path=DEFAULT_TRIM_INPUT)
    t_llm_end = time.perf_counter()
    print(f"Timing — LLM (JSON): {t_llm_end - t_llm_start:.2f} s")

    t_vid_start = time.perf_counter()
    output = execute_agent_decision(llm_result)
    t_vid_end = time.perf_counter()
    print(f"Timing — video output (FFmpeg): {t_vid_end - t_vid_start:.2f} s")

    print(f"Timing — total: {t_vid_end - t0:.2f} s")
    print("Video saved to:", output)
