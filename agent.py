import requests
import json
from trim_tool import trim_video
from speech_to_text import speech_to_text
from merge_tool import merge_videos          # ← NEW IMPORT

OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_llm(prompt):

    system_prompt = """
You are a video editing agent. 
You have access to the following tools:

1. trim_video: Trims a video between two timestamps.
   - start_time (number): Start time in seconds.
   - end_time (number): End time in seconds.

2. merge_videos: Merges a list of video files into one.
   - video_list (list of strings): Absolute or relative paths to the videos to merge, in order.
   - output_path (string): Path for the merged output file. Default: "videos/merged.mp4"

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

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3:latest",
            "prompt": system_prompt + "\nInstruction: " + prompt,
            "stream": False
        }
    )

    text = response.json()["response"]

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end != 0:
        json_str = text[start:end]
        return json.loads(json_str)
    else:
        raise ValueError(f"Could not find JSON in response: {text}")


def run_agent(user_prompt):

    result = ask_llm(user_prompt)

    tool_name = result.get("tool")

    if tool_name == "trim_video":
        params = result.get("parameters", {})
        start = params.get("start_time")
        end = params.get("end_time")

        if start is not None and end is not None:
            print(f"Agent decided to trim video from {start}s to {end}s.")
            output = trim_video("videos/1_feb5.mp4", start, end)
            return output
        else:
            return "Error: Missing parameters for trim_video."

    # ↓ NEW BLOCK — merge tool handler
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
    # ↑ END NEW BLOCK

    elif tool_name is None:
        return "Agent could not find an appropriate tool to fulfill the request."
    else:
        return f"Error: Unknown tool {tool_name} requested."


if __name__ == "__main__":

    # instruction = input("Enter instruction: ")
    instruction = speech_to_text("audio/first4seconds.wav")
    print("User prompt:", instruction)

    output = run_agent(instruction)

    print("Video saved to:", output)