import requests
import json
from trim_tool import trim_video
from speech_to_text import speech_to_text 
OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_llm(prompt):

    system_prompt = """
You are a video editing agent. 
You have access to the following tools:

1. trim_video: Trims a video between two timestamps.
   - start_time (number): Start time in seconds.
   - end_time (number): End time in seconds.

Analyze the user's instruction and decide which tool to use. If no tool is appropriate, set "tool" to null.

Return ONLY JSON in this exact format:
{
 "tool": "tool_name",
 "parameters": {
    "start_time": number,
    "end_time": number
 }
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

    # LLMs often add conversational padding (e.g. "Here's the JSON:")
    # We find the start and end of the JSON object instead.
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
             
    elif tool_name is None:
        return "Agent could not find an appropriate tool to fulfill the request."
    else:
        return f"Error: Unknown tool {tool_name} requested."


if __name__ == "__main__":

    instruction = input("Enter instruction: ")

    output = run_agent(instruction)

    print("Video saved to:", output)