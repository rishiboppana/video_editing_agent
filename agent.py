import requests
import json
from trim_tool import trim_video

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_llm(prompt):

    system_prompt = """
You are a video editing agent.

Extract the trimming timestamps from the instruction.

Return ONLY JSON in this format:
{
 "start_time": number,
 "end_time": number
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

    start = result["start_time"]
    end = result["end_time"]

    output = trim_video("videos/input1.mov", start, end)

    return output


if __name__ == "__main__":

    instruction = input("Enter instruction: ")

    output = run_agent(instruction)

    print("Video saved to:", output)