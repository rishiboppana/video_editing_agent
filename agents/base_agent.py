import json
import re
import requests
from config import OLLAMA_MODEL, OLLAMA_BASE_URL


class BaseAgent:
    def __init__(self, model: str = None):
        self.model = model or OLLAMA_MODEL
        self.name = self.__class__.__name__

    def call_llm(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def extract_json(self, text: str) -> dict:
        # Try fenced code block first, then bare JSON object
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # Find the outermost JSON object or array
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"No valid JSON found in response:\n{text[:300]}")

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"{self.name}.run() not implemented")

    def validate(self, output: dict, **kwargs) -> tuple:
        raise NotImplementedError(f"{self.name}.validate() not implemented")
