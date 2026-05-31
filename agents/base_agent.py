import json
import re
import time

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


class BaseAgent:
    def __init__(self, model: str = None):
        self.model = model or OLLAMA_MODEL
        self.name = self.__class__.__name__

    # ------------------------------------------------------------------
    # LLM call  — retries on timeout / server errors, fails fast on
    # ConnectionError (Ollama not running is a config problem, not transient)
    # ------------------------------------------------------------------

    def call_llm(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "stream": False}
        last_err = None

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]

            except requests.exceptions.ConnectionError:
                # Ollama is not running — no point retrying
                raise RuntimeError(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                    "Run 'ollama serve' first."
                )
            except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))   # 3s, then 6s
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2)

        raise RuntimeError(f"LLM call failed after 3 attempts: {last_err}")

    # ------------------------------------------------------------------
    # JSON extraction  — handles fenced blocks, bare objects, and common
    # LLM formatting errors (trailing commas, single-quoted strings)
    # ------------------------------------------------------------------

    def extract_json(self, text: str) -> dict:
        # 1. Fenced code blocks  ```json ... ``` or ``` ... ```
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                parsed = self._try_parse(m.group(1).strip())
                if parsed is not None:
                    return parsed

        # 2. Outermost { } or [ ]  (most common bare-object response)
        for open_c, close_c in [('{', '}'), ('[', ']')]:
            start = text.find(open_c)
            end = text.rfind(close_c)
            if start != -1 and end > start:
                parsed = self._try_parse(text[start:end + 1])
                if parsed is not None:
                    return parsed

        raise ValueError(f"No valid JSON found in LLM response:\n{text[:400]}")

    def _try_parse(self, text: str):
        """
        Try json.loads with progressive repairs:
          1. As-is
          2. Remove trailing commas before } or ]
          3. Replace smart/curly quotes with straight ones
        Returns parsed object or None.
        """
        candidates = [
            text,
            re.sub(r",\s*([}\]])", r"\1", text),                         # trailing commas
            re.sub(r",\s*([}\]])", r"\1", text).replace("'", '"'),        # single quotes
            text.replace("“", '"').replace("”", '"'),            # curly quotes
        ]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"{self.name}.run() not implemented")

    def validate(self, output: dict, **kwargs) -> tuple:
        raise NotImplementedError(f"{self.name}.validate() not implemented")
