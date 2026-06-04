import json
import re
import time

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


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

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                # Allow up to 8192 output tokens so large JSON responses
                # are never cut off mid-generation.
                "num_predict": 8192,
            },
        }
        last_err = None

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                    timeout=OLLAMA_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]

            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                    "Run 'ollama serve' first."
                )
            except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2)

        raise RuntimeError(f"LLM call failed after 3 attempts: {last_err}")

    # ------------------------------------------------------------------
    # JSON extraction  — handles fenced blocks, bare objects, common
    # formatting errors, and truncated responses
    # ------------------------------------------------------------------

    def extract_json(self, text: str) -> dict:
        # 1. Fenced code blocks  ```json ... ``` or ``` ... ```
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                parsed = self._try_parse(m.group(1).strip())
                if parsed is not None:
                    return parsed

        # 2. Outermost { } or [ ]
        for open_c, close_c in [('{', '}'), ('[', ']')]:
            start = text.find(open_c)
            end = text.rfind(close_c)
            if start != -1 and end > start:
                parsed = self._try_parse(text[start:end + 1])
                if parsed is not None:
                    return parsed

        # 3. Truncated JSON — LLM hit token limit before closing the object.
        #    Count unmatched braces and close them.
        start = text.find('{')
        if start != -1:
            candidate = text[start:]
            parsed = self._try_close_truncated(candidate)
            if parsed is not None:
                return parsed

        raise ValueError(f"No valid JSON found in LLM response:\n{text[:400]}")

    def _try_parse(self, text: str):
        """Try json.loads with progressive repairs. Returns parsed object or None."""
        # Pre-process: strip unit suffixes on bare numeric values (e.g. 4.5s → 4.5)
        # LLMs frequently write timestamps as "4.5s" which is not valid JSON.
        cleaned = re.sub(r'(?<=[:\[,\s])(\d+\.?\d*)\s*s(?=\s*[,\]\}])', r'\1', text)

        base = re.sub(r",\s*([}\]])", r"\1", cleaned)  # strip trailing commas
        # Replace left/right curly double-quotes (" / ") with straight quotes
        candidates = [
            text,
            cleaned,
            base,
            base.replace("'", '"'),   # single -> double quotes
        ]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def _try_close_truncated(self, text: str):
        """
        Close a JSON object that was cut off mid-generation.

        Uses a bracket stack to determine the exact closing sequence needed.
        This produces the correct order (e.g. }]} not ]}} ) which the naive
        brace-counting approach gets wrong for nested structures.
        """
        truncated = text.rstrip()

        # Strip a dangling partial string value at the end
        # e.g. ..."reason":"Two people are in the foregr  ->  ..."reason":"..."
        truncated = re.sub(r',?\s*"[^"]*$', '', truncated)
        truncated = re.sub(r',\s*$', '', truncated)

        if not truncated.startswith('{'):
            return None

        # Walk the cleaned text with a stack to find exactly what's unclosed
        OPENERS = {'{': '}', '[': ']'}
        CLOSERS = set('}]')
        stack = []
        in_string = False
        escape_next = False

        for ch in truncated:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in OPENERS:
                stack.append(OPENERS[ch])
            elif ch in CLOSERS and stack and stack[-1] == ch:
                stack.pop()

        if not stack:
            return None

        # Close in reverse stack order — this gives the correct nesting sequence
        closing = ''.join(reversed(stack))
        repaired = re.sub(r",\s*([}\]])", r"\1", truncated + closing)

        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"{self.name}.run() not implemented")

    def validate(self, output: dict, **kwargs) -> tuple:
        raise NotImplementedError(f"{self.name}.validate() not implemented")
