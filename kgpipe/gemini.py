"""Direct Gemini calls for the stages outside LightRAG (merge decisions, hub checks)."""
from __future__ import annotations

import json
import os
import threading
import time

import requests

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class Gemini:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ["AI_STUDIO_KEY"]
        self.calls: dict[str, int] = {}
        self.tokens = {"input": 0, "output": 0}
        self._lock = threading.Lock()
        self.session = requests.Session()

    def json_call(self, kind: str, prompt: str, schema: dict, system: str | None = None) -> dict:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseSchema": schema, "temperature": 0},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        for attempt in range(6):
            r = self.session.post(f"{BASE}/{self.model}:generateContent",
                                  headers={"x-goog-api-key": self.key}, json=body, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            r.raise_for_status()
            data = r.json()
            usage = data.get("usageMetadata", {})
            with self._lock:
                self.calls[kind] = self.calls.get(kind, 0) + 1
                self.tokens["input"] += usage.get("promptTokenCount", 0)
                self.tokens["output"] += usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            return json.loads(text)
        raise RuntimeError(f"Gemini {self.model} kept failing: {r.status_code} {r.text[:300]}")
