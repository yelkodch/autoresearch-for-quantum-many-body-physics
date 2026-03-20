"""OpenAI-compatible chat completion client supporting Groq and Gemini."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def chat_completion(prompt: str, model: str | None = None, temperature: float = 0.4) -> str:
    """Call an OpenAI-compatible API and return the assistant message."""
    gemini_default = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    groq_default = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Try Gemini first, then Groq
    providers = [
        (
            "gemini",
            os.getenv("GEMINI_API_KEY"),
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            [model] if model else [gemini_default, "gemini-2.5-flash", "gemini-2.5-pro"],
        ),
        (
            "groq",
            os.getenv("GROQ_API_KEY"),
            "https://api.groq.com/openai/v1",
            [model] if model else [groq_default],
        ),
    ]

    last_error = None
    for provider_name, api_key, base_url, model_names in providers:
        if not api_key:
            continue
        for model_name in model_names:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=16384,
                )
                text = response.choices[0].message.content
                if text:
                    return text.strip()
            except Exception as e:
                last_error = e
                print(f"  [warn] {provider_name} model {model_name} failed: {e}")
                continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
