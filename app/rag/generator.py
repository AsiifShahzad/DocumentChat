# rag/generator.py

import os
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Fallback order: best to worst
GROQ_MODELS = [
    "llama-3.3-70b-versatile",  # most capable
    "llama-3.1-8b-instant",     # faster, lighter
    "gemma2-9b-it"              # fallback
]


def generate_answer(prompt: str) -> str:

    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty — cannot send to Groq")

    if len(prompt) > 6000:
        prompt = prompt[:6000]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    last_error = None

    for model in GROQ_MODELS:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided document context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "max_tokens": 1024
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload)

        if response.ok:
            print(f"Model used: {model}")
            data = response.json()
            return data["choices"][0]["message"]["content"]

        # If model is decommissioned or unavailable, try next one
        error = response.json().get("error", {})
        print(f"Model {model} failed: {error.get('message', response.status_code)}")
        last_error = response

    last_error.raise_for_status()