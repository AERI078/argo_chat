# agents/llm_caller.py — single point for all Groq LLM calls
# swap model or provider here without touching anything else

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL


class LLMCaller:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

    def call(self, messages: list[dict], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
