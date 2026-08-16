from abc import ABC, abstractmethod

import httpx

from app.config import Settings


SYSTEM_PROMPT = """You are an internal employee-information assistant.
Answer only from the supplied context. If the context does not contain enough
information, say so clearly. Never invent employee details. Keep the answer concise
and mention employee IDs when they help disambiguate people."""


class AnswerGenerator(ABC):
    name: str

    @abstractmethod
    async def generate(self, question: str, contexts: list[str]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(AnswerGenerator):
    name = "extractive"

    async def generate(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "I could not find relevant employee information for that question."
        intro = "I found the following relevant employee information:"
        return intro + "\n- " + "\n- ".join(contexts)


class OpenAICompatibleGenerator(AnswerGenerator):
    name = "openai"

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.base_url = settings.openai_base_url.rstrip("/")

    async def generate(self, question: str, contexts: list[str]) -> str:
        context = "\n\n".join(contexts)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        }
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()


class OllamaGenerator(AnswerGenerator):
    name = "ollama"

    def __init__(self, settings: Settings):
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url.rstrip("/")

    async def generate(self, question: str, contexts: list[str]) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n" + "\n\n".join(contexts)
        prompt += f"\n\nQuestion: {question}\nAnswer:"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"].strip()


def create_generator(settings: Settings) -> AnswerGenerator:
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAICompatibleGenerator(settings)
    if provider == "ollama":
        return OllamaGenerator(settings)
    if provider == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

