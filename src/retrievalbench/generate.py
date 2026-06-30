from typing import Protocol

from openai import AsyncOpenAI

from retrievalbench.config import GenerationConfig
from retrievalbench.model import RetrievedChunk


class Generator(Protocol):
    name: str

    async def generate(self, query: str, context: list[RetrievedChunk]) -> str: ...


class OpenAIGenerator:
    """Grounded answer generation via Chat Completions."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature  # G4: keep at 0 for reproducibility
        self.client = AsyncOpenAI()

    async def generate(self, query: str, context: list[RetrievedChunk]) -> str:
        context_text = "\n\n".join(
            f"[{i}] {chunk.text}" for i, chunk in enumerate(context)
        )

        system = (
            "You are a retrieval QA assistant. Answer the question using ONLY the "
            "provided context. If the answer is not in the context, reply exactly: "
            '"I don\'t know." Do not use any outside knowledge.'
        )
        user = f"Context:\n{context_text}\n\nQuestion: {query}"

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        return response.choices[0].message.content or ""


# temperature is deliberately not threaded through config (G4: stays at 0).
# Only `model` varies per experiment.
_GENERATORS: dict[str, type[Generator]] = {
    "openai": OpenAIGenerator,
}


def build_generator(cfg: GenerationConfig) -> Generator:
    cls = _GENERATORS[cfg.type]
    return cls(model=cfg.model)
