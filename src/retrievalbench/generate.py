from collections.abc import Callable
from typing import Protocol

from openai import AsyncOpenAI

from retrievalbench.config import GenerationConfig
from retrievalbench.model import RetrievedChunk

# USD per 1M tokens, (input, output). Hardcoded because provider prices are not
# queryable at runtime — and they drift, so they live in ONE place with a date.
# Verified 2026-08. Adding a generator model REQUIRES adding its price here;
# an unknown model raises at construction rather than silently costing $0.00
# (a silent zero is exactly what made `total_cost_usd` meaningless before).
_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = _PRICE_PER_1M[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class Generator(Protocol):
    name: str

    # (answer, cost_usd) per Design §5.6 — the runner needs the cost to make
    # `total_cost_usd` real, which the recommendation engine budgets against.
    async def generate(
        self, query: str, context: list[RetrievedChunk]
    ) -> tuple[str, float]: ...


class OpenAIGenerator:
    """Grounded answer generation via Chat Completions."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        # Construction-time validation (same philosophy as config's extra="forbid"):
        # an unpriced model fails HERE, loudly, instead of quietly reporting $0.00
        # cost for a whole run and corrupting every recommendation downstream.
        if model not in _PRICE_PER_1M:
            raise ValueError(
                f"No price entry for generator model {model!r}. Add it to "
                f"generate._PRICE_PER_1M (known: {sorted(_PRICE_PER_1M)})."
            )
        self.model = model
        self.temperature = temperature  # G4: keep at 0 for reproducibility
        self.client = AsyncOpenAI()

    async def generate(
        self, query: str, context: list[RetrievedChunk]
    ) -> tuple[str, float]:
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

        # Extract the fields we need, never keep the response object (async
        # hygiene). usage is None only if the provider omits it — treat as 0.
        answer = response.choices[0].message.content or ""
        usage = response.usage
        cost = (
            token_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
            if usage is not None
            else 0.0
        )
        return answer, cost


# temperature is deliberately not threaded through config (G4: stays at 0).
# Only `model` varies per experiment.
#
# Typed as Callable, not `type[Generator]`: a Protocol describes instances, so
# `type[Generator]` promises nothing about the CONSTRUCTOR and mypy rejects
# `cls(model=...)`. Callable[[str], Generator] states what the registry actually
# holds — something you call with a model name to get a Generator.
_GENERATORS: dict[str, Callable[[str], Generator]] = {
    "openai": OpenAIGenerator,
}


def build_generator(cfg: GenerationConfig) -> Generator:
    factory = _GENERATORS[cfg.type]
    return factory(cfg.model)
