from typing import Protocol

from openai import AsyncOpenAI

from retrievalbench.config import EmbeddingConfig


class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAITextEmbedderSmall:
    """
    Embeds texts using OpenAI text embedder small
    """

    name = "openAI_text_embedder_small"
    dim = 1536

    def __init__(self):
        self.client = AsyncOpenAI()

    async def embed(self, texts: list[str]) -> list[list[float]]:

        response = await self.client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )

        return [embd.embedding for embd in response.data]


# Only one embedder today; the registry still makes it config-swappable and
# ready for a second entry. OpenAITextEmbedderSmall takes no constructor args
# (model is fixed internally), so the builder just instantiates it.
_EMBEDDERS: dict[str, type[Embedder]] = {
    "openai_small": OpenAITextEmbedderSmall,
}


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    cls = _EMBEDDERS[cfg.type]
    return cls()
