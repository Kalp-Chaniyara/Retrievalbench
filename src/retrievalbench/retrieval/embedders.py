from typing import Protocol

from openai import AsyncOpenAI


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


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
