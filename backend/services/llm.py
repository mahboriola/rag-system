from typing import List

from openai import AsyncAzureOpenAI, AsyncOpenAI

from config.settings import Settings

settings = Settings()


class OpenAI(AsyncAzureOpenAI, AsyncOpenAI):
    def __init__(self):
        if settings.OPENAI_TYPE == "azure":
            self.client = AsyncAzureOpenAI(
                azure_endpoint=settings.OPENAI_ENDPOINT,
                api_key=settings.OPENAI_API_KEY,
                api_version=settings.OPENAI_API_VERSION,
            )
        else:
            self.client = AsyncOpenAI(
                base_url=settings.OPENAI_ENDPOINT,
                api_key=settings.OPENAI_API_KEY,
            )

    async def get_embedding(self, text: str) -> List[float]:
        embedding = await self.client.embeddings.create(
            input=text,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )

        return embedding.data[0].embedding
