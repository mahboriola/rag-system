from typing import List

from openai import AsyncAzureOpenAI, AsyncOpenAI

from config.settings import Settings

settings = Settings()


class OpenAI(AsyncAzureOpenAI, AsyncOpenAI):
    """
    A wrapper class for interacting with OpenAI's API services.

    This class provides unified access to:
    - Text embedding generation
    - Chat completions
    - Automatic handling of Azure OpenAI and standard OpenAI endpoints
    """

    def __init__(self):
        """Initialize the OpenAI client based on configuration settings."""
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
        """
        Generate an embedding vector for the given text.

        Args:
            text (str): The text to generate an embedding for

        Returns:
            list[float]: The embedding vector
        """
        embedding = await self.client.embeddings.create(
            input=text,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )

        return embedding.data[0].embedding
