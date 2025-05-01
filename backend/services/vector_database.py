from typing import List, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qdrant_models

from config.settings import Settings
from services.llm import OpenAI

settings = Settings()


class VectorDatabase:
    def __init__(self):
        self.qdrant = self._get_client()
        self.llm = OpenAI()

    @classmethod
    def _get_client(cls) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=settings.QDRANT_ENDPOINT,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )

    @classmethod
    async def assert_collection(cls):
        qdrant = cls._get_client()
        assert (
            await qdrant.get_collection(settings.QDRANT_COLLECTION_NAME) is not None
        ), "Qdrant Collection is not created"

    @classmethod
    async def create_collection(cls):
        qdrant = cls._get_client()
        await qdrant.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=settings.VECTOR_DIMENSIONS,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    @classmethod
    async def delete_collection(cls) -> None:
        qdrant = cls._get_client()
        await qdrant.delete_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
        )

    async def create_filters(self, filters_data: List[Tuple]) -> qdrant_models.Filter:
        qdrant_filter = qdrant_models.Filter(
            should=[
                qdrant_models.FieldCondition(
                    key=k, match=qdrant_models.MatchValue(value=v)
                )
                for k, v in filters_data
            ]
        )

        return qdrant_filter

    async def upsert(
        self,
        points: List[qdrant_models.PointStruct],
    ) -> None:
        await self.qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
        )

    async def search_context(
        self, query: str, filters: qdrant_models.Filter = None
    ) -> List[qdrant_models.ScoredPoint]:
        query_embedding = await self.llm.get_embedding(query)

        search_results = await self.qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=5,
            query_filter=filters,
        )

        return search_results.points
