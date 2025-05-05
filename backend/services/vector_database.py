from typing import List, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qdrant_models

from config.settings import Settings
from services.llm import OpenAI

settings = Settings()


class VectorDatabase:
    """
    A wrapper class for interacting with the Qdrant vector database.

    This class provides high-level operations for:
    - Storing and managing vector embeddings
    - Performing similarity searches
    - Managing metadata and filters
    - Handling collection operations
    """

    def __init__(self):
        """Initialize connection to the Qdrant vector database."""
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
        """
        Assert that the Qdrant collection exists.
        This method checks if the collection is created and raises an error if not.
        """
        qdrant = cls._get_client()
        assert (
            await qdrant.get_collection(settings.QDRANT_COLLECTION_NAME) is not None
        ), "Qdrant Collection is not created"

    @classmethod
    async def create_collection(cls):
        """
        Create the needed Qdrant collection with the specified configuration.
        This method sets up the collection for storing vector embeddings and metadata.
        """
        qdrant = cls._get_client()
        await qdrant.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=settings.VECTOR_DIMENSIONS,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        await qdrant.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="product_name",
            field_schema=qdrant_models.PayloadSchemaType.TEXT,
        )
        await qdrant.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="keywords",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )

    @classmethod
    async def delete_collection(cls) -> None:
        """
        Delete the Qdrant collection.
        This method removes the collection and all its data.
        """
        qdrant = cls._get_client()
        await qdrant.delete_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
        )

    async def create_filters(self, filters_data: List[Tuple]) -> qdrant_models.Filter:
        """
        Create Qdrant filters from filter items.

        Args:
            filters_data (List[Tuple]): List of (key, value) pairs to filter on

        Returns:
            qdrant_models.Filter: Qdrant filter object
        """
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
        """
        Insert or update points in the vector database.

        Args:
            points (List[qdrant_models.PointStruct]): List of points to upsert
        """
        await self.qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
        )

    async def search_context(
        self, query: str, filters: qdrant_models.Filter = None
    ) -> List[qdrant_models.ScoredPoint]:
        """
        Search for similar vectors in the database.

        Args:
            query (str): The query text to search for
            filters (qdrant_models.Filter, optional): Optional filters to apply to the search

        Returns:
            List[qdrant_models.ScoredPoint]: List of matching vectors with their scores
        """
        query_embedding = await self.llm.get_embedding(query)

        search_results = await self.qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=5,
            query_filter=filters,
        )

        return search_results.points
