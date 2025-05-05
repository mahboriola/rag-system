import asyncio
import json
from io import BytesIO
from typing import Any, Dict, List
from uuid import uuid4

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential
from markitdown import MarkItDown
from openai import AsyncAzureOpenAI, AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qdrant_models

from config.settings import Settings
from services.chunker import TextChunker
from services.llm import OpenAI
from services.vector_database import VectorDatabase

settings = Settings()


class IngestionPipeline:
    """
    A pipeline that processes documents for storage in the RAG system.

    This class coordinates the following steps:
    1. Breaks documents into manageable chunks
    2. Extracts metadata and contextual information
    3. Generates vector embeddings
    4. Stores chunks and metadata in the vector database
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the pipeline with required services."""
        self.chunker = TextChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, clean_html_tags=True
        )
        self.llm = OpenAI()
        self.vector_database = VectorDatabase()

    async def _extract_text_from_pdf(self, pdf: bytes) -> str:
        """
        Extract text from a PDF document.

        Args:
            pdf (bytes): Raw content of the PDF document

        Returns:
            str: Extracted text in markdown format
        """
        try:
            md = MarkItDown()
            result = md.convert(BytesIO(pdf))
            pdf_md = result.markdown
        except:
            pdf_md = ""

        if not pdf_md.strip():
            di_client = DocumentIntelligenceClient(
                endpoint=settings.AZURE_OCR_ENDPOINT,
                credential=AzureKeyCredential(settings.AZURE_OCR_KEY),
            )

            async with di_client:
                poller = await di_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=AnalyzeDocumentRequest(bytes_source=pdf),
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )
                result = await poller.result()
                pdf_md = result.content

        return pdf_md

    async def _extract_metadata(self, text: str) -> Dict[str, str | List[str]]:
        """
        Extract metadata from the provided text.

        Args:
            text (str): Text content of the document

        Returns:
            Dict[str, str | List[str]]: Extracted metadata in JSON format
        """
        response = await self.llm.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant that extracts relevant information from documents.

                    Your task is to analyze the provided document and extract key information.

                    Extract the following information:
                    - Product Name
                    - Keywords

                    The output should be in JSON format with the following structure:
                    {
                        "product_name": "<Product Name>",
                        "keywords": ["<Keyword1>", "<Keyword2>", ...]
                    }
                    Please provide the output in JSON format.
                    Do not include any other text or explanation.
                    """,
                },
                {"role": "user", "content": text},
            ],
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
        )

        response_str = response.choices[0].message.content.lower()
        return json.loads(response_str)

    async def _embed_and_get_qdrant_point(
        self, chunk: str, metadata: Dict[str, Any]
    ) -> qdrant_models.PointStruct:
        """
        Generate vector embedding for a text chunk and create a Qdrant point.

        Args:
            chunk (str): Text chunk
            metadata (Dict[str, Any]): Metadata associated with the chunk

        Returns:
            qdrant_models.PointStruct: Qdrant point with vector embedding and metadata
        """
        # Embed the chunk
        embedding = await self.llm.get_embedding(chunk)

        payload = {"text": chunk}
        payload.update(metadata)

        return qdrant_models.PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload,
        )

    async def _store_chunks_in_vector_db(
        self, chunks: List[str], metadata: Dict[str, Any]
    ) -> None:
        """
        Store text chunks and their metadata in the vector database.

        Args:
            chunks (List[str]): List of text chunks
            metadata (Dict[str, Any]): Metadata associated with the chunks
        """
        point_tasks = []
        for chunk in chunks:
            if chunk.strip():
                point_tasks.append(self._embed_and_get_qdrant_point(chunk, metadata))
        points = await asyncio.gather(*point_tasks)
        await self.vector_database.upsert(points)

    async def process(self, pdf_name: str, pdf_bytes: bytes) -> List[str]:
        """
        Process the input text and return a list of chunks.

        Args:
            pdf_name (str): Name of the PDF file
            pdf_bytes (bytes): Raw content of the PDF file

        Returns:
            List[str]: List of processed text chunks
        """
        pdf_md = await self._extract_text_from_pdf(pdf_bytes)
        metadata = await self._extract_metadata(pdf_md)
        metadata.update({"filename": pdf_name})
        chunks = await self.chunker.split_text(pdf_md)
        await self._store_chunks_in_vector_db(chunks, metadata)

        return chunks
