import asyncio
from typing import List

from fastapi import APIRouter, File, UploadFile

from core.ingestion_pipeline import IngestionPipeline
from models.ingestion import IngestionResponse

router = APIRouter(tags=["Document Ingestion"])


@router.post("/documents", response_model=IngestionResponse, status_code=200)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF documents.

    Args:
        files: List of PDF files to be uploaded

    Returns:
        dict: Status of the upload operation
    """
    pipeline = IngestionPipeline()

    ingestion_tasks = []
    for f in files:
        if f.content_type == "application/pdf":
            ingestion_tasks.append(pipeline.process(f.filename, await f.read()))

    results = await asyncio.gather(*ingestion_tasks)
    total_chunks = sum(len(chunks) for chunks in results)

    return {
        "message": "Documents processed successfully",
        "documents_indexed": len(results),
        "total_chunks": total_chunks,
    }
