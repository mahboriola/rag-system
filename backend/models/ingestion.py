from pydantic import BaseModel


class IngestionResponse(BaseModel):
    message: str
    documents_indexed: int
    total_chunks: int