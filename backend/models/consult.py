from typing import List
from pydantic import BaseModel, Field


class ConsultRequest(BaseModel):
    question: str


class ConsultResponse(BaseModel):
    answer: str
    references: List[str]
