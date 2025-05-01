from fastapi import APIRouter

from core.question_pipeline import QuestionPipeline
from models.consult import ConsultRequest, ConsultResponse

router = APIRouter(tags=["Consult"])


@router.post("/question", response_model=ConsultResponse, status_code=200)
async def consult_files(request: ConsultRequest):
    pipeline = QuestionPipeline()

    answer, references = await pipeline.answer_question(request.question)

    return {"answer": answer, "references": references}
