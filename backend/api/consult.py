from fastapi import APIRouter

from core.question_pipeline import QuestionPipeline
from models.consult import ConsultRequest, ConsultResponse

router = APIRouter(tags=["Consult"])


@router.post("/question", response_model=ConsultResponse, status_code=200)
async def consult_files(request: ConsultRequest) -> ConsultResponse:
    """
    Process a question and return an answer with relevant references.

    Args:
        request (ConsultRequest): The request object containing the question to be answered.

    Returns:
        ConsultResponse: An object containing:
            - answer (str): The generated answer to the question
            - references (list): List of relevant references from the source documents

    Example:
        POST /question
        {
            "question": "What is the operating temperature range?"
        }
    """
    pipeline = QuestionPipeline()
    answer, references = await pipeline.answer_question(request.question)
    return {"answer": answer, "references": references}
