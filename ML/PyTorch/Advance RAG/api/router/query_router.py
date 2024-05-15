from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.rag.chain import ChainFactory

router = APIRouter(
    prefix="/query",
    tags=["Query"],
)


@router.get("")
def query(question: str, chain: ChainFactory) -> StreamingResponse:
    """Ask a question to the our AI model with RAG."""
    response = chain.stream(question)

    return StreamingResponse(
        response,
        media_type="text/event-stream",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Type": "text/event-stream",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        },
    )
