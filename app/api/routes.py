# app/api/routes.py
from fastapi import APIRouter, HTTPException
from app.api.schemas import ChatRequest, ChatResponse
from src.services.llm_services import LangChainService, RAGService

from fastapi.responses import StreamingResponse

router = APIRouter()
chat_service = LangChainService()
rag_service = RAGService()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    answer = await chat_service.get_response(request.message, request.session_id)
    return ChatResponse(answer=answer)


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    return StreamingResponse(
        chat_service.get_streaming_response(request.message, request.session_id),
        media_type="text/event-stream",
    )


@router.post("/chat/rag")
async def rag_chat(request: ChatRequest):
    try:
        return StreamingResponse(
            rag_service.get_rag_streaming_response(request.message, request.session_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
