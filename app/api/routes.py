# app/api/routes.py
from fastapi import APIRouter
from app.api.schemas import ChatRequest, ChatResponse
from src.services.llm_services import LangChainService

from fastapi.responses import StreamingResponse

router = APIRouter()
chat_service = LangChainService()


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
