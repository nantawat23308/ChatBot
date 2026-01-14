from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str  # Critical for tracking "Memory" between requests


class ChatResponse(BaseModel):
    answer: str
