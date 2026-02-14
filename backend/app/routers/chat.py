from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import ChatRequest, ChatResponse
from app.services.llm import generate_openai_commentary

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/commentary", response_model=ChatResponse)
async def commentary(payload: ChatRequest, request: Request):
    settings = request.app.state.settings
    out = await generate_openai_commentary(payload.prompt, payload.context, settings)
    return ChatResponse(**out)
