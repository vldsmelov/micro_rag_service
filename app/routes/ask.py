from typing import List

from fastapi import APIRouter, HTTPException

from models import AskRequest, AskResponse, SourceItem
from services import get_embedding, search_qdrant, ask_llm
from prompts.ask_prompt import build_ask_prompt


router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Простой RAG: вопрос -> ответ + ссылки на НПА."""
    embedding = get_embedding(req.question)
    results = search_qdrant(embedding, limit=req.top_k)

    if not results:
        raise HTTPException(status_code=404, detail="Qdrant не вернул ни одного результата")

    prompt = build_ask_prompt(req.question, results)
    answer = ask_llm(prompt)

    sources: List[SourceItem] = []
    for i, point in enumerate(results, start=1):
        payload = point.get("payload", {}) or {}
        sources.append(SourceItem(
            rank=i,
            id=point.get("id"),
            score=point.get("score"),
            source=payload.get("source"),
            chunk_idx=payload.get("chunk_idx"),
        ))

    return AskResponse(answer=answer, sources=sources)
