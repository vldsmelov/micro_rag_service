from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models import DocumentAnalysisResponse, SourceItem
from app.services import (
    collect_legal_context_for_document,
    ask_llm,
    parse_analysis_json,
)
from app.prompts.document_analysis_prompt import build_document_analysis_prompt


router = APIRouter()


@router.post("/analyze_document", response_model=DocumentAnalysisResponse)
def analyze_document(
    document: UploadFile = File(..., description="Plain text .txt документ"),
    top_k: int = 10,
):
    """
    ИИ-юрист: анализирует .txt документ и оценивает соответствие нормативным актам.
    """
    raw_bytes = document.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл")

    # Декодирование текста
    try:
        document_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        document_text = raw_bytes.decode("cp1251", errors="replace")

    # Собираем релевантный контекст из Qdrant
    legal_points = collect_legal_context_for_document(
        document_text=document_text,
        per_chunk_k=3,
        max_chunks=5,
        total_limit=top_k,
    )

    if not legal_points:
        raise HTTPException(status_code=404, detail="Qdrant не вернул ни одного результата для анализа документа")

    # Строим промпт и спрашиваем LLM
    prompt = build_document_analysis_prompt(document_text, legal_points)
    llm_raw = ask_llm(prompt)
    grade, grade_label, explanation, recommendations = parse_analysis_json(llm_raw)

    # Источники
    sources: List[SourceItem] = []
    for i, point in enumerate(legal_points, start=1):
        payload = point.get("payload", {}) or {}
        sources.append(SourceItem(
            rank=i,
            id=point.get("id"),
            score=point.get("score"),
            source=payload.get("source"),
            chunk_idx=payload.get("chunk_idx"),
        ))

    return DocumentAnalysisResponse(
        grade=grade,
        grade_label=grade_label,
        explanation=explanation,
        recommendations=recommendations,
        sources=sources,
    )
