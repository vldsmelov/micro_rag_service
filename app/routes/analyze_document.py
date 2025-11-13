from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models import (
    DocumentAnalysisResponse,
    SourceItem,
    SectionScore,
    KeyRisk,
)
from app.services import (
    collect_legal_context_for_document,
    ask_llm,
    parse_document_analysis_json,
)
from app.prompts.document_analysis_prompt import build_document_analysis_prompt
from app.render.report_renderer import render_html_report


router = APIRouter()

# Канонический список секций и их порядок (10 штук по 10 баллов)
CANONICAL_SECTIONS = [
    {
        "code": "parties",
        "title": "Идентификация сторон, полномочия, реквизиты",
        "max_score": 10,
    },
    {
        "code": "subject",
        "title": "Предмет/результат, измеримость (SOW/ТЗ)",
        "max_score": 10,
    },
    {
        "code": "timeline_acceptance",
        "title": "Сроки, поставка/оказание, приемка, milestones",
        "max_score": 10,
    },
    {
        "code": "price_payment",
        "title": "Цена, налоги, порядок платежей, удержания",
        "max_score": 10,
    },
    {
        "code": "liability",
        "title": "Ответственность, неустойки/штрафы, лимиты (cap)",
        "max_score": 10,
    },
    {
        "code": "warranties",
        "title": "Гарантии и заверения сторон",
        "max_score": 10,
    },
    {
        "code": "force_majeure",
        "title": "Форс-мажор",
        "max_score": 10,
    },
    {
        "code": "termination",
        "title": "Изменение/расторжение, односторонний отказ",
        "max_score": 10,
    },
    {
        "code": "governing_law",
        "title": "Применимое право и подсудность/арбитраж",
        "max_score": 10,
    },
    {
        "code": "documents_priority",
        "title": "Конфликты/приоритет документов, приложения",
        "max_score": 10,
    },
]

ALLOWED_SECTION_CODES = {s["code"] for s in CANONICAL_SECTIONS}
EXCLUDED_SECTION_CODES = {"ip", "personal_data", "signatures_form"}  # на всякий случай

GRADE_LABEL_MAP = {
    "green": "Зелёный",
    "yellow": "Жёлтый",
    "red": "Красный",
    "unknown": "Не определён",
}


def compute_grade(total_score: float, max_score: float) -> str:
    """Простейшая шкала: <50 — red, 50–75 — yellow, >75 — green."""
    if max_score <= 0:
        return "unknown"
    percent = (total_score / max_score) * 100.0
    if percent >= 75:
        return "green"
    elif percent >= 50:
        return "yellow"
    else:
        return "red"


@router.post("/analyze_document", response_model=DocumentAnalysisResponse)
def analyze_document(
    document: UploadFile = File(..., description="Plain text .txt документ"),
    top_k: int = 10,
):
    """
    ИИ-юрист: анализирует .txt документ и оценивает его по 10 фиксированным вопросам.
    Документ — основной объект оценки, Qdrant — нормативный контекст.
    """
    raw_bytes = document.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл")

    # Декодирование текста
    try:
        document_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        document_text = raw_bytes.decode("cp1251", errors="replace")

    # Релевантный контекст из Qdrant
    legal_points = collect_legal_context_for_document(
        document_text=document_text,
        per_chunk_k=3,
        max_chunks=5,
        total_limit=top_k,
    )
    if not legal_points:
        raise HTTPException(
            status_code=404,
            detail="Qdrant не вернул ни одного результата для анализа документа",
        )

    # Запрос к LLM
    prompt = build_document_analysis_prompt(document_text, legal_points)
    llm_raw = ask_llm(prompt)
    data = parse_document_analysis_json(llm_raw)

    summary = (data.get("summary") or "").strip() or llm_raw
    recommendations = (data.get("recommendations") or "").strip() or None
    sections_data = data.get("sections", []) or []
    key_risks_data = data.get("key_risks", []) or []

    # ---- Секции: сначала то, что вернула модель, потом гарантируем 10 штук ----
    sections_by_code: dict[str, SectionScore] = {}

    for s in sections_data:
        code = (s.get("code") or "").strip()
        if not code:
            continue
        if code in EXCLUDED_SECTION_CODES:
            continue
        if code not in ALLOWED_SECTION_CODES:
            continue

        title = (s.get("title") or "").strip()
        s_score = float(s.get("score") or 0.0)
        s_max = float(s.get("max_score") or 10.0)  # если модель пришлёт, но мы всё равно зафиксируем 10
        comment = (s.get("comment") or "").strip() or None

        sections_by_code[code] = SectionScore(
            code=code,
            title=title,
            score=s_score,
            max_score=s_max,
            grade="unknown",  # пока без цвета — посчитаем позже
            comment=comment,
        )

    sections: List[SectionScore] = []
    total_score = 0.0
    total_max = 0.0

    for sec in CANONICAL_SECTIONS:
        code = sec["code"]
        default_title = sec["title"]
        max_score = float(sec["max_score"])

        if code in sections_by_code:
            s_obj = sections_by_code[code]
            if not s_obj.title:
                s_obj.title = default_title
            s_obj.max_score = max_score
        else:
            # модель не оценила — внесём секцию с 0 баллов
            s_obj = SectionScore(
                code=code,
                title=default_title,
                score=0.0,
                max_score=max_score,
                grade="unknown",
                comment="Раздел не был явно оценён моделью или явно не раскрыт в тексте договора.",
            )

        # накопим сумму баллов
        total_score += s_obj.score
        total_max += s_obj.max_score
        sections.append(s_obj)

    # ---- Теперь считаем цвет по каждой секции (на основе её процента) ----
    for s in sections:
        if s.max_score <= 0:
            s.grade = "unknown"
            continue
        percent = (s.score / s.max_score) * 100.0
        if percent >= 75:
            s.grade = "green"
        elif percent >= 50:
            s.grade = "yellow"
        else:
            s.grade = "red"

    # ---- Общий grade считаем ТОЛЬКО на бэкенде ----
    grade = compute_grade(total_score, total_max)
    grade_label = GRADE_LABEL_MAP[grade]

    # ---- Ключевые риски ----
    key_risks: List[KeyRisk] = []
    for r in key_risks_data:
        code = (r.get("section_code") or "").strip()
        if code in EXCLUDED_SECTION_CODES:
            continue
        if code and code not in ALLOWED_SECTION_CODES:
            continue

        title = (r.get("title") or "").strip()
        issue = (r.get("issue") or "").strip()
        if not issue:
            continue

        recommendation_r = (r.get("recommendation") or "").strip() or None

        key_risks.append(KeyRisk(
            section_code=code,
            title=title,
            issue=issue,
            recommendation=recommendation_r,
        ))

    # ---- Источники НПА ----
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

    # Собираем ответ
    resp = DocumentAnalysisResponse(
        grade=grade,
        grade_label=grade_label,
        score=total_score,
        score_max=total_max,
        summary=summary,
        recommendations=recommendations,
        key_risks=key_risks,
        sections=sections,
        sources=sources,
    )

    # Рендерим HTML
    report_path = render_html_report(
        analysis=resp,
        original_filename=document.filename,
        document_text=document_text,
    )
    resp.report_path = report_path

    return resp
