from typing import List, Dict

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models import (
    DocumentAnalysisResponse,
    SourceItem,
    SectionScore,
    KeyRisk,
)
from app.services import (
    ask_llm,
    parse_document_analysis_json,
)
from app.prompts.document_analysis_prompt_doc_only import (
    build_document_analysis_prompt_doc_only,
)
from app.render.report_renderer import render_html_report


router = APIRouter()

# Канонический список секций (10 штук по 10 баллов)
CANONICAL_SECTIONS = [
    {"code": "parties",            "title": "Идентификация сторон, полномочия, реквизиты",                 "max_score": 10},
    {"code": "subject",            "title": "Предмет/результат, измеримость (SOW/ТЗ)",                     "max_score": 10},
    {"code": "timeline_acceptance","title": "Сроки, поставка/оказание, приемка, milestones",               "max_score": 10},
    {"code": "price_payment",      "title": "Цена, налоги, порядок платежей, удержания",                   "max_score": 10},
    {"code": "liability",          "title": "Ответственность, неустойки/штрафы, лимиты (cap)",             "max_score": 10},
    {"code": "warranties",         "title": "Гарантии и заверения сторон",                                 "max_score": 10},
    {"code": "force_majeure",      "title": "Форс-мажор",                                                   "max_score": 10},
    {"code": "termination",        "title": "Изменение/расторжение, односторонний отказ",                  "max_score": 10},
    {"code": "governing_law",      "title": "Применимое право и подсудность/арбитраж",                     "max_score": 10},
    {"code": "documents_priority", "title": "Конфликты/приоритет документов, приложения",                  "max_score": 10},
]

ALLOWED_SECTION_CODES = {s["code"] for s in CANONICAL_SECTIONS}
EXCLUDED_SECTION_CODES = {"ip", "personal_data", "signatures_form"}  # на будущее

# Нормализация кодов, которые любит придумывать модель
CODE_NORMALIZATION: Dict[str, str] = {
    "terms": "timeline_acceptance",
    "delivery_conditions": "timeline_acceptance",
    "delivery_and_acceptance": "timeline_acceptance",

    "price": "price_payment",
    "price_and_payment": "price_payment",

    "dispute_resolution": "governing_law",
    "applicable_law": "governing_law",

    "quality": "warranties",
    "obligations": "liability",
    "termination_conditions": "termination",
}

GRADE_LABEL_MAP = {
    "green": "Зелёный",
    "yellow": "Жёлтый",
    "red": "Красный",
    "unknown": "Не определён",
}


def compute_grade(total_score: float, max_score: float) -> str:
    if max_score <= 0:
        return "unknown"
    percent = (total_score / max_score) * 100.0
    if percent >= 75:
        return "green"
    elif percent >= 50:
        return "yellow"
    else:
        return "red"


@router.post("/analyze_document_llm", response_model=DocumentAnalysisResponse)
def analyze_document_llm_only(
    document: UploadFile = File(..., description="Plain text .txt документ"),
):
    """
    Анализ договора силами LLM БЕЗ RAG.
    - Смотрим только в текст договора.
    - Оцениваем по 10 фиксированным осям.
    - Считаем суммарный скор и grade на бэкенде.
    - Источники (sources) остаются пустыми.
    """
    raw_bytes = document.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл")

    # Декодирование
    try:
        document_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        document_text = raw_bytes.decode("cp1251", errors="replace")

    # Шаг 1 (doc-only) — используем как основной и единственный
    prompt_doc = build_document_analysis_prompt_doc_only(document_text)
    llm_raw = ask_llm(prompt_doc)
    data = parse_document_analysis_json(llm_raw)

    if not data:
        summary = "Модель не смогла корректно распарсить ответ. Проверьте формат промпта и ответа."
        recommendations = None
        sections_data = []
        key_risks_data = []
    else:
        summary = (data.get("summary") or "").strip()
        recommendations = (data.get("recommendations") or "").strip() or None
        sections_data = data.get("sections", []) or []
        key_risks_data = data.get("key_risks", []) or []

    # ---------- Преобразуем секции в 10 канонических осей ----------
    sections_by_code: dict[str, SectionScore] = {}

    for s in sections_data:
        raw_code = (s.get("code") or "").strip()
        if not raw_code:
            continue

        code = CODE_NORMALIZATION.get(raw_code, raw_code)

        if code in EXCLUDED_SECTION_CODES:
            continue
        if code not in ALLOWED_SECTION_CODES:
            continue

        title = (s.get("title") or "").strip()
        s_score = float(s.get("score") or 0.0)
        s_max = 10.0
        comment = (s.get("comment") or "").strip() or (s.get("details") or "").strip() or None

        sections_by_code[code] = SectionScore(
            code=code,
            title=title,
            score=s_score,
            max_score=s_max,
            grade="unknown",
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
        else:
            s_obj = SectionScore(
                code=code,
                title=default_title,
                score=5.0,
                max_score=max_score,
                grade="unknown",
                comment="Раздел не был явно оценён моделью или явно не раскрыт в тексте договора.",
            )

        total_score += s_obj.score
        total_max += s_obj.max_score
        sections.append(s_obj)

    # считаем grade по каждой секции
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

    # общий grade
    grade = compute_grade(total_score, total_max)
    grade_label = GRADE_LABEL_MAP[grade]

    # ---------- Ключевые риски ----------
    key_risks: List[KeyRisk] = []
    for r in key_risks_data:
        raw_code = (r.get("section_code") or "").strip()
        code = CODE_NORMALIZATION.get(raw_code, raw_code)

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

    # ---------- Источники НПА (для LLM-only — пусто) ----------
    sources: List[SourceItem] = []

    # ---------- Собираем ответ ----------
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

    # HTML-отчёт
    report_path = render_html_report(
        analysis=resp,
        original_filename=document.filename,
        document_text=document_text,
    )
    resp.report_path = report_path

    return resp
