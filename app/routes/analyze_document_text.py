from fastapi import APIRouter, Body, HTTPException
from app.models import DocumentAnalysisResponse
from app.render.report_renderer import render_html_report
from app.prompts.document_analysis_prompt_doc_only import build_document_analysis_prompt_doc_only
from app.prompts.document_analysis_prompt import build_document_analysis_prompt
from app.services import (
    ask_llm,
    parse_document_analysis_json,
    collect_legal_context_for_document,
)
from .analyze_document_llm_only import (  # переиспользуем уже готовую канонизацию
    CANONICAL_SECTIONS, ALLOWED_SECTION_CODES, EXCLUDED_SECTION_CODES,
    CODE_NORMALIZATION, GRADE_LABEL_MAP, compute_grade,
)
from app.models import SectionScore, KeyRisk, SourceItem

router = APIRouter()

@router.post("/analyze_document_text", response_model=DocumentAnalysisResponse)
def analyze_document_text(
    body: str = Body(..., media_type="text/plain"),
    top_k: int = 10,
    use_rag: bool = True,
):
    document_text = (body or "").strip()
    if not document_text:
        raise HTTPException(status_code=400, detail="Пустое тело запроса (ожидается text/plain)")

    # ШАГ 1: LLM без RAG
    prompt_doc = build_document_analysis_prompt_doc_only(document_text)
    llm_doc_raw = ask_llm(prompt_doc)
    doc_only = parse_document_analysis_json(llm_doc_raw)

    data = {}
    sources = []

    if use_rag:
        # ШАГ 2: RAG
        legal_points = collect_legal_context_for_document(
            document_text=document_text, per_chunk_k=3, max_chunks=5, total_limit=top_k
        )
        if legal_points:
            prompt_rag = build_document_analysis_prompt(document_text, legal_points, doc_only or {})
            llm_rag_raw = ask_llm(prompt_rag)
            data = parse_document_analysis_json(llm_rag_raw) or {}
            # источники
            for i, p in enumerate(legal_points, 1):
                pl = p.get("payload", {}) or {}
                sources.append(SourceItem(
                    rank=i, id=p.get("id"), score=p.get("score"),
                    source=pl.get("source"), chunk_idx=pl.get("chunk_idx")
                ))
        else:
            data = doc_only or {}
    else:
        data = doc_only or {}

    # Извлекаем поля
    summary = (data.get("summary") or "").strip() or "Краткое резюме недоступно (модель не распарсила ответ)."
    recommendations = (data.get("recommendations") or "").strip() or None
    sections_in = data.get("sections", []) or []
    key_risks_in = data.get("key_risks", []) or []

    # Канонизируем 10 секций
    sections_by_code = {}
    for s in sections_in:
        raw_code = (s.get("code") or "").strip()
        if not raw_code:
            continue
        code = CODE_NORMALIZATION.get(raw_code, raw_code)
        if code in EXCLUDED_SECTION_CODES or code not in ALLOWED_SECTION_CODES:
            continue
        title = (s.get("title") or "").strip()
        score = float(s.get("score") or 0.0)
        comment = (s.get("comment") or "").strip() or (s.get("details") or "").strip() or None
        sections_by_code[code] = SectionScore(
            code=code, title=title, score=score, max_score=10.0, grade="unknown", comment=comment
        )

    sections = []
    total_score = 0.0
    total_max = 0.0
    for sec in CANONICAL_SECTIONS:
        code = sec["code"]
        title = sec["title"]
        if code in sections_by_code:
            s_obj = sections_by_code[code]
            if not s_obj.title:
                s_obj.title = title
        else:
            # 👉 по вашей новой логике: ставим среднюю оценку 5/10,
            # если раздел не определён явно
            s_obj = SectionScore(
                code=code, title=title, score=5.0, max_score=10.0, grade="unknown",
                comment="Раздел не был явно оценён моделью; выставлена средняя оценка 5 из 10 по умолчанию."
            )
        total_score += s_obj.score
        total_max += s_obj.max_score
        sections.append(s_obj)

    # Цвет по секциям
    for s in sections:
        pct = (s.score / s.max_score) * 100 if s.max_score > 0 else 0
        s.grade = "green" if pct >= 75 else "yellow" if pct >= 50 else "red"

    grade = compute_grade(total_score, total_max)
    grade_label = GRADE_LABEL_MAP[grade]

    # Риски
    key_risks = []
    for r in key_risks_in:
        raw_code = (r.get("section_code") or "").strip()
        code = CODE_NORMALIZATION.get(raw_code, raw_code)
        if code and code not in ALLOWED_SECTION_CODES:
            continue
        issue = (r.get("issue") or "").strip()
        if not issue:
            continue
        key_risks.append(KeyRisk(
            section_code=code, title=(r.get("title") or "").strip(),
            issue=issue, recommendation=(r.get("recommendation") or "").strip() or None
        ))

    resp = DocumentAnalysisResponse(
        grade=grade, grade_label=grade_label,
        score=total_score, score_max=total_max,
        summary=summary, recommendations=recommendations,
        key_risks=key_risks, sections=sections, sources=sources
    )

    report_path = render_html_report(analysis=resp, original_filename="text/plain", document_text=document_text)
    resp.report_path = report_path
    return resp
