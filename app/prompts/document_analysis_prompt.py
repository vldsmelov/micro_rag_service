from pathlib import Path
import textwrap
from typing import List, Dict, Any
import json


# Путь к шаблону
TEMPLATE_PATH = Path(__file__).resolve().parent / "resources" / "document_analysis_prompt.txt"
PROMPT_TEMPLATE = TEMPLATE_PATH.read_text(encoding="utf-8")


def build_document_analysis_prompt(
    document_text: str,
    docs: List[Dict[str, Any]],
    pre_analysis: Dict[str, Any],
) -> str:
    """
    Шаг 2: перепроверка предварительного анализа по договору
    с опорой на фрагменты НПА из Qdrant.
    Документ остаётся главным, НПА — нормативный контекст.
    """
    # Подрежем текст договора
    doc_preview = document_text.strip()
    if len(doc_preview) > 8000:
        doc_preview = doc_preview[:8000] + "\n\n[Документ обрезан по длине для анализа]"

    # Сбор нормативного контекста
    context_chunks = []
    for i, point in enumerate(docs, start=1):
        payload = point.get("payload", {}) or {}
        text = payload.get("text", "")
        source = payload.get("source", "неизвестный файл")

        short_text = textwrap.shorten(text, width=1200, placeholder=" ...")

        context_chunks.append(
            f"[{i}] Источник: {source}\n"
            f"Фрагмент:\n{short_text}\n"
        )

    legal_context = "\n\n".join(context_chunks)

    # JSON предварительного анализа (Шаг 1)
    prelim_json_str = json.dumps(pre_analysis, ensure_ascii=False, indent=2)
    if len(prelim_json_str) > 6000:
        prelim_json_str = prelim_json_str[:6000] + "\n... [обрезано]"

    # Подстановка в шаблон (без .format, без f-строк — ничего не ломается)
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("{{DOCUMENT_TEXT}}", doc_preview)
    prompt = prompt.replace("{{LEGAL_CONTEXT}}", legal_context)
    prompt = prompt.replace("{{PRE_ANALYSIS_JSON}}", prelim_json_str)

    return prompt
