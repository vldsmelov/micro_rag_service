import textwrap
from typing import List, Dict, Any


def build_ask_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    """Промпт для эндпоинта /ask (вопрос → ответ)."""
    context_chunks = []
    for i, point in enumerate(docs, start=1):
        payload = point.get("payload", {}) or {}
        text = payload.get("text", "")
        source = payload.get("source", "неизвестный файл")

        # защита от гигантских кусков
        short_text = textwrap.shorten(
            text,
            width=1500,
            placeholder=" ...",
        )

        context_chunks.append(
            f"[{i}] Источник: {source}\n"
            f"Фрагмент:\n{short_text}\n"
        )

    context = "\n\n".join(context_chunks)

    prompt = f"""
Ты — юридический ассистент по российскому праву.

Тебе дан вопрос пользователя и выдержки из нормативных правовых актов.
Отвечай строго на основе этих выдержек. Если в контексте нет нужной информации,
честно напиши, что в предоставленных документах ответ не найден.

Вопрос пользователя:
{question}

Контекст документов:
{context}

Задача:
1. Дай краткий и чёткий ответ на вопрос (2–5 предложений).
2. В конце выведи список использованных источников в формате:
   [N] краткое описание — путь к файлу

Не выдумывай номера статей или реквизиты, если их нет в тексте фрагмента.
Отвечай по-русски.
"""
    return textwrap.dedent(prompt).strip()
