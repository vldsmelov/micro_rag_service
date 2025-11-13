import os
import json
from typing import List, Any, Dict

import requests
from fastapi import HTTPException


# ---------- Конфиг ----------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "krith/qwen2.5-32b-instruct:IQ4_XS")

# "ollama" -> /api/embed + /api/chat
# "openai" -> /v1/embeddings + /v1/chat/completions
API_MODE = os.getenv("OLLAMA_API_MODE", "ollama").lower()


# ---------- Клиент эмбеддингов и LLM ----------

def get_embedding(text: str) -> List[float]:
    """Получить эмбеддинг из Ollama или OpenAI-совместимого API."""
    if API_MODE == "ollama":
        url = f"{OLLAMA_URL.rstrip('/')}/api/embed"
        payload = {"model": EMBED_MODEL, "input": text}
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка вызова Ollama /api/embed: {resp.status_code} {resp.text}",
            )
        data = resp.json()
        try:
            return data["embeddings"][0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Неожиданный формат ответа /api/embed: {e}")
    else:
        url = f"{OLLAMA_URL.rstrip('/')}/v1/embeddings"
        payload = {"model": EMBED_MODEL, "input": text}
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка вызова /v1/embeddings: {resp.status_code} {resp.text}",
            )
        data = resp.json()
        try:
            return data["data"][0]["embedding"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Неожиданный формат ответа /v1/embeddings: {e}")


def ask_llm(prompt: str) -> str:
    """Получить текстовый ответ от LLM."""
    if API_MODE == "ollama":
        url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка вызова Ollama /api/chat: {resp.status_code} {resp.text}",
            )
        data = resp.json()
        try:
            return data["message"]["content"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Неожиданный формат ответа /api/chat: {e}")
    else:
        url = f"{OLLAMA_URL.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка вызова /v1/chat/completions: {resp.status_code} {resp.text}",
            )
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Неожиданный формат ответа /v1/chat/completions: {e}")


# ---------- Qdrant ----------

def search_qdrant(vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Поиск ближайших документов в Qdrant."""
    url = f"{QDRANT_URL.rstrip('/')}/collections/{QDRANT_COLLECTION}/points/search"
    payload = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка поиска в Qdrant: {resp.status_code} {resp.text}",
        )
    data = resp.json()
    return data.get("result", [])


# ---------- Разрезание документа и сбор правового контекста ----------

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200, max_chunks: int = 5) -> List[str]:
    """Грубая нарезка текста на куски для эмбеддингов."""
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length and len(chunks) < max_chunks:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)

    return chunks


def collect_legal_context_for_document(
    document_text: str,
    per_chunk_k: int = 3,
    max_chunks: int = 5,
    total_limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Для документа:
    - режем на куски,
    - для каждого кусочка берём эмбеддинг и ищем в Qdrant,
    - агрегируем результаты.
    """
    chunks = split_into_chunks(document_text, max_chunks=max_chunks)
    all_points: Dict[Any, Dict[str, Any]] = {}

    for chunk in chunks:
        embedding = get_embedding(chunk)
        results = search_qdrant(embedding, limit=per_chunk_k)
        for point in results:
            point_id = point.get("id")
            score = point.get("score") or 0.0
            if point_id in all_points:
                if score > (all_points[point_id].get("score") or 0.0):
                    all_points[point_id] = point
            else:
                all_points[point_id] = point

    sorted_points = sorted(
        all_points.values(),
        key=lambda p: p.get("score") or 0.0,
        reverse=True,
    )

    return sorted_points[:total_limit]


# ---------- Парсер JSON для анализа документа ----------

def parse_document_analysis_json(raw: str) -> Dict[str, Any]:
    """
    Пытается распарсить ответ модели как JSON.
    Возвращает dict (может быть пустым, если всё плохо).
    """
    # 1. Прямая попытка
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Вырезаем от первой { до последней }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}

    return {}
