FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Кладём код приложения
COPY app ./app

ENV OLLAMA_URL=http://localhost:11434
ENV QDRANT_URL=http://localhost:6333
ENV QDRANT_COLLECTION=docs
ENV EMBED_MODEL=bge-m3
ENV LLM_MODEL=krith/qwen2.5-32b-instruct:IQ4_XS
ENV OLLAMA_API_MODE=ollama

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
