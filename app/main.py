from fastapi import FastAPI

from routes.ask import router as ask_router
from routes.analyze_document import router as analyze_router


app = FastAPI(
    title="ИИ-юрист: Qdrant + Ollama",
    description="RAG-сервис: отвечает на вопросы и анализирует документы на соответствие НПА.",
    version="0.3.0",
)

app.include_router(ask_router)
app.include_router(analyze_router)
