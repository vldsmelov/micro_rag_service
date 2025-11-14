from fastapi import FastAPI

from app.routes.ask import router as ask_router
from app.routes.analyze_document import router as analyze_router
from app.routes.analyze_document_llm_only import router as analyze_llm_router
from app.routes.analyze_document_text import router as analyze_text_router


app = FastAPI(
    title="ИИ-юрист: Qdrant + Ollama",
    description="RAG-сервис: отвечает на вопросы и анализирует документы на соответствие НПА.",
    version="0.3.0",
)

app.include_router(ask_router)
app.include_router(analyze_router)
app.include_router(analyze_llm_router)
app.include_router(analyze_text_router)
