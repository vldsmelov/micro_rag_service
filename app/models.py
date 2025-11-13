from typing import List, Any, Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceItem(BaseModel):
    rank: int
    id: Any
    score: Optional[float] = None
    source: Optional[str] = None
    chunk_idx: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class DocumentAnalysisResponse(BaseModel):
    grade: str                 # "green" / "yellow" / "red" / "unknown"
    grade_label: str           # "Зелёный" / "Жёлтый" / "Красный" / "Не определён"
    explanation: str           # почему такой вывод
    recommendations: Optional[str] = None
    sources: List[SourceItem]
