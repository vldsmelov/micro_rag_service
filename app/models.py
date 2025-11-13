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


class SectionScore(BaseModel):
    code: str                     # напр. "parties"
    title: str                    # человекочитаемое имя
    score: float                  # набранный балл
    max_score: float              # максимум по разделу
    grade: str                    # "green" / "yellow" / "red" / "unknown"
    comment: Optional[str] = None # комментарий по разделу


class KeyRisk(BaseModel):
    section_code: str             # код раздела
    title: str                    # название раздела
    issue: str                    # в чём проблема
    recommendation: Optional[str] = None  # что сделать


class DocumentAnalysisResponse(BaseModel):
    # Общая оценка документа
    grade: str                    # "green" / "yellow" / "red" / "unknown"
    grade_label: str              # "Зелёный" / "Жёлтый" / "Красный" / "Не определён"
    score: float                  # общий балл (0–100)
    score_max: float              # максимум (обычно 100)
    summary: str                  # краткое резюме (2 части: кто/о чём + экспертная оценка)
    recommendations: Optional[str] = None  # общий блок "Замечания и рекомендации"

    # Детализация
    key_risks: List[KeyRisk]      # "куда смотреть в первую очередь"
    sections: List[SectionScore]  # таблица секций и баллов

    # На что опирались
    sources: List[SourceItem]

    # Сгенерированный HTML-отчёт
    report_path: Optional[str] = None
