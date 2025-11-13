from pathlib import Path
from datetime import datetime
import re

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.models import DocumentAnalysisResponse


BASE_DIR = Path(__file__).resolve().parents[2]   # /app
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def render_html_report(
    analysis: DocumentAnalysisResponse,
    original_filename: str | None = None,
    document_text: str | None = None,
) -> str:
    """
    Рендерит HTML-отчёт по анализу документа и сохраняет его в папку /reports.
    Возвращает относительный путь к файлу (например, 'reports/report_20251113_120301_doc.html').
    """
    template = env.get_template("report.html")

    created_at = datetime.now()
    created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S")

    # Небольшой превью текста для блока "Показать текст"
    doc_preview = None
    if document_text:
        doc_preview = document_text.strip()
        if len(doc_preview) > 4000:
            doc_preview = doc_preview[:4000] + "\n\n[Текст обрезан для отчёта]"

    grade_colors = {
        "green": "#16a34a",
        "yellow": "#facc15",
        "red": "#ef4444",
        "unknown": "#6b7280",
    }
    overall_color = grade_colors.get(analysis.grade, "#6b7280")

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", original_filename or "document")
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}_{safe_name}.html"
    filepath = REPORTS_DIR / filename

    html = template.render(
        analysis=analysis,
        created_at=created_at_str,
        original_filename=original_filename,
        doc_preview=doc_preview,
        overall_color=overall_color,
    )

    filepath.write_text(html, encoding="utf-8")

    # Вернём путь относительно корня проекта (/app)
    return str(filepath.relative_to(BASE_DIR))
