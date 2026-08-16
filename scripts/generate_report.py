from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    KeepTogether, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "NP_Employee_Assistant_Report.pdf"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(20 * mm, 12 * mm, "NP Employee Assistant — AI Engineer Assessment")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def architecture_table():
    data = [
        ["employee_np.xlsx", "→", "Clean + validate", "→", "JSONL documents"],
        ["User question", "→", "FastAPI /ask", "→", "TF-IDF vector search"],
        ["Top-k sources", "→", "Grounded prompt", "→", "Generator + citations"],
    ]
    table = Table(data, colWidths=[37*mm, 8*mm, 39*mm, 8*mm, 55*mm], rowHeights=13*mm)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F4F7")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#98A2B3")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D5DD")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
    ]))
    return table


def build_report():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#175CD3"), spaceAfter=12))
    styles["Heading1"].textColor = colors.HexColor("#175CD3")
    styles["Heading1"].spaceBefore = 10
    styles["Heading1"].spaceAfter = 6
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13

    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                          topMargin=18*mm, bottomMargin=20*mm,
                          title="NP Employee Assistant Design Report")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=footer)])

    story = [
        Spacer(1, 28*mm),
        Paragraph("NP Employee Assistant", styles["TitleCenter"]),
        Paragraph("AI Engineer Assessment — Design Report", ParagraphStyle(
            name="Subtitle", parent=styles["Heading2"], alignment=TA_CENTER,
            textColor=colors.HexColor("#475467"))),
        Spacer(1, 12*mm),
        Paragraph("A local-first retrieval-augmented assistant over 10,018 employee records, "
                  "with FastAPI, source citations, reproducible preprocessing, and optional "
                  "OpenAI-compatible or Ollama generation.", styles["BodyText"]),
        PageBreak(),
        Paragraph("1. Architecture", styles["Heading1"]),
        architecture_table(), Spacer(1, 5*mm),
    ]

    sections = [
        ("Request lifecycle", "At startup, the workbook is validated and converted into one "
         "self-contained text document per employee. TF-IDF creates sparse embeddings stored "
         "in a local Joblib index. POST /ask validates a question, retrieves top-k records by "
         "cosine similarity, applies a minimum relevance threshold, and passes only those "
         "records to the configured answer generator. The response exposes its sources."),
        ("2. Data ingestion and processing", "Required columns are validated; duplicate employee "
         "IDs are removed; whitespace and missing text are normalized; numeric fields are coerced; "
         "and Excel serial or datetime hire dates become ISO dates. Each row remains one chunk "
         "because it is already a coherent factual unit. Processed records are stored as JSONL."),
        ("3. AI/ML implementation", "TF-IDF word and bigram embeddings form a transparent offline "
         "baseline that performs well for names, IDs, roles, and departments. Cosine similarity "
         "ranks relevant records. The default extractive mode works without secrets. OpenAI-compatible "
         "and Ollama adapters enable generative RAG; temperature is zero and prompts prohibit "
         "unsupported claims."),
        ("Evaluation", "The evaluation script derives labeled department and job-title queries and "
         "reports Precision@5 and Mean Reciprocal Rank. Unit and API tests cover document creation, "
         "retrieval relevance, health checks, valid questions, and input validation."),
        ("4. API layer", "FastAPI provides GET /health and POST /ask, automatic OpenAPI/Swagger "
         "documentation, typed request and response schemas, bounded top-k input, whitespace "
         "normalization, validation errors, and safe handling of upstream model failures."),
        ("5. Limitations", "TF-IDF is lexical and may miss paraphrases. Top-k retrieval cannot safely "
         "calculate exact whole-dataset aggregates. Indexing currently rebuilds at startup. LLM "
         "prompt constraints cannot eliminate hallucinations. Authentication, authorization, audit "
         "logging, and field-level salary permissions are outside this prototype."),
        ("Accuracy and hallucination improvements", "Use dense domain-tested embeddings, hybrid "
         "search, metadata filtering, and a reranker. Route count and average questions to validated "
         "structured queries. Require citations, reject low-confidence answers, build a human-labeled "
         "evaluation set, and evaluate faithfulness after every model or prompt change."),
        ("Production deployment", "Deploy the Docker image through CI behind an authenticated API "
         "gateway. Move data to PostgreSQL and vectors to pgvector or a managed vector database. "
         "Schedule ingestion separately, version indexes, store secrets in a secret manager, use TLS "
         "and rate limits, and enforce role-based access to sensitive employee fields."),
        ("Monitoring", "Track latency, errors, request volume, provider cost, empty-result rate, "
         "retrieval scores, source use, and feedback. Monitor freshness, missing values, duplicates, "
         "schema and distribution drift. Run regression evaluations for every data, model, embedding, "
         "or prompt release and alert on deterioration."),
    ]
    for heading, body in sections:
        story.append(KeepTogether([
            Paragraph(heading, styles["Heading1"]),
            Paragraph(body, styles["BodyText"]), Spacer(1, 2*mm),
        ]))
    doc.build(story)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_report()

