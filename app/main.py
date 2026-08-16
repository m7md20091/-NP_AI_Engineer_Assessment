from contextlib import asynccontextmanager
import logging

import httpx
from fastapi import FastAPI, HTTPException, Request

from app.config import get_settings
from app.data_processing import ingest_dataset
from app.llm import create_generator
from app.rag import RAGService
from app.schemas import HealthResponse, QuestionRequest, QuestionResponse, SourceRecord
from app.vector_store import LocalVectorStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


def build_service() -> RAGService:
    documents = ingest_dataset(settings.dataset_path, settings.processed_path)
    store = LocalVectorStore(settings.artifact_dir / "employee_index.joblib")
    store.build(documents)
    return RAGService(store, create_generator(settings), settings.min_relevance_score)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building employee search index")
    app.state.rag_service = build_service()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Grounded question answering over the NP employee dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    service: RAGService = request.app.state.rag_service
    return HealthResponse(
        status="ok",
        indexed_documents=len(service.store.documents),
        llm_provider=service.generator.name,
    )


@app.post("/ask", response_model=QuestionResponse, tags=["questions"])
async def ask(payload: QuestionRequest, request: Request) -> QuestionResponse:
    service: RAGService = request.app.state.rag_service
    try:
        answer, results = await service.ask(payload.question, payload.top_k or settings.top_k)
    except httpx.HTTPError as exc:
        logger.exception("Configured model provider failed")
        raise HTTPException(status_code=502, detail="The model provider is unavailable") from exc
    except Exception as exc:
        logger.exception("Question processing failed")
        raise HTTPException(status_code=500, detail="Unable to process the question") from exc

    sources = [
        SourceRecord(
            employee_id=str(result.document["metadata"]["employee_id"]),
            full_name=str(result.document["metadata"]["full_name"]),
            score=round(result.score, 4),
            text=str(result.document["text"]),
        )
        for result in results
    ]
    return QuestionResponse(
        question=payload.question,
        answer=answer,
        sources=sources,
        provider=service.generator.name,
    )

