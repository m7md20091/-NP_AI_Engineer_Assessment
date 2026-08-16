from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["Who works in IT?"])
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 3:
            raise ValueError("Question must contain at least 3 non-whitespace characters")
        return normalized


class SourceRecord(BaseModel):
    employee_id: str
    full_name: str
    score: float
    text: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRecord]
    provider: str


class HealthResponse(BaseModel):
    status: str
    indexed_documents: int
    llm_provider: str

