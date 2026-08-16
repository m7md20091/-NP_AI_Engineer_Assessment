from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    document: dict[str, object]
    score: float


class LocalVectorStore:
    """Persisted sparse-vector store using word and character TF-IDF embeddings."""

    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.documents: list[dict[str, object]] = []

    def build(self, documents: list[dict[str, object]]) -> None:
        if not documents:
            raise ValueError("Cannot build an index with no documents")
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(
            [str(document["text"]) for document in documents]
        )
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "matrix": self.matrix, "documents": documents},
            self.artifact_path,
        )

    def load(self) -> None:
        payload = joblib.load(self.artifact_path)
        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.documents = payload["documents"]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Vector store is not initialized")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [
            SearchResult(document=self.documents[index], score=float(scores[index]))
            for index in indices
        ]

