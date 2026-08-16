from app.llm import AnswerGenerator
from app.vector_store import LocalVectorStore, SearchResult


class RAGService:
    def __init__(
        self,
        store: LocalVectorStore,
        generator: AnswerGenerator,
        min_relevance_score: float = 0.05,
    ):
        self.store = store
        self.generator = generator
        self.min_relevance_score = min_relevance_score

    async def ask(self, question: str, top_k: int) -> tuple[str, list[SearchResult]]:
        results = [
            result
            for result in self.store.search(question, top_k=top_k)
            if result.score >= self.min_relevance_score
        ]
        answer = await self.generator.generate(
            question, [str(result.document["text"]) for result in results]
        )
        return answer, results

