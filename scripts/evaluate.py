"""Small retrieval evaluation using labels already present in the spreadsheet."""

from collections import defaultdict

from app.config import get_settings
from app.data_processing import ingest_dataset
from app.vector_store import LocalVectorStore


def main() -> None:
    settings = get_settings()
    documents = ingest_dataset(settings.dataset_path, settings.processed_path)
    store = LocalVectorStore(settings.artifact_dir / "employee_index.joblib")
    store.build(documents)

    cases: list[tuple[str, str, str]] = []
    seen: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        metadata = document["metadata"]
        for field, phrase in (("department", "department"), ("job_title", "job title")):
            value = str(metadata[field])
            if value not in seen[field] and len(seen[field]) < 8:
                cases.append((f"employees with {phrase} {value}", field, value))
                seen[field].add(value)

    precisions = []
    reciprocal_ranks = []
    for query, field, expected in cases:
        results = store.search(query, top_k=5)
        matches = [result.document["metadata"][field] == expected for result in results]
        precisions.append(sum(matches) / len(matches))
        first_match = next((index + 1 for index, match in enumerate(matches) if match), None)
        reciprocal_ranks.append(1 / first_match if first_match else 0)

    print(f"Evaluation cases: {len(cases)}")
    print(f"Mean Precision@5: {sum(precisions) / len(precisions):.3f}")
    print(f"Mean Reciprocal Rank: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    main()

