from app.config import get_settings
from app.data_processing import ingest_dataset
from app.vector_store import LocalVectorStore


def main() -> None:
    settings = get_settings()
    documents = ingest_dataset(settings.dataset_path, settings.processed_path)
    store = LocalVectorStore(settings.artifact_dir / "employee_index.joblib")
    store.build(documents)
    print(f"Processed and indexed {len(documents):,} employee records.")


if __name__ == "__main__":
    main()

