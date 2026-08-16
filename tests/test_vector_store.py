from app.vector_store import LocalVectorStore


def test_search_returns_relevant_employee(tmp_path):
    documents = [
        {"id": "1", "text": "Alice works in Finance as an Accountant", "metadata": {}},
        {"id": "2", "text": "Bob works in IT as a Software Engineer", "metadata": {}},
    ]
    store = LocalVectorStore(tmp_path / "index.joblib")
    store.build(documents)
    results = store.search("software engineer in IT", top_k=1)
    assert results[0].document["id"] == "2"
    assert results[0].score > 0

