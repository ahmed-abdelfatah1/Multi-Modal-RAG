"""Text retriever using sentence-transformers (bge-small-en) and Qdrant."""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.retrieval.types import Retrieved, RetrievalResult

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "text"


class TextRetriever:
    """Text retriever using BGE embeddings."""

    def __init__(self) -> None:
        """Initialize the text retriever."""
        self._model: SentenceTransformer | None = None
        self._client: QdrantClient | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(MODEL_NAME, device=settings.resolved_device)
        return self._model

    @property
    def client(self) -> QdrantClient:
        """Lazy-load the Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=settings.qdrant_url.replace("http://", ""),
                port=settings.qdrant_port,
            )
        return self._client

    def retrieve(self, query: str, top_k: int = 8) -> RetrievalResult:
        """Retrieve relevant text chunks."""
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        try:
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding.tolist(),
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            print(f"text_retriever: search failed: {e}")
            return RetrievalResult(query=query, items=[], mode="text_only")

        items = []
        for result in results.points:
            payload = result.payload or {}
            items.append(
                Retrieved(
                    id=str(result.id),
                    source_type="text",
                    doc_id=payload.get("doc_id", "unknown"),
                    page_number=payload.get("page_number", 0),
                    score=result.score,
                    payload=payload,
                )
            )

        return RetrievalResult(query=query, items=items, mode="text_only")


_retriever: TextRetriever | None = None


def get_text_retriever() -> TextRetriever:
    """Get the singleton text retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = TextRetriever()
    return _retriever


if __name__ == "__main__":
    retriever = get_text_retriever()
    result = retriever.retrieve("government budget", top_k=5)
    print(f"Query: {result.query}")
    print(f"Results: {len(result.items)}")
    for item in result.items:
        print(f"  - {item.doc_id} p.{item.page_number}: {item.score:.3f}")
