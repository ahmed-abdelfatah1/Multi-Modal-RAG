"""Vision retriever using ColSmol-256M and Qdrant."""

import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from qdrant_client import QdrantClient

from src.config import settings
from src.retrieval.types import Retrieved, RetrievalResult

MODEL_NAME = "vidore/colSmol-256M"
COLLECTION_NAME = "pages"


class VisionRetriever:
    """Vision retriever using ColSmol-256M multi-vector embeddings."""

    def __init__(self) -> None:
        """Initialize the vision retriever."""
        self._model: ColIdefics3 | None = None
        self._processor: ColIdefics3Processor | None = None
        self._client: QdrantClient | None = None

    def _load_model(self) -> None:
        use_cuda = settings.resolved_device == "cuda"
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = settings.resolved_device

        model = ColIdefics3.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map=device_map,
        )
        # Inference mode (equivalent to model.eval())
        model.train(False)
        self._model = model
        self._processor = ColIdefics3Processor.from_pretrained(MODEL_NAME)

    @property
    def model(self) -> ColIdefics3:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore[return-value]

    @property
    def processor(self) -> ColIdefics3Processor:
        """Lazy-load the processor."""
        if self._processor is None:
            self._load_model()
        return self._processor  # type: ignore[return-value]

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
        """Retrieve relevant page images."""
        processed = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            query_embeddings = self.model(**processed)

        # Shape: (1, n_tokens, 128) — take the first (only) query
        multi_vec = query_embeddings[0].float().cpu().tolist()

        try:
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=multi_vec,
                limit=top_k,
                with_payload=True,
            )
        except Exception:
            return RetrievalResult(query=query, items=[], mode="vision_only")

        items = []
        for result in results.points:
            payload = result.payload or {}
            items.append(
                Retrieved(
                    id=str(result.id),
                    source_type="page",
                    doc_id=payload.get("doc_id", "unknown"),
                    page_number=payload.get("page_number", 0),
                    score=result.score,
                    payload=payload,
                )
            )

        return RetrievalResult(query=query, items=items, mode="vision_only")


_retriever: VisionRetriever | None = None


def get_vision_retriever() -> VisionRetriever:
    """Get the singleton vision retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = VisionRetriever()
    return _retriever


if __name__ == "__main__":
    retriever = get_vision_retriever()
    result = retriever.retrieve("budget chart", top_k=5)
    print(f"Query: {result.query}")
    print(f"Results: {len(result.items)}")
    for item in result.items:
        print(f"  - {item.doc_id} p.{item.page_number}: {item.score:.3f}")
        if "image_filename" in item.payload:
            print(f"    Image: {item.payload['image_filename']}")
