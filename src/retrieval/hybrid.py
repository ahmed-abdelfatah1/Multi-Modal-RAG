"""Hybrid retriever with Reciprocal Rank Fusion."""

from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from src.retrieval.text_retriever import get_text_retriever
from src.retrieval.types import Retrieved, RetrievalResult
from src.retrieval.vision_retriever import get_vision_retriever


class RetrieverMode(str, Enum):
    """Retriever mode options."""

    TEXT_ONLY = "text_only"
    VISION_ONLY = "vision_only"
    HYBRID = "hybrid"


# RRF constant (standard value)
RRF_K = 60


def reciprocal_rank_fusion(
    results_list: list[list[Retrieved]],
    k: int = RRF_K,
) -> list[Retrieved]:
    """Apply Reciprocal Rank Fusion to multiple result lists.

    Args:
        results_list: List of result lists from different retrievers.
        k: RRF constant (default 60).

    Returns:
        Fused and re-ranked list of results.
    """
    # Calculate RRF scores
    scores: dict[str, float] = {}
    items: dict[str, Retrieved] = {}

    for results in results_list:
        for rank, item in enumerate(results, start=1):
            # Use (doc_id, page_number) as key to deduplicate
            key = f"{item.doc_id}:{item.page_number}"

            if key not in scores:
                scores[key] = 0
                items[key] = item

            # RRF formula: 1 / (k + rank)
            scores[key] += 1 / (k + rank)

            # Prefer page type over text type (more info)
            if item.source_type == "page" and items[key].source_type == "text":
                items[key] = item

    # Sort by fused score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build fused results with updated scores
    fused = []
    for key in sorted_keys:
        item = items[key]
        fused.append(
            Retrieved(
                id=item.id,
                source_type=item.source_type,
                doc_id=item.doc_id,
                page_number=item.page_number,
                score=scores[key],
                payload=item.payload,
            )
        )

    return fused


class HybridRetriever:
    """Hybrid retriever combining text and vision channels."""

    def __init__(self, mode: RetrieverMode = RetrieverMode.HYBRID) -> None:
        """Initialize the hybrid retriever.

        Args:
            mode: Retrieval mode to use.
        """
        self.mode = mode
        self._text_retriever = None
        self._vision_retriever = None

    @property
    def text_retriever(self):
        """Lazy-load text retriever."""
        if self._text_retriever is None:
            self._text_retriever = get_text_retriever()
        return self._text_retriever

    @property
    def vision_retriever(self):
        """Lazy-load vision retriever."""
        if self._vision_retriever is None:
            self._vision_retriever = get_vision_retriever()
        return self._vision_retriever

    def retrieve(self, query: str, top_k: int = 8) -> RetrievalResult:
        """Retrieve relevant items using the configured mode.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            RetrievalResult with retrieved items.
        """
        if self.mode == RetrieverMode.TEXT_ONLY:
            return self.text_retriever.retrieve(query, top_k)

        if self.mode == RetrieverMode.VISION_ONLY:
            return self.vision_retriever.retrieve(query, top_k)

        # Hybrid mode: run both in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(self.text_retriever.retrieve, query, top_k)
            vision_future = executor.submit(self.vision_retriever.retrieve, query, top_k)

            text_result = text_future.result()
            vision_result = vision_future.result()

        # Apply RRF fusion
        fused_items = reciprocal_rank_fusion(
            [text_result.items, vision_result.items],
        )

        return RetrievalResult(
            query=query,
            items=fused_items[:top_k],
            mode="hybrid",
        )


# Singleton instances by mode
_retrievers: dict[RetrieverMode, HybridRetriever] = {}


def get_hybrid_retriever(mode: RetrieverMode = RetrieverMode.HYBRID) -> HybridRetriever:
    """Get a hybrid retriever instance for the given mode."""
    if mode not in _retrievers:
        _retrievers[mode] = HybridRetriever(mode)
    return _retrievers[mode]


if __name__ == "__main__":
    # Smoke test
    for mode in RetrieverMode:
        print(f"\n=== Mode: {mode.value} ===")
        retriever = get_hybrid_retriever(mode)
        result = retriever.retrieve("government budget spending", top_k=5)
        print(f"Query: {result.query}")
        print(f"Results: {len(result.items)}")
        for item in result.items:
            print(f"  - [{item.source_type}] {item.doc_id} p.{item.page_number}: {item.score:.3f}")
