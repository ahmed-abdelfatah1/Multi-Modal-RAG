"""LangGraph QA pipeline with retrieve, generate, validate nodes."""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.generation.generator import Generator, get_generator
from src.generation.schema import Answer
from src.retrieval.hybrid import HybridRetriever, RetrieverMode, get_hybrid_retriever
from src.retrieval.types import RetrievalResult


class QAState(TypedDict):
    """State for the QA graph."""

    query: str
    mode: str
    retrieved: RetrievalResult | None
    answer: Answer | None
    regen_count: int
    needs_regen: bool


def retrieve_node(state: QAState) -> dict:
    """Retrieve relevant documents."""
    mode = RetrieverMode(state.get("mode", "hybrid"))
    retriever = get_hybrid_retriever(mode)

    retrieved = retriever.retrieve(state["query"], top_k=8)

    return {"retrieved": retrieved}


def generate_node(state: QAState) -> dict:
    """Generate answer from retrieved context."""
    generator = get_generator()

    regen = state.get("regen_count", 0) > 0
    answer = generator.generate(
        state["query"],
        state["retrieved"],
        regen=regen,
    )

    return {"answer": answer}


def validate_node(state: QAState) -> dict:
    """Validate that citations match retrieved sources."""
    answer = state["answer"]
    retrieved = state["retrieved"]

    if not answer or not retrieved:
        return {"needs_regen": False}

    # Get valid (doc_id, page_number) pairs from retrieved
    valid_pages = set()
    for item in retrieved.items:
        valid_pages.add((item.doc_id, item.page_number))

    # Check all citations
    needs_regen = False
    for citation in answer.citations:
        key = (citation.doc_id, citation.page_number)
        if key not in valid_pages:
            needs_regen = True
            break

    return {"needs_regen": needs_regen}


def should_regenerate(state: QAState) -> str:
    """Decide whether to regenerate or end."""
    if state.get("needs_regen") and state.get("regen_count", 0) < 1:
        return "regenerate"
    return "end"


def increment_regen_count(state: QAState) -> dict:
    """Increment regeneration counter."""
    return {"regen_count": state.get("regen_count", 0) + 1}


def build_qa_graph() -> StateGraph:
    """Build the QA pipeline graph."""
    workflow = StateGraph(QAState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("increment_regen", increment_regen_count)

    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")

    # Conditional edge from validate
    workflow.add_conditional_edges(
        "validate",
        should_regenerate,
        {
            "regenerate": "increment_regen",
            "end": END,
        },
    )

    workflow.add_edge("increment_regen", "generate")

    return workflow


# Compiled graph (lazy)
_graph = None


def get_qa_graph():
    """Get the compiled QA graph."""
    global _graph
    if _graph is None:
        workflow = build_qa_graph()
        _graph = workflow.compile()
    return _graph


def ask(query: str, mode: str = "hybrid") -> Answer:
    """Ask a question and get an answer.

    Args:
        query: The question to ask.
        mode: Retrieval mode ("hybrid", "text_only", or "vision_only").

    Returns:
        Generated Answer.
    """
    graph = get_qa_graph()

    initial_state: QAState = {
        "query": query,
        "mode": mode,
        "retrieved": None,
        "answer": None,
        "regen_count": 0,
        "needs_regen": False,
    }

    result = graph.invoke(initial_state)

    return result["answer"]


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is the main topic?"

    print(f"Query: {query}")
    print("Running QA pipeline...")

    answer = ask(query)

    print("\n" + "=" * 60)
    print(answer.model_dump_json(indent=2))
