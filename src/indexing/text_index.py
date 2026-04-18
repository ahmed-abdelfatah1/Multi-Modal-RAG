"""Build text index using sentence-transformers (bge-small-en) and Qdrant."""

import argparse
import json
import sys
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import settings

# Collection name for text embeddings
COLLECTION_NAME = "text"

# BGE model via sentence-transformers
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client."""
    return QdrantClient(
        host=settings.qdrant_url.replace("http://", ""),
        port=settings.qdrant_port,
    )


def generate_point_id(content: str, doc_id: str, page_number: int) -> str:
    """Generate unique point ID as a UUID (required by Qdrant)."""
    data = f"{content}:{doc_id}:{page_number}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, data))


def load_primary_corpus_texts() -> list[dict]:
    """Load text content from primary corpus as synthetic QA anchors.

    ViDoRe dataset doesn't include OCR text, so we use the QA pairs as
    synthetic text anchors for the text retrieval channel.
    """
    qa_path = settings.corpus_primary_dir / "qa.jsonl"
    if not qa_path.exists():
        print(f"Warning: {qa_path} not found", file=sys.stderr)
        return []

    # Group by page to avoid duplicates when multiple QA hit one page
    page_texts: dict[str, dict] = {}

    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # page is stored as string in the dataset - normalize to int
            page_number = int(record["page"])
            page_key = f"{record['source']}_{page_number}"

            if page_key not in page_texts:
                page_texts[page_key] = {
                    "doc_id": record["source"],
                    "page_number": page_number,
                    "image_filename": record["image_filename"],
                    "content": f"Q: {record['query']}\nA: {record['answer']}",
                    "chunk_type": "synthetic_qa_anchor",
                    "source_corpus": "primary",
                }
            else:
                page_texts[page_key]["content"] += (
                    f"\n\nQ: {record['query']}\nA: {record['answer']}"
                )

    return list(page_texts.values())


def load_secondary_corpus_texts() -> list[dict]:
    """Load text chunks from secondary corpus."""
    parsed_dir = settings.corpus_secondary_dir / "parsed"
    if not parsed_dir.exists():
        return []

    texts = []
    for doc_dir in parsed_dir.iterdir():
        if not doc_dir.is_dir():
            continue

        chunks_path = doc_dir / "chunks.jsonl"
        if not chunks_path.exists():
            continue

        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                texts.append({
                    "doc_id": chunk["doc_id"],
                    "page_number": chunk["page_number"],
                    "content": chunk["content"],
                    "chunk_type": chunk["chunk_type"],
                    "source_corpus": "secondary",
                })

    return texts


def build_text_index(recreate: bool = False) -> int:
    """Build the text index in Qdrant."""
    client = get_qdrant_client()

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME in collection_names:
        if recreate:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        else:
            info = client.get_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' exists with {info.points_count} points")
            return info.points_count

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
            on_disk=True,
        ),
    )

    # Payload indexes
    client.create_payload_index(COLLECTION_NAME, "doc_id", "keyword")
    client.create_payload_index(COLLECTION_NAME, "source_corpus", "keyword")
    client.create_payload_index(COLLECTION_NAME, "chunk_type", "keyword")
    client.create_payload_index(COLLECTION_NAME, "page_number", "integer")

    print("Loading texts from primary corpus...")
    primary_texts = load_primary_corpus_texts()
    print(f"  Loaded {len(primary_texts)} text entries from primary corpus")

    print("Loading texts from secondary corpus...")
    secondary_texts = load_secondary_corpus_texts()
    print(f"  Loaded {len(secondary_texts)} text entries from secondary corpus")

    all_texts = primary_texts + secondary_texts
    if not all_texts:
        print("No texts to index!")
        return 0

    print(f"Loading embedding model: {MODEL_NAME} (device={settings.resolved_device})...")
    model = SentenceTransformer(MODEL_NAME, device=settings.resolved_device)

    print(f"Embedding and indexing {len(all_texts)} texts...")
    batch_size = settings.text_batch_size
    contents = [t["content"] for t in all_texts]

    embeddings = model.encode(
        contents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    points: list[PointStruct] = []
    for text_data, embedding in zip(all_texts, embeddings):
        point_id = generate_point_id(
            text_data["content"],
            text_data["doc_id"],
            text_data["page_number"],
        )
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "doc_id": text_data["doc_id"],
                    "page_number": text_data["page_number"],
                    "chunk_type": text_data["chunk_type"],
                    "source_corpus": text_data["source_corpus"],
                    "content": text_data["content"][:500],
                },
            )
        )

    print("Uploading to Qdrant...")
    for i in tqdm(range(0, len(points), 100)):
        batch = points[i : i + 100]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"Indexed {len(points)} text chunks")
    return len(points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build text index")
    parser.add_argument("--recreate", action="store_true", help="Recreate index")
    args = parser.parse_args()

    n = build_text_index(recreate=args.recreate)
    print(f"Done! Indexed {n} points")
