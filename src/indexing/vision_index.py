"""Build vision index using ColSmol-256M multi-vector embeddings and Qdrant."""

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from src.config import settings

# Collection name for vision embeddings
COLLECTION_NAME = "pages"

# ColSmol-256M — same ColVision family, 80.1 ViDoRe nDCG@5, Apache 2.0.
# Chosen over colqwen2-v1.0 because we only have 4 GB dedicated VRAM.
MODEL_NAME = "vidore/colSmol-256M"

# Multi-vector patch embedding dimension (shared across ColVision family)
VECTOR_SIZE = 128


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client."""
    return QdrantClient(
        host=settings.qdrant_url.replace("http://", ""),
        port=settings.qdrant_port,
    )


_PRIMARY_DOC_ID_RE = re.compile(r"([0-9a-f-]{36}\.pdf)_page_\d+", re.IGNORECASE)


def _extract_primary_doc_id(filename: str) -> str | None:
    """Pull the UUID.pdf out of a flattened primary-corpus page filename.

    Names look like ``..._<uuid>.pdf_page_N.png``; the UUID+".pdf" is the
    doc_id. Returns None if the pattern does not match.
    """
    m = _PRIMARY_DOC_ID_RE.search(filename)
    return m.group(1) if m else None


def generate_point_id(image_filename: str, doc_id: str = "") -> str:
    """Generate unique point ID as a UUID (required by Qdrant).

    Includes doc_id because secondary corpus docs share page_NNN.png naming
    across different PDFs and would otherwise collide at the UUID layer.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:{image_filename}"))


def load_page_images() -> list[dict]:
    """Load all page images from both corpora."""
    images = []

    # Primary corpus
    primary_pages = settings.corpus_primary_dir / "pages"
    if primary_pages.exists():
        qa_path = settings.corpus_primary_dir / "qa.jsonl"
        page_to_doc: dict[str, str] = {}
        if qa_path.exists():
            with open(qa_path, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    page_to_doc[record["image_filename"]] = record["source"]

        for img_path in primary_pages.glob("*.png"):
            # qa.jsonl only covers 72 of the 972 primary pages; for the rest,
            # parse the UUID out of the flattened filename instead of stamping
            # "unknown" on the payload.
            doc_id = page_to_doc.get(img_path.name) or _extract_primary_doc_id(img_path.name) or "unknown"
            try:
                page_num = int(img_path.stem.split("_")[-1])
            except (ValueError, IndexError):
                page_num = 1

            images.append({
                "path": img_path,
                "image_filename": img_path.name,
                "doc_id": doc_id,
                "page_number": page_num,
                "source_corpus": "primary",
            })

    # Secondary corpus
    secondary_parsed = settings.corpus_secondary_dir / "parsed"
    if secondary_parsed.exists():
        for doc_dir in secondary_parsed.iterdir():
            if not doc_dir.is_dir():
                continue
            pages_dir = doc_dir / "pages"
            if not pages_dir.exists():
                continue

            for img_path in sorted(pages_dir.glob("*.png")):
                try:
                    page_num = int(img_path.stem.split("_")[-1])
                except (ValueError, IndexError):
                    page_num = 1

                images.append({
                    "path": img_path,
                    "image_filename": img_path.name,
                    "doc_id": doc_dir.name,
                    "page_number": page_num,
                    "source_corpus": "secondary",
                })

    return images


def load_vision_model() -> tuple[ColIdefics3, ColIdefics3Processor]:
    """Load ColSmol-256M model and processor."""
    use_cuda = settings.resolved_device == "cuda"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = settings.resolved_device

    print(f"Loading {MODEL_NAME} (dtype={dtype}, device={device_map})...")

    model = ColIdefics3.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    processor = ColIdefics3Processor.from_pretrained(MODEL_NAME)
    return model, processor


def build_vision_index(recreate: bool = False, batch_size: int | None = None) -> int:
    """Build the vision index in Qdrant using ColSmol-256M multi-vector embeddings."""
    if batch_size is None:
        batch_size = settings.vision_batch_size

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

    print(f"Creating collection '{COLLECTION_NAME}' with multi-vector support...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM,
            ),
            on_disk=True,
        ),
        quantization_config=BinaryQuantization(
            binary=BinaryQuantizationConfig(always_ram=True),
        ),
    )

    client.create_payload_index(COLLECTION_NAME, "doc_id", "keyword")
    client.create_payload_index(COLLECTION_NAME, "source_corpus", "keyword")
    client.create_payload_index(COLLECTION_NAME, "page_number", "integer")

    print("Loading page images...")
    page_images = load_page_images()
    print(f"  Found {len(page_images)} page images")

    if not page_images:
        print("No images to index!")
        return 0

    model, processor = load_vision_model()

    print(f"Embedding and indexing {len(page_images)} images (batch_size={batch_size})...")
    start_time = time.time()
    total_indexed = 0

    for i in tqdm(range(0, len(page_images), batch_size)):
        batch = page_images[i : i + batch_size]

        pil_images = []
        valid_meta = []
        for img_data in batch:
            try:
                img = Image.open(img_data["path"]).convert("RGB")
                pil_images.append(img)
                valid_meta.append(img_data)
            except Exception as e:
                print(f"  Warning: Could not load {img_data['path']}: {e}", file=sys.stderr)

        if not pil_images:
            continue

        processed = processor.process_images(pil_images).to(model.device)
        with torch.no_grad():
            embeddings = model(**processed)

        # embeddings shape: (B, n_patches, 128) — one multi-vector per page
        points = []
        for img_data, embedding in zip(valid_meta, embeddings):
            point_id = generate_point_id(img_data["image_filename"], img_data["doc_id"])
            vec = embedding.float().cpu().tolist()
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "doc_id": img_data["doc_id"],
                        "page_number": img_data["page_number"],
                        "image_filename": img_data["image_filename"],
                        "source_corpus": img_data["source_corpus"],
                    },
                )
            )

        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_indexed += len(points)

    elapsed = time.time() - start_time
    pages_per_sec = total_indexed / elapsed if elapsed > 0 else 0
    print(f"\nIndexed {total_indexed} pages in {elapsed:.1f}s ({pages_per_sec:.2f} pages/sec)")

    return total_indexed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vision index with ColSmol-256M")
    parser.add_argument("--recreate", action="store_true", help="Recreate index")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default 2; drop to 1 if OOM on 4GB VRAM)",
    )
    args = parser.parse_args()

    n = build_vision_index(recreate=args.recreate, batch_size=args.batch_size)
    print(f"Done! Indexed {n} points")
