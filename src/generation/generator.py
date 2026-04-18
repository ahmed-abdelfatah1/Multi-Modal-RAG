"""Answer generation using Gemini 2.5 Flash."""

import argparse
import hashlib
import io
import sys
import time
from pathlib import Path

from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.generation.cache import get_cache, hash_image_bytes
from src.generation.prompts import REGEN_SUFFIX, SYSTEM_PROMPT
from src.generation.schema import Answer
from src.retrieval.types import RetrievalResult

# Maximum images per call (Gemini payload limit)
MAX_IMAGES = 4


class Generator:
    """Answer generator using Gemini 2.5 Flash."""

    def __init__(self, min_seconds_between_calls: float = 0) -> None:
        """Initialize the generator.

        Args:
            min_seconds_between_calls: Minimum delay between API calls (for rate limiting).
        """
        self._client = None
        self.min_delay = min_seconds_between_calls
        self._last_call_time = 0.0
        self.cache = get_cache()

    @property
    def client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            from google import genai
            api_key = settings.gemini_api_key or None
            self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        return self._client

    def _enforce_rate_limit(self) -> None:
        """Enforce minimum delay between API calls."""
        if self.min_delay > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)

    def _build_contents(
        self,
        question: str,
        retrieved: RetrievalResult,
        regen: bool = False,
    ) -> tuple[list, list[str]]:
        """Build contents list for Gemini API.

        Returns:
            Tuple of (contents list, image hashes for caching).
        """
        from google.genai import types

        contents = []
        image_hashes = []

        # System prompt
        prompt_text = SYSTEM_PROMPT
        if regen:
            prompt_text += REGEN_SUFFIX

        contents.append(prompt_text)
        contents.append(f"\nQuestion: {question}\n\nSources:\n")

        # Add text chunks
        text_items = [item for item in retrieved.items if item.source_type == "text"]
        for item in text_items:
            content = item.payload.get("content", "")
            if content:
                contents.append(f"\n[Source: {item.doc_id} p.{item.page_number}]\n{content}\n")

        # Add page images (capped at MAX_IMAGES)
        page_items = [item for item in retrieved.items if item.source_type == "page"][:MAX_IMAGES]

        if len(page_items) < len([i for i in retrieved.items if i.source_type == "page"]):
            print(
                f"Warning: Capping images at {MAX_IMAGES} (had more)",
                file=sys.stderr,
            )

        for item in page_items:
            image_filename = item.payload.get("image_filename")
            if not image_filename:
                continue

            # Try to load the image
            image_path = self._find_image(image_filename, item.payload.get("source_corpus"))
            if image_path and image_path.exists():
                try:
                    img = Image.open(image_path)

                    # Add caption
                    contents.append(f"\n[Page image: {item.doc_id} p.{item.page_number}]")
                    contents.append(img)

                    # Hash for cache
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    image_hashes.append(hash_image_bytes(img_bytes.getvalue()))

                except Exception as e:
                    print(f"Warning: Could not load image {image_path}: {e}", file=sys.stderr)

        return contents, image_hashes

    def _find_image(self, filename: str, source_corpus: str | None) -> Path | None:
        """Find image path given filename and corpus."""
        if source_corpus == "primary":
            path = settings.corpus_primary_dir / "pages" / filename
            if path.exists():
                return path

        elif source_corpus == "secondary":
            # Search in parsed directories
            parsed_dir = settings.corpus_secondary_dir / "parsed"
            if parsed_dir.exists():
                for doc_dir in parsed_dir.iterdir():
                    path = doc_dir / "pages" / filename
                    if path.exists():
                        return path

        # Fallback: search both
        for corpus_dir in [settings.corpus_primary_dir, settings.corpus_secondary_dir]:
            for path in corpus_dir.rglob(filename):
                return path

        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(self, contents: list, config_str: str) -> Answer:
        """Make the actual API call with retry logic."""
        from google.genai import types

        self._enforce_rate_limit()

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=settings.temperature,
                max_output_tokens=settings.max_new_tokens,
                response_mime_type="application/json",
                response_schema=Answer,
                # 2.5 Flash thinks by default and thinking tokens eat the output
                # budget, which breaks structured-output parsing. Disable it.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        self._last_call_time = time.time()

        if response.parsed is None:
            finish = getattr(response.candidates[0], "finish_reason", "?") if response.candidates else "?"
            safety = getattr(response.candidates[0], "safety_ratings", None) if response.candidates else None
            usage = getattr(response, "usage_metadata", None)
            print(f"  [debug] finish_reason={finish} safety={safety} usage={usage}", file=sys.stderr)
            print(f"  [debug] response.text={response.text!r}", file=sys.stderr)
            raise ValueError(f"Failed to parse response (finish={finish}): {response.text!r}")

        return response.parsed

    def generate(
        self,
        question: str,
        retrieved: RetrievalResult,
        regen: bool = False,
    ) -> Answer:
        """Generate an answer for the question.

        Args:
            question: The question to answer.
            retrieved: Retrieved context items.
            regen: Whether this is a regeneration attempt.

        Returns:
            Generated Answer.
        """
        contents, image_hashes = self._build_contents(question, retrieved, regen)

        # Build config string for cache key
        config_str = f"temp={settings.temperature}|max={settings.max_new_tokens}"

        # Build prompt text for cache (excluding images)
        prompt_parts = [str(c) for c in contents if isinstance(c, str)]
        prompt_text = "".join(prompt_parts)

        # Check cache
        cached = self.cache.get(prompt_text, image_hashes, config_str)
        if cached:
            print("cache hit", file=sys.stderr)
            return Answer(**cached)

        print("cache miss", file=sys.stderr)

        # Call API
        answer = self._call_api(contents, config_str)

        # Cache result
        self.cache.set(prompt_text, image_hashes, config_str, answer.model_dump())

        return answer


# Singleton instance
_generator: Generator | None = None


def get_generator(min_seconds_between_calls: float = 0) -> Generator:
    """Get the singleton generator instance."""
    global _generator
    if _generator is None:
        _generator = Generator(min_seconds_between_calls)
    return _generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument("--query", type=str, required=True, help="Query to answer")
    args = parser.parse_args()

    # Import here to avoid circular imports
    from src.retrieval.hybrid import RetrieverMode, get_hybrid_retriever

    print(f"Query: {args.query}")
    print("Retrieving context...")

    retriever = get_hybrid_retriever(RetrieverMode.HYBRID)
    retrieved = retriever.retrieve(args.query, top_k=6)
    print(f"Retrieved {len(retrieved.items)} items")

    print("Generating answer...")
    generator = get_generator()
    answer = generator.generate(args.query, retrieved)

    print("\n" + "=" * 60)
    print(answer.model_dump_json(indent=2))
