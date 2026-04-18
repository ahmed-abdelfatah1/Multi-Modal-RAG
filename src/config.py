"""Configuration settings using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini API
    gemini_api_key: str = ""
    gemini_cache_dir: Path = Path("./data/gemini_cache")

    # Qdrant
    qdrant_url: str = "http://localhost"
    qdrant_port: int = 6333

    # Model cache
    hf_home: Path = Path("./data/hf_cache")

    # Device for retrieval models
    device: Literal["cuda", "cpu", "mps"] = "cuda"

    # Data directories
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./data/indices")

    # Indexing batch sizes
    vision_batch_size: int = 1
    text_batch_size: int = 64

    # Optional override for Poppler bin dir (pdf2image). None = rely on PATH.
    # Needed on Windows when winget modified PATH but current process was
    # started before; set POPPLER_BIN env var in .env.
    poppler_bin: Path | None = None

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.1

    @property
    def qdrant_endpoint(self) -> str:
        """Full Qdrant endpoint URL."""
        return f"{self.qdrant_url}:{self.qdrant_port}"

    @property
    def resolved_device(self) -> str:
        """Return the device to actually use, falling back to CPU if CUDA requested
        but not available (e.g., CPU-only torch wheel installed)."""
        if self.device == "cuda":
            try:
                import torch  # noqa: PLC0415
                if not torch.cuda.is_available():
                    return "cpu"
            except ImportError:
                return "cpu"
        return self.device

    @property
    def corpus_primary_dir(self) -> Path:
        """Primary corpus directory."""
        return self.data_dir / "corpus_primary"

    @property
    def corpus_secondary_dir(self) -> Path:
        """Secondary corpus directory."""
        return self.data_dir / "corpus_secondary"

    @property
    def eval_dir(self) -> Path:
        """Evaluation output directory."""
        return self.data_dir / "eval"


# Global settings instance
settings = Settings()


if __name__ == "__main__":
    # Smoke test: print current settings
    print("Current settings:")
    print(f"  Qdrant endpoint: {settings.qdrant_endpoint}")
    print(f"  Device: {settings.device}")
    print(f"  Data dir: {settings.data_dir}")
    print(f"  Vision batch size: {settings.vision_batch_size}")
    print(f"  Gemini API key set: {bool(settings.gemini_api_key)}")
