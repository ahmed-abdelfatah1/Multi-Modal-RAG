"""Disk-based cache for Gemini API responses."""

import hashlib
import json
from pathlib import Path

from src.config import settings


class GeminiCache:
    """Simple disk-based cache for Gemini responses."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to settings.
        """
        self.cache_dir = cache_dir or settings.gemini_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, prompt: str, image_hashes: list[str], config_str: str) -> str:
        """Generate cache key from inputs."""
        data = f"{prompt}|{'|'.join(image_hashes)}|{config_str}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _get_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.json"

    def get(self, prompt: str, image_hashes: list[str], config_str: str) -> dict | None:
        """Get cached response if available.

        Args:
            prompt: The prompt text.
            image_hashes: Hashes of images included in the request.
            config_str: String representation of config.

        Returns:
            Cached response dict or None if not found.
        """
        key = self._make_key(prompt, image_hashes, config_str)
        path = self._get_path(key)

        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None

        return None

    def set(
        self,
        prompt: str,
        image_hashes: list[str],
        config_str: str,
        response: dict,
    ) -> None:
        """Cache a response.

        Args:
            prompt: The prompt text.
            image_hashes: Hashes of images included in the request.
            config_str: String representation of config.
            response: Response dict to cache.
        """
        key = self._make_key(prompt, image_hashes, config_str)
        path = self._get_path(key)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)

    def clear(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of entries cleared.
        """
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count


# Singleton instance
_cache: GeminiCache | None = None


def get_cache() -> GeminiCache:
    """Get the singleton cache instance."""
    global _cache
    if _cache is None:
        _cache = GeminiCache()
    return _cache


def hash_image_bytes(image_bytes: bytes) -> str:
    """Hash image bytes for cache key."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]
