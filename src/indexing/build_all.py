"""Build all indices (text and vision)."""

import argparse
import sys

from src.indexing.text_index import build_text_index
from src.indexing.vision_index import build_vision_index


def main() -> None:
    """Build all indices."""
    parser = argparse.ArgumentParser(description="Build all indices")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate indices from scratch",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only build text index",
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Only build vision index",
    )
    args = parser.parse_args()

    if args.text_only:
        print("Building text index only...")
        build_text_index(recreate=args.recreate)
    elif args.vision_only:
        print("Building vision index only...")
        build_vision_index(recreate=args.recreate)
    else:
        print("Building text index...")
        build_text_index(recreate=args.recreate)
        print("\nBuilding vision index...")
        build_vision_index(recreate=args.recreate)

    print("\nAll indices built successfully!")


if __name__ == "__main__":
    main()
