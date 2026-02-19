"""Duplicate detection using perceptual hashing."""

from __future__ import annotations

from typing import List

import imagehash
from PIL import Image


HashType = imagehash.ImageHash


def compute_hash(image: Image.Image) -> HashType:
    """Compute a perceptual hash for a PIL image."""
    return imagehash.phash(image)


def is_duplicate(image: Image.Image, existing_hashes: List[HashType], threshold: int = 5) -> bool:
    """Return True if the image hash is close to an existing hash.

    Args:
        image: Input PIL image.
        existing_hashes: Mutable list of previously seen hashes.
        threshold: Max Hamming distance considered duplicate.
    """
    current_hash = compute_hash(image)

    for known_hash in existing_hashes:
        if current_hash - known_hash <= threshold:
            return True

    existing_hashes.append(current_hash)
    return False
