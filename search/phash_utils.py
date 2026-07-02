"""
Perceptual hashing utilities for video frame deduplication.

pHash computes a fingerprint of an image based on its DCT (frequency domain)
representation. Two visually similar frames will have similar hashes, measured
by Hamming distance — the number of bits that differ.

  Hamming 0    = identical images
  Hamming ≤ 8  = near-duplicate (same scene, minor camera movement)
  Hamming > 20 = clearly different scene
"""
import imagehash
from PIL import Image

# Default deduplication threshold — frames within this Hamming distance
# are considered near-duplicates and only one is kept per scene.
PHASH_THRESHOLD = 8


def compute_phash(image_path: str) -> str:
    """Compute a perceptual hash string for the given image file."""
    img = Image.open(image_path).convert("RGB")
    return str(imagehash.phash(img))


def hamming_distance(h1: str, h2: str) -> int:
    """Return the Hamming distance between two pHash strings."""
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)


def is_unique_frame(new_hash: str, seen_hashes: list, threshold: int = PHASH_THRESHOLD) -> bool:
    """
    Return True if new_hash is visually distinct from all hashes in seen_hashes.
    Uses Hamming distance — if any existing hash is within the threshold,
    the frame is considered a near-duplicate.
    """
    for h in seen_hashes:
        if hamming_distance(new_hash, h) <= threshold:
            return False
    return True


def deduplicate_scenes(scenes: list, hashes: list, threshold: int = PHASH_THRESHOLD) -> list:
    """
    Given a list of scenes and their corresponding pHashes, return only the
    scenes whose frames are visually distinct from all earlier scenes.

    scenes  : list of dicts with at least {"id", "start", "end"}
    hashes  : list of pHash strings, same order as scenes
    Returns : filtered list of (scene, phash) tuples
    """
    kept = []
    seen = []
    for scene, phash in zip(scenes, hashes):
        if is_unique_frame(phash, seen, threshold):
            kept.append((scene, phash))
            seen.append(phash)
    return kept
