"""
Utility functions for curator operations.

Extracted from semantic_curator.py to reduce file size and improve modularity.
"""

import hashlib
import json
from typing import List

import numpy as np

from ace.models.playbook import PlaybookBullet


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Args:
        embedding1: First embedding (384-dim)
        embedding2: Second embedding (384-dim)

    Returns:
        Cosine similarity score (0.0 to 1.0, higher = more similar)
    """
    vec1 = np.array(embedding1, dtype=np.float32)
    vec2 = np.array(embedding2, dtype=np.float32)

    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Cosine similarity = dot product of normalized vectors
    similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return similarity


def compute_bullet_hash(bullet: PlaybookBullet) -> str:
    """
    Compute deterministic SHA-256 hash of bullet state.

    Used for diff journal to track exactly what changed during updates.
    Excludes timestamps to focus on semantic state.

    Args:
        bullet: PlaybookBullet to hash

    Returns:
        SHA-256 hash (hex string)
    """
    stable_state = {
        "id": bullet.id,
        "domain_id": bullet.domain_id,
        "content": bullet.content,
        "section": bullet.section,
        "helpful_count": bullet.helpful_count,
        "harmful_count": bullet.harmful_count,
        "tags": sorted(bullet.tags),
    }
    canonical_json = json.dumps(stable_state, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
