#!/usr/bin/env python3
"""
Playbook Consolidation Script (T054)

Consolidates and deduplicates playbook bullets using semantic similarity.
Removes bullets with ≥0.8 cosine similarity (per .env SIMILARITY_THRESHOLD).
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, select, delete, func
from sqlalchemy.orm import Session
import numpy as np
from typing import List, Tuple
from datetime import datetime

from ace.models.playbook import PlaybookBullet
from ace.utils.database import create_db_engine


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_duplicates(
    session: Session,
    domain_id: str,
    similarity_threshold: float = 0.8,
) -> List[Tuple[str, str, float]]:
    """Find duplicate bullets within a domain using semantic similarity.

    Returns list of (bullet1_id, bullet2_id, similarity) tuples.
    """
    duplicates = []

    # Get all bullets for this domain
    stmt = select(PlaybookBullet).where(PlaybookBullet.domain_id == domain_id)
    bullets = list(session.execute(stmt).scalars())

    # Compare all pairs
    for i in range(len(bullets)):
        for j in range(i + 1, len(bullets)):
            b1, b2 = bullets[i], bullets[j]

            # Convert embeddings to numpy arrays
            emb1 = np.array(b1.embedding)
            emb2 = np.array(b2.embedding)

            similarity = cosine_similarity(emb1, emb2)

            if similarity >= similarity_threshold:
                duplicates.append((b1.id, b2.id, similarity))

    return duplicates


def merge_duplicates(
    session: Session,
    bullet1_id: str,
    bullet2_id: str,
) -> None:
    """Merge two duplicate bullets, keeping the one with higher effectiveness."""
    b1 = session.get(PlaybookBullet, bullet1_id)
    b2 = session.get(PlaybookBullet, bullet2_id)

    if not b1 or not b2:
        return

    # Calculate effectiveness ratios
    eff1 = b1.helpful_count / max(1, b1.helpful_count + b1.harmful_count)
    eff2 = b2.helpful_count / max(1, b2.helpful_count + b2.harmful_count)

    # Keep the more effective bullet, merge the other's counts
    if eff1 >= eff2:
        keeper, removed = b1, b2
    else:
        keeper, removed = b2, b1

    # Merge counts
    keeper.helpful_count += removed.helpful_count
    keeper.harmful_count += removed.harmful_count

    # Delete the less effective bullet
    session.delete(removed)

    print(f"  Merged {removed.id[:8]} -> {keeper.id[:8]} (similarity-based deduplication)")


def remove_stale_bullets(
    session: Session,
    domain_id: str = None,
) -> int:
    """Remove stale bullets with no effectiveness data (helpful=0, harmful=0).

    Args:
        session: Database session
        domain_id: Optional domain to filter by

    Returns:
        Number of bullets removed
    """
    stmt = delete(PlaybookBullet).where(
        PlaybookBullet.helpful_count == 0,
        PlaybookBullet.harmful_count == 0,
    )

    if domain_id:
        stmt = stmt.where(PlaybookBullet.domain_id == domain_id)

    result = session.execute(stmt)
    return result.rowcount


def consolidate_domain(
    session: Session,
    domain_id: str,
    similarity_threshold: float,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Consolidate playbook for a single domain.

    Returns:
        (num_merged, num_stale_removed)
    """
    print(f"\n[{domain_id}]")

    # Find duplicates
    duplicates = find_duplicates(session, domain_id, similarity_threshold)
    print(f"  Found {len(duplicates)} duplicate pairs (≥{similarity_threshold} similarity)")

    num_merged = 0
    if duplicates and not dry_run:
        for b1_id, b2_id, similarity in duplicates:
            merge_duplicates(session, b1_id, b2_id)
            num_merged += 1

    # Remove stale bullets
    if not dry_run:
        num_stale = remove_stale_bullets(session, domain_id)
    else:
        # Count stale bullets in dry run
        stmt = select(func.count()).select_from(PlaybookBullet).where(
            PlaybookBullet.domain_id == domain_id,
            PlaybookBullet.helpful_count == 0,
            PlaybookBullet.harmful_count == 0,
        )
        num_stale = session.execute(stmt).scalar() or 0

    print(f"  Removed {num_stale} stale bullets (helpful=0, harmful=0)")

    return num_merged, num_stale


def main():
    """Run playbook consolidation."""
    # Create database engine
    engine = create_db_engine()

    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN MODE - No changes will be made\n")

    with Session(engine) as session:
        # Get all domains
        stmt = select(PlaybookBullet.domain_id).distinct()
        domains = list(session.execute(stmt).scalars())

        print(f"Consolidating {len(domains)} domains with similarity threshold {similarity_threshold}")

        total_merged = 0
        total_stale = 0

        for domain in sorted(domains):
            num_merged, num_stale = consolidate_domain(
                session, domain, similarity_threshold, dry_run
            )
            total_merged += num_merged
            total_stale += num_stale

        if not dry_run:
            session.commit()
            print(f"\n✓ Consolidation complete:")
            print(f"  - {total_merged} duplicate bullets merged")
            print(f"  - {total_stale} stale bullets removed")
        else:
            print(f"\n[DRY RUN] Would merge {total_merged} and remove {total_stale} bullets")


if __name__ == "__main__":
    main()
