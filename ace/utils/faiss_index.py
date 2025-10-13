"""
FAISS Index Management

Per-domain FAISS indices for fast cosine similarity search with multi-tenant isolation.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import faiss
import os
from pathlib import Path

from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="faiss")


class FAISSIndexManager:
    """
    Manages per-domain FAISS indices for semantic search.

    Implements CHK086: Separate embeddings index per domain.
    Uses IndexFlatIP (inner product) for exact cosine similarity.
    """

    def __init__(self, index_dir: str = "faiss_indices", dimension: int = 384):
        """
        Initialize FAISS index manager.

        Args:
            index_dir: Directory to store per-domain index files
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self._indices: Dict[str, faiss.IndexFlatIP] = {}
        self._bullet_id_maps: Dict[str, List[str]] = {}  # Maps index position to bullet ID

        logger.info(
            "faiss_manager_init",
            index_dir=str(self.index_dir),
            dimension=dimension,
        )

    def _get_index_path(self, domain_id: str) -> Path:
        """Get file path for domain's FAISS index."""
        return self.index_dir / f"faiss_index_{domain_id}.index"

    def _get_mapping_path(self, domain_id: str) -> Path:
        """Get file path for domain's bullet ID mapping."""
        return self.index_dir / f"faiss_index_{domain_id}.mapping"

    def get_or_create_index(self, domain_id: str) -> faiss.IndexFlatIP:
        """
        Get or create FAISS index for domain.

        Args:
            domain_id: Domain namespace (e.g., "customer-acme")

        Returns:
            FAISS IndexFlatIP instance
        """
        if domain_id in self._indices:
            return self._indices[domain_id]

        index_path = self._get_index_path(domain_id)
        if index_path.exists():
            # Load existing index
            logger.info("loading_faiss_index", domain_id=domain_id, path=str(index_path))
            index = faiss.read_index(str(index_path))
            self._indices[domain_id] = index

            # Load bullet ID mapping
            mapping_path = self._get_mapping_path(domain_id)
            if mapping_path.exists():
                with open(mapping_path, "r") as f:
                    self._bullet_id_maps[domain_id] = [line.strip() for line in f]
            else:
                self._bullet_id_maps[domain_id] = []
        else:
            # Create new index
            logger.info("creating_faiss_index", domain_id=domain_id, dimension=self.dimension)
            index = faiss.IndexFlatIP(self.dimension)
            self._indices[domain_id] = index
            self._bullet_id_maps[domain_id] = []

        return index

    def add_vectors(
        self, domain_id: str, embeddings: np.ndarray, bullet_ids: List[str]
    ) -> None:
        """
        Add embedding vectors to domain's FAISS index.

        Args:
            domain_id: Domain namespace
            embeddings: Numpy array of shape (n, dimension)
            bullet_ids: List of bullet IDs corresponding to embeddings
        """
        if len(embeddings) != len(bullet_ids):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) must match bullet_ids count ({len(bullet_ids)})"
            )

        index = self.get_or_create_index(domain_id)

        # Normalize for cosine similarity (use inner product on normalized vectors)
        embeddings_norm = embeddings.astype("float32")
        faiss.normalize_L2(embeddings_norm)

        # Add to index
        index.add(embeddings_norm)

        # Update bullet ID mapping
        self._bullet_id_maps[domain_id].extend(bullet_ids)

        logger.info(
            "added_vectors_to_index",
            domain_id=domain_id,
            num_vectors=len(embeddings),
            total_vectors=index.ntotal,
        )

    def search(
        self, domain_id: str, query_embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for k most similar vectors in domain's index.

        Args:
            domain_id: Domain namespace
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return

        Returns:
            List of (bullet_id, similarity_score) tuples sorted by similarity (descending)
        """
        index = self.get_or_create_index(domain_id)

        if index.ntotal == 0:
            logger.warning("empty_index_search", domain_id=domain_id)
            return []

        # Normalize query
        query_norm = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_norm)

        # Search
        similarities, indices = index.search(query_norm, min(k, index.ntotal))

        # Map indices to bullet IDs
        bullet_id_map = self._bullet_id_maps[domain_id]
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(bullet_id_map):
                bullet_id = bullet_id_map[idx]
                results.append((bullet_id, float(similarity)))

        logger.debug(
            "faiss_search_complete",
            domain_id=domain_id,
            k=k,
            results_count=len(results),
        )

        return results

    def save_index(self, domain_id: str) -> None:
        """Save domain's FAISS index and bullet ID mapping to disk."""
        if domain_id not in self._indices:
            logger.warning("no_index_to_save", domain_id=domain_id)
            return

        index = self._indices[domain_id]
        index_path = self._get_index_path(domain_id)

        # Save FAISS index
        faiss.write_index(index, str(index_path))

        # Save bullet ID mapping
        mapping_path = self._get_mapping_path(domain_id)
        with open(mapping_path, "w") as f:
            for bullet_id in self._bullet_id_maps[domain_id]:
                f.write(f"{bullet_id}\n")

        logger.info(
            "saved_faiss_index",
            domain_id=domain_id,
            path=str(index_path),
            num_vectors=index.ntotal,
        )

    def get_index_size(self, domain_id: str) -> int:
        """Get number of vectors in domain's index."""
        index = self.get_or_create_index(domain_id)
        return index.ntotal


# Global singleton instance
_faiss_manager: Optional[FAISSIndexManager] = None


def get_faiss_manager(index_dir: str = "faiss_indices", dimension: int = 384) -> FAISSIndexManager:
    """
    Get global FAISS index manager instance (singleton).

    Args:
        index_dir: Index directory (only applies on first call)
        dimension: Embedding dimension (only applies on first call)

    Returns:
        FAISSIndexManager instance
    """
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FAISSIndexManager(index_dir=index_dir, dimension=dimension)
    return _faiss_manager
