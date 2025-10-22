"""
Shared test fixtures and configuration for ACE Playbook tests.

This module provides shared pytest fixtures used across all test files,
particularly the mock_sentence_transformers fixture that ensures consistent
embedding behavior in tests.
"""

from pathlib import Path
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Ensure the repository root is importable so ``import ace`` works without
# installing the package in editable mode.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module", autouse=True)
def mock_sentence_transformers():
    """
    Mock SentenceTransformer to avoid downloading models during tests.

    Uses numpy's RandomState with text-derived seeds to generate reproducible
    but truly random embeddings. This approach avoids false duplicate detection
    caused by hash-based or character-position approaches that produce similar
    vectors after normalization.

    Scope: module - one mock per test file
    Autouse: True - automatically applied to all tests
    """
    with patch("ace.utils.embeddings.SentenceTransformer") as mock_cls:
        mock_model = Mock()

        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]

            embeddings = []
            for text in texts:
                # Use hash-based seed with random generation for better orthogonality
                # Pure hash mapping creates vectors that are too similar after normalization
                import hashlib

                # Generate seed from full text hash
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                seed_value = int(text_hash[:16], 16)  # Use 16 hex chars for larger seed space

                # Use numpy random with this seed - provides better orthogonality than pure hash mapping
                rng = np.random.RandomState(seed_value % (2**32))

                # Generate random values
                embedding = rng.randn(384).astype(np.float32)

                # Normalize to unit vector (standard for sentence embeddings)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                embeddings.append(embedding)

            return np.array(embeddings)

        mock_model.encode.side_effect = mock_encode
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_cls.return_value = mock_model
        yield mock_cls
