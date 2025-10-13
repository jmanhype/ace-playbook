"""
Unit tests for utility services (EmbeddingService and FAISSIndexManager).

Tests embeddings generation, FAISS index operations, and domain isolation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil

from ace.utils.embeddings import EmbeddingService, get_embedding_service
from ace.utils.faiss_index import FAISSIndexManager, get_faiss_manager


# ============================================================================
# EmbeddingService Tests
# ============================================================================


# Mock fixture now provided by tests/conftest.py


@pytest.fixture
def embedding_service():
    """Create EmbeddingService instance."""
    return EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_embedding_service_initialization(embedding_service):
    """Test EmbeddingService initializes correctly."""
    assert embedding_service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding_service._model is None  # Lazy loading


def test_lazy_loading_on_first_use(embedding_service):
    """Test model is lazy-loaded on first encode() call."""
    assert embedding_service._model is None

    # First encode triggers loading
    embedding_service.encode("test text")
    assert embedding_service._model is not None


def test_encode_single_text(embedding_service):
    """Test encoding single text string."""
    embedding = embedding_service.encode("Test strategy")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 384)
    assert np.allclose(np.linalg.norm(embedding[0]), 1.0, atol=1e-6)  # Unit vector


def test_encode_multiple_texts(embedding_service):
    """Test encoding multiple texts."""
    texts = ["Strategy 1", "Strategy 2", "Strategy 3"]
    embeddings = embedding_service.encode(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)
    for i in range(3):
        assert np.allclose(np.linalg.norm(embeddings[i]), 1.0, atol=1e-6)


def test_encode_single_wrapper(embedding_service):
    """Test encode_single() returns list of floats."""
    embedding = embedding_service.encode_single("Test text")

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)


def test_encode_batch_wrapper(embedding_service):
    """Test encode_batch() returns list of lists."""
    texts = ["Text A", "Text B"]
    embeddings = embedding_service.encode_batch(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == 384 for emb in embeddings)


def test_get_dimension(embedding_service):
    """Test get_dimension() returns correct embedding dimension."""
    dimension = embedding_service.get_dimension()
    assert dimension == 384


def test_encode_empty_string(embedding_service):
    """Test encoding empty string."""
    embedding = embedding_service.encode("")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 384)


def test_encode_unicode_text(embedding_service):
    """Test encoding unicode text."""
    text = "Strategy with émojis 🚀 and spëcial çharacters"
    embedding = embedding_service.encode(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 384)


def test_singleton_pattern():
    """Test get_embedding_service() returns singleton."""
    service1 = get_embedding_service()
    service2 = get_embedding_service()

    assert service1 is service2


# ============================================================================
# FAISSIndexManager Tests
# ============================================================================


@pytest.fixture
def temp_index_dir():
    """Create temporary directory for FAISS indices."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def faiss_manager(temp_index_dir):
    """Create FAISSIndexManager instance."""
    return FAISSIndexManager(index_dir=temp_index_dir, dimension=384)


def test_faiss_manager_initialization(faiss_manager, temp_index_dir):
    """Test FAISSIndexManager initializes correctly."""
    assert faiss_manager.dimension == 384
    assert faiss_manager.index_dir == Path(temp_index_dir)
    assert faiss_manager.index_dir.exists()


def test_create_new_index(faiss_manager):
    """Test creating new FAISS index for domain."""
    domain_id = "test-domain-001"
    index = faiss_manager.get_or_create_index(domain_id)

    assert index is not None
    assert index.ntotal == 0  # Empty index
    assert domain_id in faiss_manager._indices


def test_add_vectors_to_index(faiss_manager):
    """Test adding vectors to FAISS index."""
    domain_id = "test-domain-002"

    # Create test embeddings
    embeddings = np.random.randn(5, 384).astype("float32")
    bullet_ids = ["bullet-1", "bullet-2", "bullet-3", "bullet-4", "bullet-5"]

    # Add to index
    faiss_manager.add_vectors(domain_id, embeddings, bullet_ids)

    # Verify
    index = faiss_manager.get_or_create_index(domain_id)
    assert index.ntotal == 5
    assert len(faiss_manager._bullet_id_maps[domain_id]) == 5


def test_add_vectors_mismatched_lengths(faiss_manager):
    """Test that mismatched embeddings and IDs raises error."""
    domain_id = "test-domain-003"

    embeddings = np.random.randn(5, 384).astype("float32")
    bullet_ids = ["bullet-1", "bullet-2", "bullet-3"]  # Only 3 IDs for 5 embeddings

    with pytest.raises(ValueError, match="Embeddings count .* must match bullet_ids count"):
        faiss_manager.add_vectors(domain_id, embeddings, bullet_ids)


def test_search_similar_vectors(faiss_manager):
    """Test searching for similar vectors in index."""
    domain_id = "test-domain-004"

    # Create embeddings
    embeddings = np.random.randn(10, 384).astype("float32")
    bullet_ids = [f"bullet-{i}" for i in range(10)]

    # Add to index
    faiss_manager.add_vectors(domain_id, embeddings, bullet_ids)

    # Search with first embedding (should be most similar to itself)
    query_embedding = embeddings[0]
    results = faiss_manager.search(domain_id, query_embedding, k=5)

    assert len(results) == 5
    assert results[0][0] == "bullet-0"  # First result should be itself
    assert results[0][1] > 0.99  # Near-perfect similarity


def test_search_empty_index(faiss_manager):
    """Test searching empty index returns empty list."""
    domain_id = "test-domain-005"

    query_embedding = np.random.randn(384).astype("float32")
    results = faiss_manager.search(domain_id, query_embedding, k=10)

    assert results == []


def test_domain_isolation_in_indices(faiss_manager):
    """Test that domains have isolated FAISS indices."""
    domain_a = "test-domain-a"
    domain_b = "test-domain-b"

    # Add vectors to domain A
    embeddings_a = np.random.randn(5, 384).astype("float32")
    bullet_ids_a = [f"bullet-a-{i}" for i in range(5)]
    faiss_manager.add_vectors(domain_a, embeddings_a, bullet_ids_a)

    # Add vectors to domain B
    embeddings_b = np.random.randn(3, 384).astype("float32")
    bullet_ids_b = [f"bullet-b-{i}" for i in range(3)]
    faiss_manager.add_vectors(domain_b, embeddings_b, bullet_ids_b)

    # Verify isolation
    index_a = faiss_manager.get_or_create_index(domain_a)
    index_b = faiss_manager.get_or_create_index(domain_b)

    assert index_a.ntotal == 5
    assert index_b.ntotal == 3
    assert index_a is not index_b  # Different index objects

    # Search domain A - should not return domain B bullets
    query = embeddings_a[0]
    results = faiss_manager.search(domain_a, query, k=10)
    assert all("bullet-a" in result[0] for result in results)


def test_save_and_load_index(faiss_manager, temp_index_dir):
    """Test saving and loading FAISS index from disk."""
    domain_id = "test-domain-006"

    # Add vectors
    embeddings = np.random.randn(5, 384).astype("float32")
    bullet_ids = [f"bullet-{i}" for i in range(5)]
    faiss_manager.add_vectors(domain_id, embeddings, bullet_ids)

    # Save index
    faiss_manager.save_index(domain_id)

    # Verify files exist
    index_path = faiss_manager._get_index_path(domain_id)
    mapping_path = faiss_manager._get_mapping_path(domain_id)
    assert index_path.exists()
    assert mapping_path.exists()

    # Create new manager and load index
    new_manager = FAISSIndexManager(index_dir=temp_index_dir, dimension=384)
    loaded_index = new_manager.get_or_create_index(domain_id)

    # Verify loaded index has same data
    assert loaded_index.ntotal == 5
    assert len(new_manager._bullet_id_maps[domain_id]) == 5
    assert new_manager._bullet_id_maps[domain_id] == bullet_ids


def test_get_index_size(faiss_manager):
    """Test get_index_size() returns correct vector count."""
    domain_id = "test-domain-007"

    # Empty index
    assert faiss_manager.get_index_size(domain_id) == 0

    # Add vectors
    embeddings = np.random.randn(7, 384).astype("float32")
    bullet_ids = [f"bullet-{i}" for i in range(7)]
    faiss_manager.add_vectors(domain_id, embeddings, bullet_ids)

    assert faiss_manager.get_index_size(domain_id) == 7


def test_search_with_k_larger_than_index(faiss_manager):
    """Test searching with k > index size returns all vectors."""
    domain_id = "test-domain-008"

    # Add 3 vectors
    embeddings = np.random.randn(3, 384).astype("float32")
    bullet_ids = [f"bullet-{i}" for i in range(3)]
    faiss_manager.add_vectors(domain_id, embeddings, bullet_ids)

    # Search with k=10 (larger than index size)
    query = embeddings[0]
    results = faiss_manager.search(domain_id, query, k=10)

    assert len(results) == 3  # Returns all available vectors


def test_faiss_singleton_pattern():
    """Test get_faiss_manager() returns singleton."""
    manager1 = get_faiss_manager()
    manager2 = get_faiss_manager()

    assert manager1 is manager2


def test_cosine_similarity_with_normalization(faiss_manager):
    """Test that FAISS search returns cosine similarity scores."""
    domain_id = "test-domain-009"

    # Create two similar vectors (same direction, different magnitudes)
    base_vector = np.random.randn(384).astype("float32")
    similar_vector = base_vector * 2.0  # Same direction, different magnitude

    embeddings = np.array([base_vector, similar_vector])
    bullet_ids = ["bullet-1", "bullet-2"]

    faiss_manager.add_vectors(domain_id, embeddings, bullet_ids)

    # Search with base vector
    results = faiss_manager.search(domain_id, base_vector, k=2)

    # Both should have high similarity (cosine ignores magnitude)
    assert len(results) == 2
    assert results[0][1] > 0.99  # Near-perfect similarity with itself
    assert results[1][1] > 0.99  # Near-perfect similarity with scaled version


def test_incremental_vector_addition(faiss_manager):
    """Test adding vectors incrementally to index."""
    domain_id = "test-domain-010"

    # Add first batch
    embeddings1 = np.random.randn(3, 384).astype("float32")
    bullet_ids1 = ["bullet-1", "bullet-2", "bullet-3"]
    faiss_manager.add_vectors(domain_id, embeddings1, bullet_ids1)
    assert faiss_manager.get_index_size(domain_id) == 3

    # Add second batch
    embeddings2 = np.random.randn(2, 384).astype("float32")
    bullet_ids2 = ["bullet-4", "bullet-5"]
    faiss_manager.add_vectors(domain_id, embeddings2, bullet_ids2)
    assert faiss_manager.get_index_size(domain_id) == 5

    # Verify all bullet IDs are tracked
    assert len(faiss_manager._bullet_id_maps[domain_id]) == 5
