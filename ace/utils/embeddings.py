"""
Embedding Generation Utilities

Wraps sentence-transformers for consistent 384-dim embeddings across ACE framework.
"""

import os
import builtins
import warnings
from typing import List, Union

import numpy as np

# Disable bitsandbytes auto-detection on CPU-only environments to avoid AttributeError
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

# Suppress noisy CPU-only bitsandbytes initialization message during import
_original_print = builtins.print


def _suppress_bitsandbytes_message(*args: any, **kwargs: any) -> None:
    """Suppress noisy bitsandbytes initialization messages.

    Args:
        *args: Positional arguments to print
        **kwargs: Keyword arguments to print
    """
    if len(args) == 1 and isinstance(args[0], str) and "cadam32bit_grad_fp32" in args[0]:
        return
    _original_print(*args, **kwargs)


builtins.print = _suppress_bitsandbytes_message
warnings.filterwarnings(
    "ignore",
    message="The installed version of bitsandbytes was compiled without GPU support.*",
)
try:
    from sentence_transformers import SentenceTransformer
finally:
    builtins.print = _original_print
    del _suppress_bitsandbytes_message

from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="embeddings")


class EmbeddingService:
    """
    Embedding service for generating semantic vectors from text.

    Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) by default.
    Implements caching and batch processing for efficiency.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Args:
            model_name: sentence-transformers model name (default: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self._model = None
        logger.info("embedding_service_init", model_name=model_name)

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model on first use."""
        if self._model is None:
            logger.info("loading_embedding_model", model_name=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", dimension=self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding (default: 32)

        Returns:
            numpy array of shape (n_texts, embedding_dim)

        Example:
            embeddings = service.encode(["Strategy 1", "Strategy 2"])
            # shape: (2, 384)
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.debug(
            "encoding_texts",
            num_texts=len(texts),
            batch_size=batch_size,
            model_name=self.model_name,
        )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        logger.debug(
            "embeddings_generated",
            num_texts=len(texts),
            embedding_shape=embeddings.shape,
        )

        return embeddings

    def encode_single(self, text: str) -> List[float]:
        """
        Generate embedding for single text, returning as list.

        Args:
            text: Text to encode

        Returns:
            List of floats (384-dim)

        Example:
            embedding = service.encode_single("Break problem into steps")
            # Returns: [0.123, -0.456, ..., 0.789] (384 values)
        """
        embedding = self.encode(text)[0]
        return embedding.tolist()

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for batch of texts, returning as list of lists.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            List of embedding lists

        Example:
            embeddings = service.encode_batch([
                "Strategy 1",
                "Strategy 2",
                "Strategy 3"
            ])
            # Returns: [[...], [...], [...]]  (3 x 384)
        """
        embeddings = self.encode(texts, batch_size=batch_size)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension (384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()


# Global singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingService:
    """
    Get global embedding service instance (singleton).

    Args:
        model_name: Model to use (only applies on first call)

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=model_name)
    return _embedding_service
