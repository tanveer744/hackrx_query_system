"""
Embedding Service for HackRx Query System
Handles text-to-vector conversion using OpenAI or HuggingFace models.
"""

import os
import logging
from typing import List, Union, Optional
import numpy as np
from abc import ABC, abstractmethod

# Optional imports - will be loaded based on availability
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text input."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding service using text-embedding-3-small model."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        self._embedding_dimension = 1536  # text-embedding-3-small dimension
        
        logger.info(f"Initialized OpenAI embedder with model: {model}")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if isinstance(text, str):
            text = [text]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self._embedding_dimension


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace embedding service using BGE-small model."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package not available. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized HuggingFace embedder with model: {model_name}")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using HuggingFace model."""
        if isinstance(text, str):
            text = [text]
        
        try:
            embeddings = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating HuggingFace embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self._embedding_dimension


class EmbeddingService:
    """Main embedding service that manages different embedding providers."""
    
    def __init__(self, provider: str = "openai", **kwargs):
        """
        Initialize embedding service.
        
        Args:
            provider: Either "openai" or "huggingface"
            **kwargs: Additional arguments for the specific embedder
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.embedder = OpenAIEmbedder(**kwargs)
        elif self.provider == "huggingface":
            self.embedder = HuggingFaceEmbedder(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'huggingface'")
        
        logger.info(f"Embedding service initialized with provider: {self.provider}")
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embedder.embed_text(text)[0]
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.embed_text(batch)
            all_embeddings.append(batch_embeddings)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedder.get_embedding_dimension()
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        return self.embed_single(query)


# Factory function for easy initialization
def create_embedder(provider: str = "openai", **kwargs) -> EmbeddingService:
    """
    Factory function to create an embedding service.
    
    Args:
        provider: "openai" or "huggingface"
        **kwargs: Provider-specific arguments
    
    Returns:
        EmbeddingService instance
    """
    return EmbeddingService(provider=provider, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample texts
    sample_texts = [
        "A grace period of thirty days is allowed for premium payment.",
        "The policy covers medical expenses up to $100,000 annually.",
        "Claims must be submitted within 90 days of the incident."
    ]
    
    # Try OpenAI first, fallback to HuggingFace
    try:
        embedder = create_embedder("openai")
        print("Using OpenAI embedder")
    except (ImportError, ValueError):
        try:
            embedder = create_embedder("huggingface")
            print("Using HuggingFace embedder")
        except ImportError:
            print("No embedding providers available. Install required packages.")
            exit(1)
    
    # Test single embedding
    single_embedding = embedder.embed_single(sample_texts[0])
    print(f"Single embedding shape: {single_embedding.shape}")
    
    # Test batch embedding
    batch_embeddings = embedder.embed_batch(sample_texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # Test query embedding
    query_embedding = embedder.embed_query("What is the grace period?")
    print(f"Query embedding shape: {query_embedding.shape}")
    
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
