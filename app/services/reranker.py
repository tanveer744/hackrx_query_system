"""
Reranker Service for HackRx Query System
Improves retrieval accuracy by reranking search results using cross-encoder models.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder based reranker for improving search result quality."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers package not available. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        
        logger.info(f"Initialized cross-encoder reranker with model: {model_name}")
    
    def rerank(self, query: str, documents: List[str], scores: List[float] = None) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query-document relevance.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            scores: Optional original scores (not used in cross-encoder)
        
        Returns:
            List of (original_index, rerank_score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Get cross-encoder scores
        rerank_scores = self.model.predict(pairs)
        
        # Create (index, score) pairs and sort by score (descending)
        indexed_scores = [(i, float(score)) for i, score in enumerate(rerank_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Reranked {len(documents)} documents")
        return indexed_scores
    
    def rerank_search_results(self, query: str, search_results: List[Dict[str, Any]], 
                             text_field: str = "text") -> List[Dict[str, Any]]:
        """
        Rerank search results from vector store.
        
        Args:
            query: Search query
            search_results: List of search result dictionaries
            text_field: Field name containing the text to rerank
        
        Returns:
            Reranked search results with added rerank_score field
        """
        if not search_results:
            return []
        
        # Extract texts for reranking
        documents = [result.get(text_field, "") for result in search_results]
        original_scores = [result.get("similarity_score", 0.0) for result in search_results]
        
        # Perform reranking
        reranked_indices = self.rerank(query, documents, original_scores)
        
        # Reorder results and add rerank scores
        reranked_results = []
        for rank, (original_idx, rerank_score) in enumerate(reranked_indices):
            result = search_results[original_idx].copy()
            result["rerank_score"] = rerank_score
            result["rerank_position"] = rank + 1
            result["original_position"] = original_idx + 1
            reranked_results.append(result)
        
        return reranked_results


class HybridReranker:
    """Hybrid reranker that combines vector similarity and cross-encoder scores."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 alpha: float = 0.7):
        """
        Initialize hybrid reranker.
        
        Args:
            model_name: Cross-encoder model name
            alpha: Weight for cross-encoder score (1-alpha for vector similarity)
        """
        self.cross_encoder = CrossEncoderReranker(model_name)
        self.alpha = alpha
        
        logger.info(f"Initialized hybrid reranker with alpha={alpha}")
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)  # All scores are the same
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def rerank_search_results(self, query: str, search_results: List[Dict[str, Any]], 
                             text_field: str = "text") -> List[Dict[str, Any]]:
        """
        Rerank using hybrid approach combining vector similarity and cross-encoder.
        
        Args:
            query: Search query
            search_results: List of search result dictionaries
            text_field: Field name containing the text to rerank
        
        Returns:
            Reranked search results with hybrid scores
        """
        if not search_results:
            return []
        
        # Get cross-encoder reranking
        cross_encoder_results = self.cross_encoder.rerank_search_results(query, search_results, text_field)
        
        # Extract scores for normalization
        vector_scores = [result.get("similarity_score", 0.0) for result in search_results]
        cross_encoder_scores = [result["rerank_score"] for result in cross_encoder_results]
        
        # Normalize scores (note: for L2 distance, lower is better, so we invert)
        # Convert L2 distances to similarity scores (higher is better)
        vector_similarities = [1.0 / (1.0 + score) for score in vector_scores]
        normalized_vector = self.normalize_scores(vector_similarities)
        normalized_cross = self.normalize_scores(cross_encoder_scores)
        
        # Combine scores
        hybrid_results = []
        for i, result in enumerate(cross_encoder_results):
            # Find original index to get normalized vector score
            orig_idx = result["original_position"] - 1
            
            # Calculate hybrid score
            hybrid_score = (self.alpha * normalized_cross[i] + 
                           (1 - self.alpha) * normalized_vector[orig_idx])
            
            result_copy = result.copy()
            result_copy["hybrid_score"] = hybrid_score
            result_copy["normalized_vector_score"] = normalized_vector[orig_idx]
            result_copy["normalized_cross_score"] = normalized_cross[i]
            
            hybrid_results.append(result_copy)
        
        # Sort by hybrid score (descending)
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Update positions
        for i, result in enumerate(hybrid_results):
            result["final_position"] = i + 1
        
        return hybrid_results


class RerankerService:
    """Main reranker service with multiple reranking strategies."""
    
    def __init__(self, strategy: str = "cross_encoder", **kwargs):
        """
        Initialize reranker service.
        
        Args:
            strategy: "cross_encoder" or "hybrid"
            **kwargs: Additional arguments for the reranker
        """
        self.strategy = strategy.lower()
        
        if self.strategy == "cross_encoder":
            self.reranker = CrossEncoderReranker(**kwargs)
        elif self.strategy == "hybrid":
            self.reranker = HybridReranker(**kwargs)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}. Use 'cross_encoder' or 'hybrid'")
        
        logger.info(f"Reranker service initialized with strategy: {self.strategy}")
    
    def rerank(self, query: str, search_results: List[Dict[str, Any]], 
              text_field: str = "text") -> List[Dict[str, Any]]:
        """
        Rerank search results.
        
        Args:
            query: Search query
            search_results: List of search result dictionaries
            text_field: Field name containing the text to rerank
        
        Returns:
            Reranked search results
        """
        return self.reranker.rerank_search_results(query, search_results, text_field)


# Factory function for easy initialization
def create_reranker(strategy: str = "cross_encoder", **kwargs) -> RerankerService:
    """
    Factory function to create a reranker service.
    
    Args:
        strategy: "cross_encoder" or "hybrid"
        **kwargs: Strategy-specific arguments
    
    Returns:
        RerankerService instance
    """
    return RerankerService(strategy=strategy, **kwargs)


# Global reranker instance (lazy loaded)
_global_reranker = None

def rerank_chunks(query: str, chunks: List[str], top_k: int = 5) -> List[int]:
    """
    Simple function to rerank chunks using cross-encoder.
    This is the function that main.py imports.
    
    Args:
        query: Search query
        chunks: List of text chunks to rerank
        top_k: Number of top chunks to return
    
    Returns:
        List of indices of top chunks sorted by relevance
    """
    global _global_reranker
    
    if not CROSS_ENCODER_AVAILABLE:
        logger.warning("Cross-encoder not available, returning original order")
        return list(range(min(top_k, len(chunks))))
    
    try:
        # Lazy load the reranker
        if _global_reranker is None:
            _global_reranker = CrossEncoderReranker()
            logger.info("Loaded global cross-encoder reranker")
        
        # Get rerank scores
        reranked_results = _global_reranker.rerank(query, chunks)
        
        # Return top_k indices
        top_indices = [idx for idx, score in reranked_results[:top_k]]
        
        logger.info(f"Reranked {len(chunks)} chunks, returning top {len(top_indices)}")
        return top_indices
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Fallback: return original order
        return list(range(min(top_k, len(chunks))))


# Example usage and testing
if __name__ == "__main__":
    # Sample search results
    sample_results = [
        {
            "text": "A grace period of thirty days is allowed for premium payment.",
            "source": "policy.pdf",
            "page": 1,
            "similarity_score": 0.8
        },
        {
            "text": "The policy covers medical expenses up to $100,000 annually.",
            "source": "policy.pdf", 
            "page": 2,
            "similarity_score": 1.2
        },
        {
            "text": "Claims must be submitted within 90 days of the incident.",
            "source": "policy.pdf",
            "page": 3,
            "similarity_score": 0.9
        }
    ]
    
    query = "What is the grace period for premium payment?"
    
    try:
        # Test cross-encoder reranker
        reranker = create_reranker("cross_encoder")
        reranked_results = reranker.rerank(query, sample_results)
        
        print("Cross-encoder reranking results:")
        for i, result in enumerate(reranked_results):
            print(f"Rank {i+1}: {result['text'][:50]}... (score: {result['rerank_score']:.4f})")
        
        # Test hybrid reranker
        hybrid_reranker = create_reranker("hybrid", alpha=0.7)
        hybrid_results = hybrid_reranker.rerank(query, sample_results)
        
        print("\nHybrid reranking results:")
        for i, result in enumerate(hybrid_results):
            print(f"Rank {i+1}: {result['text'][:50]}... (hybrid: {result['hybrid_score']:.4f})")
            
    except ImportError:
        print("Reranker dependencies not available. Install sentence-transformers to use reranking.")
    except Exception as e:
        print(f"Error testing reranker: {e}")
