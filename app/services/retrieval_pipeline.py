"""
Complete Retrieval Pipeline for HackRx Query System
Integrates embedder, vector store, and reranker into a unified service.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .embedder import create_embedder, EmbeddingService
from .vector_store import DocumentVectorStore
from .reranker import create_reranker, RerankerService

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Complete retrieval pipeline for semantic search."""
    
    def __init__(self, 
                 embedding_provider: str = "openai",
                 embedding_model: Optional[str] = None,
                 use_reranker: bool = True,
                 reranker_strategy: str = "cross_encoder",
                 data_dir: str = "data",
                 **kwargs):
        """
        Initialize the complete retrieval pipeline.
        
        Args:
            embedding_provider: "openai" or "huggingface"
            embedding_model: Specific model name (optional)
            use_reranker: Whether to use reranking
            reranker_strategy: "cross_encoder" or "hybrid"
            data_dir: Directory for storing vector index
            **kwargs: Additional arguments for components
        """
        self.embedding_provider = embedding_provider
        self.use_reranker = use_reranker
        self.data_dir = data_dir
        
        # Initialize embedder
        embedder_kwargs = {}
        if embedding_model:
            if embedding_provider == "openai":
                embedder_kwargs["model"] = embedding_model
            else:
                embedder_kwargs["model_name"] = embedding_model
        
        self.embedder = create_embedder(embedding_provider, **embedder_kwargs)
        logger.info(f"Initialized embedder: {embedding_provider}")
        
        # Initialize vector store
        embedding_dim = self.embedder.get_embedding_dimension()
        self.vector_store = DocumentVectorStore(embedding_dim, data_dir)
        logger.info(f"Initialized vector store with dimension {embedding_dim}")
        
        # Initialize reranker (optional)
        self.reranker = None
        if use_reranker:
            try:
                self.reranker = create_reranker(reranker_strategy, **kwargs)
                logger.info(f"Initialized reranker: {reranker_strategy}")
            except ImportError:
                logger.warning("Reranker dependencies not available, proceeding without reranking")
                self.use_reranker = False
    
    def add_documents(self, 
                     documents: List[Dict[str, Any]],
                     batch_size: int = 100) -> None:
        """
        Add documents to the retrieval system.
        
        Args:
            documents: List of document dictionaries with 'text' and metadata
            batch_size: Batch size for embedding generation
        """
        if not documents:
            logger.warning("No documents provided")
            return
        
        # Extract texts and metadata
        texts = [doc["text"] for doc in documents]
        sources = [doc.get("source", "unknown") for doc in documents]
        pages = [doc.get("page") for doc in documents]
        sections = [doc.get("section") for doc in documents]
        
        logger.info(f"Processing {len(documents)} documents...")
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedder.embed_batch(batch_texts, batch_size)
            all_embeddings.append(batch_embeddings)
            logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Add to vector store
        self.vector_store.add_documents(texts, embeddings, sources, pages, sections)
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, 
               query: str, 
               k: int = 5,
               use_reranking: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            use_reranking: Override default reranking setting
        
        Returns:
            List of search results with scores and metadata
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Determine if we should use reranking
        should_rerank = use_reranking if use_reranking is not None else self.use_reranker
        
        # Step 1: Embed query
        query_embedding = self.embedder.embed_query(query)
        logger.debug(f"Generated query embedding for: '{query[:50]}...'")
        
        # Step 2: Vector search
        # Get more results if we're going to rerank
        search_k = k * 2 if should_rerank and self.reranker else k
        search_results = self.vector_store.search_documents(query_embedding, search_k)
        
        if not search_results:
            logger.info("No search results found")
            return []
        
        logger.debug(f"Vector search returned {len(search_results)} results")
        
        # Step 3: Rerank if enabled
        if should_rerank and self.reranker:
            try:
                reranked_results = self.reranker.rerank(query, search_results)
                final_results = reranked_results[:k]  # Take top k after reranking
                logger.debug(f"Reranking returned {len(final_results)} results")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using vector search results")
                final_results = search_results[:k]
        else:
            final_results = search_results[:k]
        
        return final_results
    
    def get_relevant_chunks(self, 
                           query: str, 
                           k: int = 5,
                           min_score_threshold: Optional[float] = None) -> List[str]:
        """
        Get relevant text chunks for LLM context.
        
        Args:
            query: Search query
            k: Number of chunks to return
            min_score_threshold: Minimum similarity score threshold
        
        Returns:
            List of relevant text chunks
        """
        results = self.search(query, k)
        
        # Filter by score threshold if provided
        if min_score_threshold is not None:
            # For L2 distance, lower is better, so we filter by maximum distance
            results = [r for r in results if r.get("similarity_score", float('inf')) <= min_score_threshold]
        
        # Extract text chunks
        chunks = [result["text"] for result in results]
        logger.info(f"Retrieved {len(chunks)} relevant chunks for query")
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        stats = self.vector_store.get_stats()
        stats.update({
            "embedding_provider": self.embedding_provider,
            "embedding_dimension": self.embedder.get_embedding_dimension(),
            "reranker_enabled": self.use_reranker,
            "reranker_available": self.reranker is not None
        })
        return stats
    
    def clear_index(self) -> None:
        """Clear all documents from the index."""
        self.vector_store.vector_store.clear()
        logger.info("Cleared vector index")


# Factory function for easy initialization
def create_retrieval_pipeline(**kwargs) -> RetrievalPipeline:
    """
    Factory function to create a retrieval pipeline.
    
    Args:
        **kwargs: Arguments for RetrievalPipeline
    
    Returns:
        RetrievalPipeline instance
    """
    return RetrievalPipeline(**kwargs)


# Example usage for FastAPI integration
class HackRxRetrieval:
    """High-level interface for HackRx API integration."""
    
    def __init__(self):
        """Initialize with default settings optimized for HackRx."""
        # Try OpenAI first, fallback to HuggingFace
        try:
            self.pipeline = create_retrieval_pipeline(
                embedding_provider="openai",
                use_reranker=True,
                reranker_strategy="hybrid",
                alpha=0.7  # Hybrid reranker weight
            )
            logger.info("Initialized HackRx retrieval with OpenAI embeddings")
        except (ImportError, ValueError):
            self.pipeline = create_retrieval_pipeline(
                embedding_provider="huggingface",
                use_reranker=True,
                reranker_strategy="cross_encoder"
            )
            logger.info("Initialized HackRx retrieval with HuggingFace embeddings")
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Process and index documents."""
        self.pipeline.add_documents(documents)
    
    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the system for relevant information."""
        return self.pipeline.search(question, k=top_k)
    
    def get_context_for_llm(self, question: str, max_chunks: int = 3) -> str:
        """Get formatted context for LLM prompt."""
        chunks = self.pipeline.get_relevant_chunks(question, k=max_chunks)
        
        if not chunks:
            return "No relevant information found in the policy documents."
        
        # Format chunks for LLM context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Context {i}]\n{chunk}")
        
        return "\n\n".join(context_parts)


# Example usage and testing
if __name__ == "__main__":
    # Sample documents
    sample_docs = [
        {
            "text": "A grace period of thirty days is allowed for premium payment.",
            "source": "policy.pdf",
            "page": 1,
            "section": "Payment Terms"
        },
        {
            "text": "The policy covers medical expenses up to $100,000 annually.",
            "source": "policy.pdf",
            "page": 2,
            "section": "Coverage"
        }
    ]
    
    # Initialize pipeline
    try:
        retrieval = HackRxRetrieval()
        
        # Add documents
        retrieval.process_documents(sample_docs)
        
        # Test query
        results = retrieval.query("What is the grace period?")
        print(f"Found {len(results)} results")
        
        # Get LLM context
        context = retrieval.get_context_for_llm("What is the grace period?")
        print(f"LLM Context:\n{context}")
        
    except Exception as e:
        print(f"Error testing retrieval pipeline: {e}")
        print("Make sure to install required dependencies and set up API keys if using OpenAI.")
