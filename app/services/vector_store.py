"""
Vector Store Service for HackRx Query System
Handles FAISS vector indexing, storage, and similarity search.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, embedding_dimension: int, index_type: str = "flat"):
        """
        Initialize vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings
            index_type: Type of FAISS index ("flat" for IndexFlatL2)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS package not available. Install with: pip install faiss-cpu")
        
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(embedding_dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Metadata storage (list of dicts)
        self.metadata: List[Dict[str, Any]] = []
        
        # Track if index is trained (for some index types)
        self.is_trained = True  # IndexFlatL2 doesn't need training
        
        logger.info(f"Initialized FAISS vector store with dimension {embedding_dimension}")
    
    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add vectors to the index with associated metadata.
        
        Args:
            embeddings: Array of embeddings (n_vectors, embedding_dim)
            metadata: List of metadata dicts for each vector
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self.embedding_dimension}")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Add metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total vectors: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector (1D array)
            k: Number of results to return
        
        Returns:
            Tuple of (distances, metadata_list)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning empty results")
            return [], []
        
        # Ensure query is 2D array with float32 dtype
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Extract results
        distances = distances[0].tolist()  # Convert to list
        indices = indices[0].tolist()
        
        # Get corresponding metadata
        results_metadata = []
        for idx in indices:
            if 0 <= idx < len(self.metadata):
                results_metadata.append(self.metadata[idx])
            else:
                logger.warning(f"Invalid index {idx} in search results")
        
        logger.debug(f"Search returned {len(results_metadata)} results")
        return distances, results_metadata
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Update dimension from loaded index
        self.embedding_dimension = self.index.d
        
        logger.info(f"Loaded index from {index_path} with {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
            "metadata_count": len(self.metadata)
        }
    
    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        # Reset index
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Clear metadata
        self.metadata = []
        
        logger.info("Cleared vector store")


class DocumentVectorStore:
    """High-level interface for document-based vector storage."""
    
    def __init__(self, embedding_dimension: int, data_dir: str = "data"):
        """
        Initialize document vector store.
        
        Args:
            embedding_dimension: Dimension of embeddings
            data_dir: Directory to store index and metadata files
        """
        self.data_dir = data_dir
        self.vector_store = VectorStore(embedding_dimension)
        
        # File paths
        self.index_path = os.path.join(data_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(data_dir, "metadata.json")
        
        # Try to load existing index
        self.load_if_exists()
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, 
                     sources: List[str], pages: Optional[List[int]] = None,
                     sections: Optional[List[str]] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: Corresponding embeddings
            sources: Source document names
            pages: Page numbers (optional)
            sections: Section headings (optional)
        """
        # Create metadata
        metadata = []
        for i, text in enumerate(texts):
            meta = {
                "text": text,
                "source": sources[i] if i < len(sources) else "unknown",
                "page": pages[i] if pages and i < len(pages) else None,
                "section": sections[i] if sections and i < len(sections) else None,
                "chunk_id": len(self.vector_store.metadata) + i
            }
            metadata.append(meta)
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, metadata)
        
        # Auto-save
        self.save()
    
    def search_documents(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
        
        Returns:
            List of search results with scores and metadata
        """
        distances, metadata_list = self.vector_store.search(query_embedding, k)
        
        # Combine distances with metadata
        results = []
        for distance, metadata in zip(distances, metadata_list):
            result = metadata.copy()
            result["similarity_score"] = float(distance)  # L2 distance (lower is better)
            results.append(result)
        
        return results
    
    def save(self) -> None:
        """Save the vector store to disk."""
        self.vector_store.save(self.index_path, self.metadata_path)
    
    def load_if_exists(self) -> bool:
        """Load existing vector store if files exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.vector_store.load(self.index_path, self.metadata_path)
                logger.info("Loaded existing vector store")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the document store."""
        stats = self.vector_store.get_stats()
        stats["data_directory"] = self.data_dir
        stats["index_file"] = self.index_path
        stats["metadata_file"] = self.metadata_path
        return stats


def create_faiss_index_from_embeddings(
    texts: list,
    embeddings: np.ndarray,
    sources: list = None,
    pages: list = None,
    sections: list = None,
    data_dir: str = "data"
) -> DocumentVectorStore:
    """
    Utility function to create a FAISS index from embeddings and metadata.
    Returns a DocumentVectorStore instance.
    """
    if sources is None:
        sources = ["unknown"] * len(texts)
    doc_store = DocumentVectorStore(embeddings.shape[1], data_dir)
    doc_store.add_documents(texts, embeddings, sources, pages, sections)
    return doc_store


def semantic_search(
    doc_store: DocumentVectorStore,
    query_embedding: np.ndarray,
    k: int = 5
) -> list:
    """
    Utility function to perform top-k semantic search using a DocumentVectorStore.
    Returns a list of result dicts.
    """
    return doc_store.search_documents(query_embedding, k)


class FAISSIndex:
    """
    Simple FAISS index class matching the requested API.
    Provides add() and search() methods with metadata support.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS index with given dimension.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        try:
            import faiss
            self.faiss = faiss
            self.dimension = dimension
            
            # Create a flat L2 index (most basic FAISS index)
            self.index = faiss.IndexFlatL2(dimension)
            
            # Store metadata and texts separately
            self.texts = []
            self.metadata = []
            
        except ImportError:
            raise ImportError("FAISS library not installed. Install with: pip install faiss-cpu")
    
    def add(self, embeddings: List[List[float]], texts: List[str], metas: List[dict]):
        """
        Add embeddings, texts, and metadata to the index.
        
        Args:
            embeddings: List of embedding vectors (as lists of floats)
            texts: List of text strings corresponding to embeddings
            metas: List of metadata dictionaries
        """
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(metas)
    
    def search(self, query_embedding: List[float], k: int = 1) -> List[dict]:
        """
        Search the index for similar vectors.
        
        Args:
            query_embedding: Query vector as list of floats
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text, metadata, and similarity scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        distances, indices = self.index.search(query_array, k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):
                result = {
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distance),
                    'similarity_score': float(1.0 / (1.0 + distance))  # Convert distance to similarity
                }
                results.append(result)
        
        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        import os
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        self.faiss.write_index(self.index, index_path)
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': self.texts,
                'metadata': self.metadata
            }, f, indent=2, ensure_ascii=False)

    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load the FAISS index and metadata from disk.
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.index = self.faiss.read_index(index_path)
        import json
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.texts = data.get('texts', [])
            self.metadata = data.get('metadata', [])

    def get_top_k_chunks(self, query: str, embedder, k: int = 5) -> list:
        """
        Embed the query string and return the top-k most similar chunks.
        Args:
            query: The query string
            embedder: An embedder object with an embed_single(str) -> List[float] method
            k: Number of top results to return
        Returns:
            List of dictionaries with text, metadata, and similarity scores
        """
        query_embedding = embedder.embed_single(query)
        return self.search(query_embedding, k)


# Example usage and testing
if __name__ == "__main__":
    # Test vector store with sample data
    embedding_dim = 384  # BGE-small dimension
    
    # Create sample embeddings and metadata
    sample_embeddings = np.random.rand(3, embedding_dim).astype(np.float32)
    sample_texts = [
        "A grace period of thirty days is allowed for premium payment.",
        "The policy covers medical expenses up to $100,000 annually.",
        "Claims must be submitted within 90 days of the incident."
    ]
    
    # Test basic vector store
    vs = VectorStore(embedding_dim)
    
    # Create metadata
    metadata = [
        {"text": text, "source": "policy.pdf", "page": i+1, "chunk_id": i}
        for i, text in enumerate(sample_texts)
    ]
    
    # Add vectors
    vs.add_vectors(sample_embeddings, metadata)
    
    # Test search
    query_embedding = np.random.rand(embedding_dim).astype(np.float32)
    distances, results = vs.search(query_embedding, k=2)
    
    print(f"Search results: {len(results)} items")
    for i, (dist, meta) in enumerate(zip(distances, results)):
        print(f"Result {i+1}: distance={dist:.4f}, text='{meta['text'][:50]}...'")
    
    # Test document vector store
    doc_store = DocumentVectorStore(embedding_dim, "test_data")
    doc_store.add_documents(
        texts=sample_texts,
        embeddings=sample_embeddings,
        sources=["policy.pdf"] * 3,
        pages=[1, 2, 3]
    )
    
    # Search documents
    doc_results = doc_store.search_documents(query_embedding, k=2)
    print(f"\nDocument search results: {len(doc_results)} items")
    for result in doc_results:
        print(f"Score: {result['similarity_score']:.4f}, Text: '{result['text'][:50]}...'")
    
    print(f"\nVector store stats: {doc_store.get_stats()}")
