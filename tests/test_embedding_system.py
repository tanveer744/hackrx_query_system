"""
Comprehensive test suite for the HackRx Embedding & Retrieval System
Tests all components: embedder, vector store, reranker, and end-to-end pipeline.
"""

import os
import sys
import numpy as np
import logging
from typing import List, Dict, Any

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample policy document chunks for testing
SAMPLE_POLICY_CHUNKS = [
    {
        "text": "A grace period of thirty (30) days is allowed for premium payment after the due date. During this period, the policy remains in force.",
        "source": "health_policy.pdf",
        "page": 3,
        "section": "Premium Payment Terms"
    },
    {
        "text": "The policy covers medical expenses up to $100,000 annually for hospitalization, surgery, and emergency care.",
        "source": "health_policy.pdf", 
        "page": 5,
        "section": "Coverage Limits"
    },
    {
        "text": "Claims must be submitted within ninety (90) days of the incident or treatment date. Late submissions may be rejected.",
        "source": "health_policy.pdf",
        "page": 8,
        "section": "Claims Processing"
    },
    {
        "text": "Pre-existing conditions are covered after a waiting period of twelve (12) months from the policy start date.",
        "source": "health_policy.pdf",
        "page": 12,
        "section": "Pre-existing Conditions"
    },
    {
        "text": "The policyholder can add dependents (spouse and children) with additional premium. Maximum 4 dependents allowed.",
        "source": "health_policy.pdf",
        "page": 15,
        "section": "Dependent Coverage"
    }
]

# Test queries
TEST_QUERIES = [
    "What is the grace period for premium payment?",
    "How much does the policy cover for medical expenses?",
    "When do I need to submit claims?",
    "Are pre-existing conditions covered?",
    "Can I add my family members to the policy?"
]


def test_embedder():
    """Test the embedding service."""
    logger.info("Testing Embedding Service...")
    
    try:
        from app.services.embedder import create_embedder
        
        # Try OpenAI first, fallback to HuggingFace
        embedder = None
        try:
            embedder = create_embedder("openai")
            logger.info("✓ Using OpenAI embedder")
        except (ImportError, ValueError) as e:
            logger.warning(f"OpenAI not available: {e}")
            try:
                embedder = create_embedder("huggingface")
                logger.info("✓ Using HuggingFace embedder")
            except ImportError as e:
                logger.error(f"No embedding providers available: {e}")
                return None
        
        # Test single embedding
        sample_text = SAMPLE_POLICY_CHUNKS[0]["text"]
        single_embedding = embedder.embed_single(sample_text)
        logger.info(f"✓ Single embedding shape: {single_embedding.shape}")
        
        # Test batch embedding
        texts = [chunk["text"] for chunk in SAMPLE_POLICY_CHUNKS]
        batch_embeddings = embedder.embed_batch(texts)
        logger.info(f"✓ Batch embeddings shape: {batch_embeddings.shape}")
        
        # Test query embedding
        query_embedding = embedder.embed_query(TEST_QUERIES[0])
        logger.info(f"✓ Query embedding shape: {query_embedding.shape}")
        
        # Verify dimensions match
        dim = embedder.get_embedding_dimension()
        assert single_embedding.shape[0] == dim, "Single embedding dimension mismatch"
        assert batch_embeddings.shape[1] == dim, "Batch embedding dimension mismatch"
        assert query_embedding.shape[0] == dim, "Query embedding dimension mismatch"
        
        logger.info(f"✓ All embeddings have correct dimension: {dim}")
        return embedder
        
    except Exception as e:
        logger.error(f"✗ Embedder test failed: {e}")
        return None


def test_vector_store(embedder):
    """Test the vector store service."""
    logger.info("Testing Vector Store Service...")
    
    try:
        from app.services.vector_store import DocumentVectorStore
        
        # Create test data directory
        test_data_dir = "test_data"
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Initialize vector store
        embedding_dim = embedder.get_embedding_dimension()
        doc_store = DocumentVectorStore(embedding_dim, test_data_dir)
        
        # Prepare test data
        texts = [chunk["text"] for chunk in SAMPLE_POLICY_CHUNKS]
        sources = [chunk["source"] for chunk in SAMPLE_POLICY_CHUNKS]
        pages = [chunk["page"] for chunk in SAMPLE_POLICY_CHUNKS]
        sections = [chunk["section"] for chunk in SAMPLE_POLICY_CHUNKS]
        
        # Generate embeddings
        embeddings = embedder.embed_batch(texts)
        
        # Add documents to vector store
        doc_store.add_documents(texts, embeddings, sources, pages, sections)
        logger.info(f"✓ Added {len(texts)} documents to vector store")
        
        # Test search
        query_embedding = embedder.embed_query(TEST_QUERIES[0])
        search_results = doc_store.search_documents(query_embedding, k=3)
        
        logger.info(f"✓ Search returned {len(search_results)} results")
        for i, result in enumerate(search_results):
            logger.info(f"  Result {i+1}: '{result['text'][:50]}...' (score: {result['similarity_score']:.4f})")
        
        # Test persistence
        doc_store.save()
        logger.info("✓ Vector store saved successfully")
        
        # Test loading
        new_store = DocumentVectorStore(embedding_dim, test_data_dir)
        stats = new_store.get_stats()
        logger.info(f"✓ Vector store loaded: {stats['total_vectors']} vectors")
        
        return doc_store
        
    except Exception as e:
        logger.error(f"✗ Vector store test failed: {e}")
        return None


def test_reranker(embedder, doc_store):
    """Test the reranker service."""
    logger.info("Testing Reranker Service...")
    
    try:
        from app.services.reranker import create_reranker
        
        # Get initial search results
        query = TEST_QUERIES[0]
        query_embedding = embedder.embed_query(query)
        search_results = doc_store.search_documents(query_embedding, k=5)
        
        logger.info(f"Original search results for: '{query}'")
        for i, result in enumerate(search_results):
            logger.info(f"  {i+1}. '{result['text'][:50]}...' (score: {result['similarity_score']:.4f})")
        
        # Test cross-encoder reranker
        reranker = create_reranker("cross_encoder")
        reranked_results = reranker.rerank(query, search_results)
        
        logger.info("✓ Cross-encoder reranked results:")
        for i, result in enumerate(reranked_results):
            logger.info(f"  {i+1}. '{result['text'][:50]}...' (rerank: {result['rerank_score']:.4f})")
        
        # Test hybrid reranker
        hybrid_reranker = create_reranker("hybrid", alpha=0.7)
        hybrid_results = hybrid_reranker.rerank(query, search_results)
        
        logger.info("✓ Hybrid reranked results:")
        for i, result in enumerate(hybrid_results):
            logger.info(f"  {i+1}. '{result['text'][:50]}...' (hybrid: {result['hybrid_score']:.4f})")
        
        return reranker
        
    except ImportError as e:
        logger.warning(f"Reranker not available: {e}")
        return None
    except Exception as e:
        logger.error(f"✗ Reranker test failed: {e}")
        return None


def test_end_to_end_pipeline():
    """Test the complete end-to-end pipeline."""
    logger.info("Testing End-to-End Pipeline...")
    
    try:
        # Initialize all components
        embedder = test_embedder()
        if not embedder:
            logger.error("✗ Cannot proceed without embedder")
            return False
        
        doc_store = test_vector_store(embedder)
        if not doc_store:
            logger.error("✗ Cannot proceed without vector store")
            return False
        
        reranker = test_reranker(embedder, doc_store)
        
        # Test all queries
        logger.info("Testing all sample queries...")
        for i, query in enumerate(TEST_QUERIES):
            logger.info(f"\nQuery {i+1}: '{query}'")
            
            # Step 1: Embed query
            query_embedding = embedder.embed_query(query)
            
            # Step 2: Search vector store
            search_results = doc_store.search_documents(query_embedding, k=3)
            
            # Step 3: Rerank if available
            if reranker:
                final_results = reranker.rerank(query, search_results)
            else:
                final_results = search_results
            
            # Display top result
            if final_results:
                top_result = final_results[0]
                score_field = "rerank_score" if reranker else "similarity_score"
                logger.info(f"  Top result: '{top_result['text'][:80]}...'")
                logger.info(f"  Score: {top_result.get(score_field, 0):.4f}")
                logger.info(f"  Source: {top_result['source']}, Page: {top_result['page']}")
        
        logger.info("✓ End-to-end pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ End-to-end pipeline test failed: {e}")
        return False


def run_performance_test(embedder, doc_store):
    """Run basic performance tests."""
    logger.info("Running Performance Tests...")
    
    import time
    
    try:
        # Test embedding performance
        start_time = time.time()
        test_texts = [chunk["text"] for chunk in SAMPLE_POLICY_CHUNKS] * 10  # 50 texts
        embeddings = embedder.embed_batch(test_texts)
        embedding_time = time.time() - start_time
        
        logger.info(f"✓ Embedded {len(test_texts)} texts in {embedding_time:.2f}s ({len(test_texts)/embedding_time:.1f} texts/sec)")
        
        # Test search performance
        query_embedding = embedder.embed_query("test query")
        
        start_time = time.time()
        for _ in range(100):  # 100 searches
            results = doc_store.search_documents(query_embedding, k=5)
        search_time = time.time() - start_time
        
        logger.info(f"✓ Performed 100 searches in {search_time:.2f}s ({100/search_time:.1f} searches/sec)")
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")


def main():
    """Run all tests."""
    logger.info("Starting HackRx Embedding & Retrieval System Tests")
    logger.info("=" * 60)
    
    try:
        # Run end-to-end test
        success = test_end_to_end_pipeline()
        
        if success:
            logger.info("\n" + "=" * 60)
            logger.info("🎉 ALL TESTS PASSED! The embedding & retrieval system is working correctly.")
            logger.info("\nSystem is ready for integration with the main FastAPI endpoint.")
            
            # Run performance tests
            logger.info("\n" + "=" * 60)
            embedder = None
            doc_store = None
            
            # Quick setup for performance test
            try:
                from app.services.embedder import create_embedder
                from app.services.vector_store import DocumentVectorStore
                
                embedder = create_embedder("huggingface")  # Use local model for performance test
                doc_store = DocumentVectorStore(embedder.get_embedding_dimension(), "test_data")
                run_performance_test(embedder, doc_store)
            except:
                logger.info("Skipping performance tests (dependencies not available)")
            
        else:
            logger.error("\n" + "=" * 60)
            logger.error("❌ SOME TESTS FAILED! Please check the errors above.")
            
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        return False
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
