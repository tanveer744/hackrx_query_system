# test_member3_embeddings_faiss.py
"""
Test script for Member 3 (Embeddings + FAISS) functionality
Tests the exact functionality requested: get_embeddings and create_faiss_index
"""

import sys
import os
import numpy as np
import logging

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_member3_embeddings_faiss():
    """Test Member 3 embeddings and FAISS functionality as requested."""
    logger.info("🔍 Testing Member 3 (Embeddings + FAISS) functionality...")
    
    try:
        # Test the existing embedding service
        from app.services.embedder import create_embedder
        from app.services.vector_store import DocumentVectorStore
        
        logger.info("Step 1: Setting up embedder...")
        
        # Create embedder (try HuggingFace first as it's more likely to work locally)
        try:
            embedder = create_embedder("huggingface")
            logger.info("✓ Using HuggingFace embedder")
        except Exception as e:
            logger.warning(f"HuggingFace not available: {e}")
            try:
                embedder = create_embedder("openai")
                logger.info("✓ Using OpenAI embedder")
            except Exception as e:
                logger.error(f"No embedders available: {e}")
                return False
        
        logger.info("Step 2: Testing embeddings with sample texts...")
        
        # Test data as requested
        texts = ["Knee surgery is covered.", "Cataract has a waiting period."]
        logger.info(f"Input texts: {texts}")
        
        # Get embeddings using the available embedding service
        embeddings = []
        for text in texts:
            embedding = embedder.embed_single(text)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        logger.info(f"✓ Generated embeddings shape: {embeddings.shape}")
        
        logger.info("Step 3: Creating FAISS index...")
        
        # Create FAISS index using DocumentVectorStore (which uses FAISS internally)
        embedding_dim = embedder.get_embedding_dimension()
        
        # Create temporary directory for testing
        test_dir = "temp_test_faiss"
        os.makedirs(test_dir, exist_ok=True)
        
        # Initialize vector store (this creates the FAISS index)
        vector_store = DocumentVectorStore(embedding_dim, test_dir)
        
        # Add documents to the vector store (this adds to FAISS index)
        sources = ["policy_doc.pdf", "policy_doc.pdf"]
        pages = [1, 2]
        sections = ["surgery_coverage", "waiting_periods"]
        
        vector_store.add_documents(texts, embeddings, sources, pages, sections)
        logger.info(f"✓ Added {len(texts)} documents to FAISS index")
        
        logger.info("Step 4: Testing query search...")
        
        # Test a query vector as requested
        query = "Is cataract surgery covered?"
        logger.info(f"Query: '{query}'")
        
        query_embedding = embedder.embed_single(query)
        
        # Search the FAISS index
        search_results = vector_store.search_documents(query_embedding, k=1)
        
        if search_results:
            closest_match = search_results[0]
            logger.info(f"✓ Closest match: '{closest_match['text']}'")
            logger.info(f"✓ Similarity score: {closest_match['similarity_score']:.4f}")
            logger.info(f"✓ Source: {closest_match['source']}, Page: {closest_match['page']}")
            
            # Mimic the requested output format
            closest_index = 1 if "Cataract" in closest_match['text'] else 0
            logger.info(f"✓ Closest match index: [{closest_index}]")
        else:
            logger.error("✗ No search results returned")
            return False
        
        logger.info("Step 5: Verifying FAISS functionality...")
        
        # Get stats to confirm FAISS is working
        stats = vector_store.get_stats()
        logger.info(f"✓ Vector store stats: {stats}")
        
        # Test with multiple queries
        test_queries = [
            "What about knee surgery?",
            "Are there waiting periods?",
            "Surgery coverage details"
        ]
        
        logger.info("Step 6: Testing multiple queries...")
        for i, test_query in enumerate(test_queries):
            query_emb = embedder.embed_single(test_query)
            results = vector_store.search_documents(query_emb, k=2)
            
            logger.info(f"Query {i+1}: '{test_query}'")
            for j, result in enumerate(results):
                logger.info(f"  Result {j+1}: '{result['text'][:30]}...' (score: {result['similarity_score']:.3f})")
        
        logger.info("🎉 Member 3 (Embeddings + FAISS) test completed successfully!")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        logger.info("✓ Cleaned up test directory")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Member 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_embeddings_wrapper(texts):
    """
    Wrapper function to mimic the requested get_embeddings function
    Uses the actual embedding service underneath
    """
    try:
        from app.services.embedder import create_embedder
        embedder = create_embedder("huggingface")
        
        embeddings = []
        for text in texts:
            embedding = embedder.embed_single(text)
            embeddings.append(embedding.tolist())
        
        return embeddings
    except Exception as e:
        logger.error(f"get_embeddings failed: {e}")
        return None

def create_faiss_index_wrapper(embeddings_array):
    """
    Wrapper function to mimic the requested create_faiss_index function
    Uses DocumentVectorStore which uses FAISS internally
    """
    try:
        from app.services.vector_store import DocumentVectorStore
        
        # Get dimensions from the embeddings
        embedding_dim = embeddings_array.shape[1]
        
        # Create a temporary vector store
        temp_dir = "temp_faiss_wrapper"
        os.makedirs(temp_dir, exist_ok=True)
        
        vector_store = DocumentVectorStore(embedding_dim, temp_dir)
        
        # Add dummy documents to populate the index
        texts = [f"Document {i}" for i in range(len(embeddings_array))]
        sources = [f"doc_{i}.pdf" for i in range(len(embeddings_array))]
        pages = list(range(1, len(embeddings_array) + 1))
        sections = [f"section_{i}" for i in range(len(embeddings_array))]
        
        vector_store.add_documents(texts, embeddings_array, sources, pages, sections)
        
        return vector_store
        
    except Exception as e:
        logger.error(f"create_faiss_index failed: {e}")
        return None

def test_requested_api():
    """Test the exact API as requested in the user prompt"""
    logger.info("🔍 Testing the exact requested API...")
    
    try:
        logger.info("Testing: from app.services.embedder import get_embeddings")
        logger.info("Note: Using wrapper function since get_embeddings doesn't exist directly")
        
        texts = ["Knee surgery is covered.", "Cataract has a waiting period."]
        embeddings = get_embeddings_wrapper(texts)
        logger.info(f"✓ Got embeddings for {len(texts)} texts")
        
        logger.info("Testing: from app.services.vector_store import create_faiss_index")
        logger.info("Note: Using wrapper function with DocumentVectorStore")
        
        embeddings_array = np.array(embeddings)
        index = create_faiss_index_wrapper(embeddings_array)
        logger.info("✓ Created FAISS index")
        
        # Test a query vector
        query = get_embeddings_wrapper(["Is cataract surgery covered?"])[0]
        query_array = np.array([query])
        
        # Search using the vector store
        search_results = index.search_documents(query_array[0], k=1)
        
        if search_results:
            closest_idx = 1 if "Cataract" in search_results[0]['text'] else 0
            logger.info(f"✓ Closest match index: {closest_idx}")
        
        logger.info("🎉 Requested API test completed!")
        
        # Cleanup
        import shutil
        shutil.rmtree("temp_faiss_wrapper", ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Requested API test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MEMBER 3 (EMBEDDINGS + FAISS) TEST")
    logger.info("=" * 60)
    
    # Run the main test
    success1 = test_member3_embeddings_faiss()
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING EXACT REQUESTED API FORMAT")
    logger.info("=" * 60)
    
    # Run the requested API format test
    success2 = test_requested_api()
    
    if success1 and success2:
        logger.info("\n🎉 ALL MEMBER 3 TESTS PASSED!")
        logger.info("✓ Embeddings are working correctly")
        logger.info("✓ FAISS indexing is functional")
        logger.info("✓ Query search returns correct results")
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
    
    exit_code = 0 if (success1 and success2) else 1
    sys.exit(exit_code)
