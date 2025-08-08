#!/usr/bin/env python3
"""
Pipeline Test Script - HackRx Query System
Tests the complete enhanced pipeline with all components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pipeline_integration():
    """Test complete pipeline integration with all services."""
    print("🚀 Testing HackRx Query System Pipeline Integration")
    print("=" * 60)
    
    try:
        # Test 1: Import all services
        print("📦 Testing service imports...")
        from app.services.document_loader import DocumentLoader
        from app.services.chunker import DocumentChunker
        from app.services.embedder import get_embeddings
        from app.services.vector_store import FAISSIndex
        from app.services.answer_generator import generate_answer
        
        try:
            from app.services.reranker import rerank_chunks
            reranker_available = True
        except ImportError:
            reranker_available = False
        
        print("✅ All core services imported successfully")
        print(f"🔧 Reranker available: {reranker_available}")
        
        # Test 2: Import main application
        print("\n📱 Testing main application import...")
        from app.main import app, verify_token, ENABLE_RERANKING, RERANKER_AVAILABLE
        print("✅ Main application imported successfully")
        print(f"🎯 Reranking enabled: {ENABLE_RERANKING}")
        print(f"🔧 Reranker available in main: {RERANKER_AVAILABLE}")
        
        # Test 3: Test schemas
        print("\n📋 Testing schemas...")
        from app.schemas.request import QueryRequest
        from app.schemas.response import QueryResponse, AnswerItem
        
        # Create test request
        test_request = QueryRequest(
            documents="Test document content",
            questions=["What is this document about?"]
        )
        print("✅ Schemas working correctly")
        
        # Test 4: Test basic service functionality
        print("\n⚙️ Testing basic service functionality...")
        
        # Test embeddings
        test_text = ["This is a test sentence"]
        embeddings = get_embeddings(test_text)
        print(f"✅ Embeddings service: Generated {len(embeddings[0])} dimensional vectors")
        
        # Test chunker
        chunker = DocumentChunker()
        test_blocks = [{"doc_id": "test", "page": 1, "text": "This is test content for chunking."}]
        chunks = chunker.chunk_text(test_blocks)
        print(f"✅ Chunker service: Generated {len(chunks)} chunks")
        
        # Test answer generator
        result = generate_answer("What is this about?", ["This is test content"])
        required_fields = ["answer", "source", "explanation"]
        has_fields = all(field in result for field in required_fields)
        print(f"✅ Answer generator: {'Working' if has_fields else 'Missing fields'}")
        
        print("\n📊 Pipeline Integration Summary:")
        print(f"✅ Core services: Working")
        print(f"✅ Main application: Working")
        print(f"✅ Schemas: Working") 
        print(f"✅ Authentication: {'Configured' if os.getenv('HACKRX_API_KEY') else 'Not configured'}")
        print(f"🔧 Reranking: {'Available' if reranker_available else 'Not available'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_integration()
    if success:
        print("\n🎉 Pipeline integration test PASSED!")
    else:
        print("\n💥 Pipeline integration test FAILED!")
        sys.exit(1)
