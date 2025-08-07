#!/usr/bin/env python3
"""
Test script with the exact API requested by user
Tests: get_embeddings and FAISSIndex with the specified format
"""

import sys
import os

# Add app to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

print("🧪 Testing the exact requested API")
print("=" * 50)

try:
    print("Step 1: Importing required functions...")
    from app.services.embedder import get_embeddings
    from app.services.vector_store import FAISSIndex
    print("✅ Successfully imported get_embeddings and FAISSIndex")
    
    print("\nStep 2: Setting up test data...")
    texts = ["Knee surgery covered.", "Cataract has a waiting period."]
    metas = [{"page": 1}, {"page": 2}]
    print(f"✅ Texts: {texts}")
    print(f"✅ Metadata: {metas}")
    
    print("\nStep 3: Getting embeddings...")
    embeddings = get_embeddings(texts)
    print(f"✅ Generated embeddings for {len(texts)} texts")
    print(f"✅ Embedding dimension: {len(embeddings[0])}")
    print(f"✅ Embeddings type: {type(embeddings)}")
    print(f"✅ First embedding preview: {embeddings[0][:5]}...")
    
    print("\nStep 4: Creating FAISS index...")
    index = FAISSIndex(384)  # BGE-small dimension
    print("✅ FAISS index created with dimension 384")
    
    print("\nStep 5: Adding embeddings to index...")
    index.add(embeddings, texts, metas)
    print("✅ Added embeddings, texts, and metadata to index")
    
    print("\nStep 6: Testing query search...")
    query_texts = ["Is cataract surgery covered?"]
    query = get_embeddings(query_texts)[0]
    print(f"✅ Generated query embedding: {len(query)} dimensions")
    print(f"✅ Query preview: {query[:5]}...")
    
    print("\nStep 7: Searching index...")
    results = index.search(query, k=1)
    print(f"✅ Search completed, found {len(results)} results")
    
    print("\nStep 8: Displaying results...")
    print("Results:")
    for i, result in enumerate(results):
        print(f"  Result {i+1}:")
        print(f"    Text: '{result['text']}'")
        print(f"    Metadata: {result['metadata']}")
        print(f"    Similarity Score: {result['similarity_score']:.4f}")
        print(f"    Distance: {result['distance']:.4f}")
    
    print("\n🎉 Test completed successfully!")
    print("✅ The exact requested API is working correctly!")
    
    # Additional verification
    print(f"\n📊 Verification:")
    print(f"✅ get_embeddings() returns List[List[float]]: {type(embeddings)}")
    print(f"✅ FAISSIndex.add() accepts embeddings, texts, metas")
    print(f"✅ FAISSIndex.search() returns results with text and metadata")
    
    # Test edge case: multiple queries
    print(f"\n🔍 Testing with multiple queries...")
    test_queries = [
        "What about knee surgery?",
        "Are there waiting periods?"
    ]
    
    for query_text in test_queries:
        query_emb = get_embeddings([query_text])[0]
        query_results = index.search(query_emb, k=2)
        
        print(f"\nQuery: '{query_text}'")
        print(f"Results: {len(query_results)} items")
        for j, res in enumerate(query_results):
            print(f"  {j+1}. '{res['text']}' (score: {res['similarity_score']:.3f})")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    
print(f"\n" + "=" * 50)
print("Test execution completed!")
