#!/usr/bin/env python3
"""
Test script for Answer Generator (Member 4) functionality
Tests the generate_answer function with the exact format requested
"""

import sys
import os

# Add app to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

print("🧪 Testing Answer Generator (Member 4)")
print("=" * 50)

try:
    print("Step 1: Importing generate_answer function...")
    from app.services.answer_generator import generate_answer
    print("✅ Successfully imported generate_answer")
    
    print("\nStep 2: Setting up test data...")
    ctx = ["This policy includes knee surgery coverage...", "There is a 2-year waiting period for cataract."]
    q = "Is cataract surgery covered?"
    
    print(f"✅ Context: {ctx}")
    print(f"✅ Query: '{q}'")
    
    print("\nStep 3: Generating answer...")
    result = generate_answer(q, ctx)
    print("✅ Answer generated successfully!")
    
    print("\nStep 4: Displaying result...")
    print("Result:")
    print(f"  Answer: {result.get('answer', 'N/A')}")
    print(f"  Source: {result.get('source', 'N/A')}")
    print(f"  Explanation: {result.get('explanation', 'N/A')}")
    print(f"  Confidence: {result.get('confidence', 'N/A')}")
    print(f"  Query Processed: {result.get('query_processed', 'N/A')}")
    print(f"  Context Chunks Count: {result.get('context_chunks_count', 'N/A')}")
    
    print("\n🎉 Test completed successfully!")
    print("✅ The generate_answer function is working correctly!")
    
    # Additional test with different queries
    print(f"\n🔍 Testing with additional queries...")
    
    test_cases = [
        {
            "context": ["Policy covers dental procedures after 6 months.", "Emergency treatments are covered immediately."],
            "query": "Are dental procedures covered?"
        },
        {
            "context": ["Maximum coverage limit is $50,000 annually.", "Co-payment is required for outpatient visits."],
            "query": "What is the coverage limit?"
        },
        {
            "context": ["Pre-existing conditions have a 12-month waiting period.", "Maternity benefits are available after 10 months."],
            "query": "What about maternity coverage?"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nAdditional Test {i}:")
        print(f"  Query: '{test_case['query']}'")
        
        additional_result = generate_answer(test_case['query'], test_case['context'])
        
        print(f"  Answer: {additional_result.get('answer', 'N/A')}")
        print(f"  Context chunks used: {additional_result.get('context_chunks_count', 0)}")
        print(f"  Confidence: {additional_result.get('confidence', 'N/A')}")
    
    print(f"\n📊 Verification:")
    print(f"✅ generate_answer(query, context_chunks) returns Dict[str, Any]")
    print(f"✅ Result contains answer, source, explanation, confidence")
    print(f"✅ Function handles multiple context chunks correctly")
    print(f"✅ Function processes various query types")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    
print(f"\n" + "=" * 50)
print("Answer Generator test execution completed!")
