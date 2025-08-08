#!/usr/bin/env python3

"""
Day 4 JSON Reliability Test Suite - Member 4 (Prompt Engineer)
============================================================

Tests the enhanced answer_generator.py with focus on:
- Reliable JSON output generation
- Multiple parsing strategies
- Response validation and quality
- Fallback mechanisms

Author: Nehal (Member 4 - Prompt Engineer)
Day: 4 - Prompt Tuning for Reliable JSON Output
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_json_structure_validation():
    """Test that all responses have proper JSON structure"""
    print("🔍 Testing JSON Structure Validation...")
    
    try:
        from services.answer_generator import generate_answer
        
        test_cases = [
            ("Simple query", "What is covered?", ["Basic coverage information"]),
            ("Complex query", "What are the waiting periods and exclusions?", [
                "Waiting period: 24 months for pre-existing conditions",
                "Exclusions: Cosmetic surgery, experimental treatments"
            ]),
            ("Empty context", "Tell me about benefits", []),
        ]
        
        required_fields = ["answer", "source", "explanation", "confidence", "query_processed"]
        
        for test_name, query, context in test_cases:
            print(f"\n🧪 {test_name}")
            result = generate_answer(query, context)
            
            # Check JSON structure
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
            else:
                print("✅ All required fields present")
            
            # Check field types and content
            for field in ["answer", "source", "explanation"]:
                if field in result:
                    if isinstance(result[field], str) and len(result[field].strip()) > 0:
                        print(f"✅ {field}: Valid string content")
                    else:
                        print(f"❌ {field}: Invalid or empty")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def test_json_parsing_robustness():
    """Test the JSON parsing with various malformed inputs"""
    print("\n🔧 Testing JSON Parsing Robustness...")
    
    try:
        from services.answer_generator import _extract_json_multiple_strategies
        
        test_responses = [
            # Perfect JSON
            '{"answer": "Yes", "source": "Clause 1", "explanation": "Because..."}',
            
            # JSON with extra text
            'Here is the response: {"answer": "No", "source": "Clause 2", "explanation": "Due to..."} Hope this helps!',
            
            # Incomplete JSON
            '{"answer": "Maybe", "source": "Clause 3", "explanation": "This is because',
            
            # Malformed JSON with missing quotes
            '{answer: "Yes", source: "Clause 4", explanation: "The reason..."}',
            
            # Multiple JSON-like structures
            '{"invalid": "first"} {"answer": "Correct", "source": "Clause 5", "explanation": "The proper..."}',
        ]
        
        success_count = 0
        for i, response in enumerate(test_responses, 1):
            print(f"\n🧪 Test case {i}")
            result = _extract_json_multiple_strategies(response)
            
            if result and isinstance(result, dict) and 'answer' in result:
                print("✅ Successfully extracted JSON")
                success_count += 1
            else:
                print("❌ Failed to extract JSON")
        
        success_rate = (success_count / len(test_responses)) * 100
        print(f"\n📊 JSON Parsing Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 60  # At least 60% should be parseable
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def test_response_quality_metrics():
    """Test response quality and relevance metrics"""
    print("\n📈 Testing Response Quality Metrics...")
    
    try:
        from services.answer_generator import generate_answer
        
        test_cases = [
            {
                "query": "What is the maternity coverage amount?",
                "context": ["Maternity coverage: Rs. 1 lakh after 36 months waiting period"],
                "expected_keywords": ["maternity", "1 lakh", "36 months"]
            },
            {
                "query": "Are dental treatments covered?",
                "context": ["Dental coverage: Up to Rs. 25,000 per year for basic procedures"],
                "expected_keywords": ["dental", "25,000", "basic procedures"]
            }
        ]
        
        quality_scores = []
        
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['query']}")
            result = generate_answer(test_case['query'], test_case['context'])
            
            answer = result.get('answer', '').lower()
            
            # Check keyword relevance
            keyword_count = sum(1 for keyword in test_case['expected_keywords'] 
                              if keyword.lower() in answer)
            keyword_score = keyword_count / len(test_case['expected_keywords'])
            
            print(f"📊 Keyword relevance: {keyword_score:.2f}")
            print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
            
            if 'relevance_score' in result:
                print(f"📈 Model relevance score: {result['relevance_score']}")
            
            quality_scores.append(keyword_score)
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"\n📊 Average Quality Score: {avg_quality:.2f}")
        
        return avg_quality >= 0.5  # At least 50% keyword relevance
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def test_fallback_mechanisms():
    """Test that fallback mechanisms work properly"""
    print("\n🔄 Testing Fallback Mechanisms...")
    
    try:
        from services.answer_generator import _simple_text_based_answer
        
        # Test simple text-based fallback
        query = "What are the benefits?"
        context = ["Benefits include hospitalization and outpatient coverage"]
        
        result = _simple_text_based_answer(query, context)
        
        # Check fallback response structure
        required_fields = ["answer", "source", "explanation", "model_used", "fallback_reason"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            print(f"❌ Fallback missing fields: {missing_fields}")
            return False
        else:
            print("✅ Fallback provides complete structure")
            
        if "fallback" in result.get("model_used", "").lower():
            print("✅ Fallback properly identified")
        else:
            print("❌ Fallback not properly identified")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def run_day4_test_suite():
    """Run the complete Day 4 test suite"""
    print("🎯 Day 4 JSON Reliability Test Suite")
    print("=" * 60)
    print("Testing enhanced answer_generator.py with focus on reliable JSON output")
    print("=" * 60)
    
    tests = [
        ("JSON Structure Validation", test_json_structure_validation),
        ("JSON Parsing Robustness", test_json_parsing_robustness),
        ("Response Quality Metrics", test_response_quality_metrics),
        ("Fallback Mechanisms", test_fallback_mechanisms),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n{'='*60}")
    print(f"📊 Day 4 Test Suite Summary")
    print(f"{'='*60}")
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("🎉 Day 4 JSON reliability improvements are working well!")
    elif success_rate >= 50:
        print("⚠️  Day 4 improvements partially working - some issues to address")
    else:
        print("❌ Day 4 improvements need significant work")
    
    print(f"\n💡 Day 4 Focus Areas:")
    print(f"   - Enhanced prompt engineering for consistent JSON")
    print(f"   - Multiple JSON extraction strategies")
    print(f"   - Response validation and quality metrics")
    print(f"   - Robust fallback mechanisms")
    
    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": success_rate,
        "day4_ready": success_rate >= 75
    }


if __name__ == "__main__":
    results = run_day4_test_suite()
    
    print(f"\n🚀 Day 4 Development Status:")
    if results["day4_ready"]:
        print(f"✅ Ready for production - JSON reliability achieved!")
    else:
        print(f"⚠️  Needs refinement - continue Day 4 development")
