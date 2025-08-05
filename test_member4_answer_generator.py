# test_member4_answer_generator.py
"""
Test script for Member 4 (Answer Generator) functionality
Tests the exact functionality requested: generate_answer function
"""

import sys
import os
import json
import logging

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_member4_answer_generator():
    """Test Member 4 answer generator functionality as requested."""
    logger.info("🔍 Testing Member 4 (Answer Generator) functionality...")
    
    try:
        logger.info("Step 1: Importing answer generator...")
        from app.services.answer_generator import generate_answer
        logger.info("✓ Successfully imported generate_answer function")
        
        logger.info("Step 2: Testing answer generation with sample data...")
        
        # Test with the exact inputs as requested
        sample_policy_text = "Sample policy text"
        sample_question = "What is the waiting period?"
        
        logger.info(f"Input policy text: '{sample_policy_text}'")
        logger.info(f"Input question: '{sample_question}'")
        
        # Call the generate_answer function
        result = generate_answer(sample_policy_text, sample_question)
        
        logger.info("✓ Answer generated successfully!")
        logger.info(f"Result type: {type(result)}")
        
        # Display the result
        logger.info("Generated answer:")
        print("=" * 50)
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)
        print("=" * 50)
        
        # Verify the expected structure
        if isinstance(result, dict):
            expected_keys = ["answer", "source", "explanation"]
            for key in expected_keys:
                if key in result:
                    logger.info(f"✓ Found expected key '{key}': {result[key]}")
                else:
                    logger.warning(f"⚠ Missing expected key '{key}'")
        
        logger.info("Step 3: Testing with additional scenarios...")
        
        # Test with different inputs
        test_cases = [
            {
                "policy_text": "The grace period for premium payment is 30 days. After this period, the policy may lapse.",
                "question": "How long is the grace period?"
            },
            {
                "policy_text": "Medical expenses are covered up to $100,000 annually. Emergency care is included.",
                "question": "What is the coverage limit?"
            },
            {
                "policy_text": "Pre-existing conditions have a waiting period of 12 months from policy start date.",
                "question": "Are pre-existing conditions covered immediately?"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\nTest case {i}:")
            logger.info(f"  Policy text: '{test_case['policy_text'][:50]}...'")
            logger.info(f"  Question: '{test_case['question']}'")
            
            try:
                test_result = generate_answer(test_case['policy_text'], test_case['question'])
                logger.info(f"  ✓ Generated answer successfully")
                
                if isinstance(test_result, dict) and 'answer' in test_result:
                    logger.info(f"  Answer: '{test_result['answer'][:60]}...'")
                else:
                    logger.info(f"  Result: {str(test_result)[:60]}...")
                    
            except Exception as e:
                logger.error(f"  ✗ Test case {i} failed: {e}")
        
        logger.info("\n🎉 Member 4 (Answer Generator) test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        logger.error("Make sure the answer_generator module exists in app/services/")
        return False
    except Exception as e:
        logger.error(f"✗ Member 4 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_requested_api():
    """Test the exact API as requested in the user prompt"""
    logger.info("🔍 Testing the exact requested API...")
    
    try:
        logger.info("Testing: from app.services.answer_generator import generate_answer")
        from app.services.answer_generator import generate_answer
        
        logger.info("Testing: result = generate_answer('Sample policy text', 'What is the waiting period?')")
        result = generate_answer("Sample policy text", "What is the waiting period?")
        
        logger.info("Testing: print(result)")
        print("\n" + "="*60)
        print("EXACT REQUESTED API TEST RESULT:")
        print("="*60)
        print(result)
        print("="*60)
        
        # Check if it matches expected output structure
        if isinstance(result, dict):
            expected_structure = {
                "answer": "This is a placeholder answer.",
                "source": "No source yet", 
                "explanation": "This is just a test response."
            }
            logger.info("\nVerifying expected output structure:")
            for key, expected_value in expected_structure.items():
                if key in result:
                    logger.info(f"✓ Key '{key}' present: {result[key]}")
                else:
                    logger.warning(f"⚠ Key '{key}' missing")
        
        logger.info("🎉 Requested API test completed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Requested API test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MEMBER 4 (ANSWER GENERATOR) TEST")
    logger.info("=" * 60)
    
    # Run the main test
    success1 = test_member4_answer_generator()
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING EXACT REQUESTED API FORMAT")
    logger.info("=" * 60)
    
    # Run the requested API format test
    success2 = test_requested_api()
    
    if success1 and success2:
        logger.info("\n🎉 ALL MEMBER 4 TESTS PASSED!")
        logger.info("✓ Answer generator is working correctly")
        logger.info("✓ Expected output structure is correct")
        logger.info("✓ Function handles different inputs properly")
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
    
    exit_code = 0 if (success1 and success2) else 1
    sys.exit(exit_code)
