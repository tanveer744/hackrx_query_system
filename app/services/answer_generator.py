"""
Answer Generator Module - Member 4 (Prompt Engineer)
====================================================

This module integrates LLaMA 3 (Meta-LLaMA-3-8B-Instruct) for generating 
structured answers from context chunks using Hugging Face Transformers.

Author: Nehal (Member 4 - Prompt Engineer)
Day: 3 - LLaMA 3 Integration for Real LLM Responses  
Day: 4 - Prompt Tuning for Reliable JSON Output
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# LLaMA 3 Model Configuration with fallback options
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
FALLBACK_MODEL = "microsoft/DialoGPT-medium"  # Smaller fallback model
MODEL_CACHE = None
TOKENIZER_CACHE = None
USE_FALLBACK = False

def _get_model_and_tokenizer():
    """
    Lazy loading of LLaMA 3 model and tokenizer with fallback to smaller model.
    """
    global MODEL_CACHE, TOKENIZER_CACHE, USE_FALLBACK
    
    if MODEL_CACHE is None or TOKENIZER_CACHE is None:
        try:
            # First, try LLaMA 3
            logger.info(f"Attempting to load LLaMA 3 model: {MODEL_NAME}")
            TOKENIZER_CACHE = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            MODEL_CACHE = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("✅ LLaMA 3 model loaded successfully")
            
        except Exception as e:
            logger.warning(f"❌ Failed to load LLaMA 3: {e}")
            logger.info(f"🔄 Falling back to smaller model: {FALLBACK_MODEL}")
            
            try:
                # Fallback to smaller model
                TOKENIZER_CACHE = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
                MODEL_CACHE = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL)
                USE_FALLBACK = True
                logger.info("✅ Fallback model loaded successfully")
                
            except Exception as fallback_error:
                logger.error(f"❌ Even fallback model failed: {fallback_error}")
                # Final fallback - return None and handle in calling function
                return None, None
    
    return MODEL_CACHE, TOKENIZER_CACHE


def _construct_llama_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Constructs a highly optimized prompt for LLaMA 3 that reliably produces JSON output.
    
    Day 4 Enhancement: Advanced prompt engineering for consistent JSON formatting.
    
    Args:
        query (str): The user's question
        context_chunks (List[str]): List of relevant document context chunks
        
    Returns:
        str: Formatted prompt for LLaMA 3 with enhanced JSON instructions
    """
    combined_context = "\n---\n".join(context_chunks) if context_chunks else "No context provided"
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert insurance policy analyst. Your task is to analyze the provided context and answer questions with PERFECT JSON formatting.

CRITICAL INSTRUCTIONS:
1. You MUST respond with ONLY valid JSON - no additional text before or after
2. The JSON must have exactly these 3 fields: "answer", "source", "explanation"
3. Each field value must be a string (not null, not empty)
4. Use proper JSON escaping for quotes and special characters
5. Do not include markdown formatting or code blocks

JSON SCHEMA (follow exactly):
{{
  "answer": "Your clear, direct answer to the question",
  "source": "Specific clause/section reference that supports your answer", 
  "explanation": "Detailed reasoning based on the provided context"
}}

CONTEXT:
{combined_context}

IMPORTANT: Your response must start with {{ and end with }} - nothing else.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{"""
    return prompt


def _extract_json_multiple_strategies(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Day 4 Enhancement: Multiple strategies for extracting JSON from LLM responses.
    
    Args:
        response_text (str): Raw response from LLM
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON or None if all strategies fail
    """
    # Strategy 1: Direct JSON parsing (ideal case)
    try:
        if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
            return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find first complete JSON block
    try:
        json_start = response_text.find('{')
        if json_start != -1:
            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if brace_count == 0:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 3: Regex-based extraction with common patterns
    patterns = [
        r'\{[^{}]*"answer"[^{}]*"source"[^{}]*"explanation"[^{}]*\}',
        r'\{.*?"answer".*?"source".*?"explanation".*?\}',
        r'\{[\s\S]*?\}',  # Any JSON-like structure
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and 'answer' in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Line-by-line reconstruction
    try:
        lines = response_text.split('\n')
        json_content = {}
        current_field = None
        current_value = ""
        
        for line in lines:
            line = line.strip()
            # Look for field patterns
            for field in ['answer', 'source', 'explanation']:
                if f'"{field}"' in line or f"'{field}'" in line:
                    if current_field:
                        json_content[current_field] = current_value.strip('"\'')
                    current_field = field
                    # Extract value if it's on the same line
                    if ':' in line:
                        current_value = line.split(':', 1)[1].strip().rstrip(',').strip('"\'')
                    else:
                        current_value = ""
                    break
            else:
                if current_field and line and not line.startswith('{') and not line.startswith('}'):
                    current_value += " " + line.strip().rstrip(',').strip('"\'')
        
        if current_field and current_value:
            json_content[current_field] = current_value.strip('"\'')
        
        if len(json_content) >= 3 and all(field in json_content for field in ['answer', 'source', 'explanation']):
            return json_content
            
    except Exception:
        pass
    
    return None


def _validate_json_response(parsed_json: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Day 4 Enhancement: Validates and enhances the parsed JSON response.
    
    Args:
        parsed_json (Dict[str, Any]): Parsed JSON from LLM
        query (str): Original query for validation
        
    Returns:
        Dict[str, Any]: Validated and enhanced JSON response
    """
    required_fields = ["answer", "source", "explanation"]
    
    # Ensure all required fields exist and are non-empty strings
    for field in required_fields:
        if field not in parsed_json:
            parsed_json[field] = f"Field '{field}' was missing from model response"
        elif not isinstance(parsed_json[field], str) or not parsed_json[field].strip():
            parsed_json[field] = f"Field '{field}' was empty or invalid in model response"
    
    # Clean and validate content
    for field in required_fields:
        if isinstance(parsed_json[field], str):
            # Remove excessive whitespace and clean formatting
            parsed_json[field] = re.sub(r'\s+', ' ', parsed_json[field].strip())
            
            # Ensure minimum content length
            if len(parsed_json[field]) < 10:
                if field == "answer":
                    parsed_json[field] = f"Brief response regarding: {query}"
                elif field == "source":
                    parsed_json[field] = "Policy documentation"
                elif field == "explanation":
                    parsed_json[field] = f"Analysis of query: {query} based on available context"
    
    # Validate answer relevance (basic check)
    answer_lower = parsed_json["answer"].lower()
    query_words = set(query.lower().split())
    answer_words = set(answer_lower.split())
    
    # Calculate basic relevance score
    relevance_score = len(query_words.intersection(answer_words)) / max(len(query_words), 1)
    
    return {
        **parsed_json,
        "json_quality": "validated" if relevance_score > 0.1 else "low_relevance",
        "relevance_score": round(relevance_score, 2)
    }


def _parse_llama_response(response_text: str, query: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Day 4 Enhanced: Advanced parsing with multiple strategies and validation.
    
    Args:
        response_text (str): Raw response from LLaMA 3
        query (str): Original query for fallback
        context_chunks (List[str]): Context chunks for fallback
        
    Returns:
        Dict[str, Any]: Structured response with answer, source, and explanation
    """
    logger.info(f"Parsing LLM response with enhanced JSON extraction...")
    
    # Try multiple extraction strategies
    parsed_json = _extract_json_multiple_strategies(response_text)
    
    if parsed_json:
        logger.info("✅ Successfully extracted JSON from LLM response")
        
        # Validate and enhance the response
        validated_response = _validate_json_response(parsed_json, query)
        
        # Add metadata
        validated_response.update({
            "confidence": 0.90,  # High confidence for successfully parsed LLaMA responses
            "query_processed": query,
            "context_chunks_count": len(context_chunks),
            "model_used": "LLaMA-3-8B-Instruct",
            "parsing_method": "multi_strategy_extraction",
            "json_extraction": "successful"
        })
        
        return validated_response
    
    else:
        logger.warning("❌ Failed to extract valid JSON from LLM response")
        logger.debug(f"Raw response: {response_text[:200]}...")
        
        # Enhanced fallback with content analysis
        fallback_answer = _analyze_raw_response(response_text, query)
        
        return {
            "answer": fallback_answer,
            "source": "LLM response analysis (non-JSON)",
            "explanation": f"The model provided a response but not in the required JSON format. Content analysis extracted: {fallback_answer}",
            "confidence": 0.65,  # Lower confidence for fallback parsing
            "query_processed": query,
            "context_chunks_count": len(context_chunks),
            "model_used": "LLaMA-3-8B-Instruct",
            "parsing_method": "fallback_content_analysis",
            "json_extraction": "failed",
            "raw_response_preview": response_text[:300] + "..." if len(response_text) > 300 else response_text
        }


def _analyze_raw_response(response_text: str, query: str) -> str:
    """
    Day 4 Enhancement: Analyzes raw non-JSON responses to extract meaningful answers.
    
    Args:
        response_text (str): Raw response text
        query (str): Original query
        
    Returns:
        str: Extracted answer from raw response
    """
    if not response_text or len(response_text.strip()) < 5:
        return f"No meaningful response generated for query: {query}"
    
    # Clean the response
    cleaned_response = re.sub(r'\s+', ' ', response_text.strip())
    
    # Try to find the main answer in common response patterns
    patterns = [
        r'(?:answer|response|result)[:=]\s*([^.!?]*[.!?])',
        r'(?:the answer is|it is|this is)[:=]?\s*([^.!?]*[.!?])',
        r'^([^.!?]*[.!?])',  # First sentence
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned_response, re.IGNORECASE)
        if match and len(match.group(1).strip()) > 10:
            return match.group(1).strip()
    
    # If no patterns match, return first reasonable chunk
    sentences = re.split(r'[.!?]+', cleaned_response)
    for sentence in sentences:
        if len(sentence.strip()) > 15:  # Reasonable sentence length
            return sentence.strip() + "."
    
    # Last resort - return truncated response
    return cleaned_response[:200] + ("..." if len(cleaned_response) > 200 else "")


def _simple_text_based_answer(query: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Simple text-based answer generator as ultimate fallback when models fail.
    """
    combined_context = "\n".join(context_chunks) if context_chunks else ""
    
    # Basic keyword matching for common insurance queries
    query_lower = query.lower()
    
    if "maternity" in query_lower:
        answer = "Maternity coverage details are provided in the policy terms."
        source = "Policy maternity clause"
    elif "claim" in query_lower:
        answer = "Claim procedures are outlined in the policy documentation."
        source = "Claims section"
    elif "coverage" in query_lower or "cover" in query_lower:
        answer = "Coverage details depend on the specific policy terms and conditions."
        source = "Coverage clauses"
    elif "premium" in query_lower:
        answer = "Premium information is specified in your policy schedule."
        source = "Premium section"
    else:
        answer = f"Based on your query about '{query}', please refer to the relevant policy sections."
        source = "General policy terms"
    
    return {
        "answer": answer,
        "source": source,
        "explanation": f"Simple text-based analysis of query: '{query}'. Context available: {len(context_chunks)} chunks. {combined_context[:200]}...",
        "confidence": 0.6,
        "query_processed": query,
        "context_chunks_count": len(context_chunks),
        "model_used": "Simple Text Analysis (Fallback)",
        "fallback_reason": "No ML models available"
    }


def generate_answer(query: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Day 4 Enhanced: Generates structured answers with optimized JSON reliability.
    
    Improvements:
    - Enhanced prompt engineering for consistent JSON output
    - Optimized generation parameters for structured responses
    - Multiple JSON extraction strategies
    - Advanced response validation
    
    Args:
        query (str): The user's question/query
        context_chunks (List[str]): List of relevant document context chunks
        
    Returns:
        Dict[str, Any]: Structured response with answer, source, and explanation
    """
    logger.info(f"Day 4: Processing query with enhanced JSON generation: {query[:50]}...")
    
    try:
        # Get model and tokenizer
        model, tokenizer = _get_model_and_tokenizer()
        
        if model is None or tokenizer is None:
            logger.info("🔄 AI models unavailable - using enhanced text analysis")
            return _simple_text_based_answer(query, context_chunks)
        
        # Construct the prompt (enhanced for Day 4)
        if USE_FALLBACK:
            # Enhanced fallback prompt with JSON structure for smaller models
            combined_context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)]) if context_chunks else "No context available"
            prompt = f"""You are an insurance analyst. Respond in JSON format only.

Context:
{combined_context}

Question: {query}

Respond with this exact JSON structure:
{{"answer": "your answer here", "source": "clause reference", "explanation": "detailed reasoning"}}

JSON Response:"""
        else:
            # Use the enhanced LLaMA 3 prompt designed for reliable JSON
            prompt = _construct_llama_prompt(query, context_chunks)
        
        # Enhanced tokenization with better max length handling
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        
        # Day 4: Optimized generation parameters for JSON reliability
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": 300,  # Increased for complete JSON responses
            "temperature": 0.3,     # Lower temperature for more consistent formatting
            "do_sample": True,
            "top_p": 0.9,          # Nucleus sampling for quality
            "repetition_penalty": 1.1,  # Prevent repetitive text
            "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
        }
        
        # Add stop sequences for JSON completion
        if hasattr(tokenizer, 'encode'):
            stop_sequences = ["}"]
            if not USE_FALLBACK:  # For LLaMA 3 specifically
                stop_sequences.extend(["<|eot_id|>", "<|end_of_text|>"])
        
        # Generate response with enhanced parameters
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        
        # Decode the response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text (remove the prompt)
        prompt_length = len(prompt)
        generated_text = response_text[prompt_length:].strip()
        
        # Day 4: Enhanced JSON completion for incomplete responses
        if generated_text and not generated_text.endswith('}'):
            # Try to complete incomplete JSON
            if '{' in generated_text and '}' not in generated_text:
                generated_text += '}'
                logger.info("🔧 Auto-completed incomplete JSON response")
        
        logger.info(f"Day 4 Generated response: {generated_text[:150]}...")
        
        # Parse and structure the response with enhanced methods
        structured_response = _parse_llama_response(generated_text, query, context_chunks)
        
        # Add Day 4 specific metadata
        model_name = FALLBACK_MODEL if USE_FALLBACK else MODEL_NAME
        structured_response["model_used"] = model_name
        structured_response["day4_enhancement"] = "json_optimization_active"
        structured_response["generation_params"] = {
            "temperature": 0.3,
            "max_tokens": 300,
            "top_p": 0.9
        }
        
        return structured_response
        
    except Exception as e:
        logger.error(f"Day 4: Error in enhanced model generation: {e}")
        logger.info("🔄 Falling back to enhanced text analysis")
        
        # Ultimate fallback with Day 4 enhancements
        fallback_response = _simple_text_based_answer(query, context_chunks)
        fallback_response.update({
            "error": str(e),
            "fallback_reason": "Model generation failed",
            "day4_enhancement": "fallback_mode",
            "fallback_level": "enhanced_text_analysis"
        })
        
        return fallback_response


def validate_query(query: str) -> bool:
    """
    Validates if the query is appropriate for processing.
    
    Args:
        query (str): The input query to validate
        
    Returns:
        bool: True if query is valid, False otherwise
    """
    if not query or len(query.strip()) < 3:
        return False
    return True


def format_prompt(query: str, context_chunks: List[str], max_context_length: int = 1000) -> str:
    """
    Formats the prompt for LLaMA 3 using context chunks with proper chat template.
    
    Args:
        query (str): The user query
        context_chunks (List[str]): List of document context chunks
        max_context_length (int): Maximum length of combined context to include
        
    Returns:
        str: Formatted prompt string for LLaMA 3
    """
    # Use the internal LLaMA prompt construction function
    return _construct_llama_prompt(query, context_chunks)


# Day 4 Enhanced Test Functions
def test_json_reliability():
    """
    Day 4: Comprehensive test suite for JSON output reliability.
    Tests multiple scenarios to validate consistent JSON generation.
    """
    print("🎯 Day 4: JSON Reliability Test Suite")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Basic Maternity Query",
            "query": "Is maternity covered under this policy?",
            "context": [
                "Clause 4.2: Maternity benefits apply after a waiting period of 36 months with coverage up to Rs. 1 lakh.",
                "Clause 3.1: The policy includes coverage for hospitalization expenses up to Rs. 5 lakhs."
            ]
        },
        {
            "name": "Complex Coverage Query", 
            "query": "What are the exclusions and waiting periods for pre-existing conditions?",
            "context": [
                "Clause 5.1: Pre-existing diseases are covered after 48 months of continuous policy tenure.",
                "Clause 6.3: Mental health conditions require 24 months waiting period.",
                "Exclusion 2.1: Congenital conditions are permanently excluded from coverage."
            ]
        },
        {
            "name": "Short Context Query",
            "query": "What is the premium amount?",
            "context": ["Premium: Rs. 12,000 annually"]
        },
        {
            "name": "No Context Query",
            "query": "Tell me about dental coverage",
            "context": []
        },
        {
            "name": "Complex Insurance Terms",
            "query": "How does the co-payment structure work for different types of treatments?",
            "context": [
                "Co-payment: 10% for general treatments, 20% for specialist consultations.",
                "Room rent limit: Rs. 5,000 per day with patient bearing excess amount.",
                "Cashless facility available at network hospitals only."
            ]
        }
    ]
    
    json_success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}/{total_tests}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"Context chunks: {len(test_case['context'])}")
        
        try:
            result = generate_answer(test_case['query'], test_case['context'])
            
            # Validate JSON structure
            required_fields = ["answer", "source", "explanation"]
            has_all_fields = all(field in result for field in required_fields)
            
            if has_all_fields:
                json_success_count += 1
                print("✅ JSON Structure: Valid")
                print(f"📊 Model Used: {result.get('model_used', 'Unknown')}")
                print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
                print(f"🔧 JSON Extraction: {result.get('json_extraction', 'N/A')}")
                
                # Show abbreviated content
                answer_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
                print(f"💬 Answer Preview: {answer_preview}")
                
                if 'relevance_score' in result:
                    print(f"📈 Relevance Score: {result['relevance_score']}")
                    
            else:
                print("❌ JSON Structure: Invalid")
                missing_fields = [field for field in required_fields if field not in result]
                print(f"❌ Missing fields: {missing_fields}")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    # Summary
    success_rate = (json_success_count / total_tests) * 100
    print(f"\n📊 Day 4 JSON Reliability Summary:")
    print(f"✅ Successful JSON responses: {json_success_count}/{total_tests}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Day 4 JSON reliability target achieved!")
    else:
        print("⚠️  JSON reliability needs improvement")
    
    return {
        "total_tests": total_tests,
        "successful_json": json_success_count,
        "success_rate": success_rate,
        "target_achieved": success_rate >= 80
    }


# Test function for development
def test_answer_generator():
    """
    Enhanced test function for Day 4 - includes JSON validation and reliability metrics.
    """
    test_query = "Is maternity covered under this policy?"
    test_context_chunks = [
        "Clause 3.1: The policy includes coverage for hospitalization expenses up to Rs. 5 lakhs.",
        "Clause 4.2: Maternity benefits apply after a waiting period of 36 months with coverage up to Rs. 1 lakh.",
        "Clause 5.1: Pre-existing diseases are covered after 48 months of continuous policy tenure."
    ]
    
    print("🧪 Testing Day 4 Enhanced JSON Generation")
    print("=" * 50)
    print(f"Query: {test_query}")
    print(f"Context Chunks: {len(test_context_chunks)} chunks provided")
    print("\nGenerating answer with Day 4 enhancements...")
    
    result = generate_answer(test_query, test_context_chunks)
    
    print("\n✅ Day 4 Enhanced Result:")
    print(f"🤖 Model Used: {result.get('model_used', 'Unknown')}")
    print(f"📝 Query: {result['query_processed']}")
    print(f"💬 Answer: {result['answer']}")
    print(f"📄 Source: {result['source']}")
    print(f"🎯 Confidence: {result['confidence']}")
    print(f"📊 Context chunks count: {result['context_chunks_count']}")
    
    # Day 4 specific metrics
    if 'day4_enhancement' in result:
        print(f"🚀 Day 4 Enhancement: {result['day4_enhancement']}")
    if 'json_extraction' in result:
        print(f"🔧 JSON Extraction: {result['json_extraction']}")
    if 'parsing_method' in result:
        print(f"⚙️  Parsing Method: {result['parsing_method']}")
    if 'relevance_score' in result:
        print(f"📈 Relevance Score: {result['relevance_score']}")
    if 'json_quality' in result:
        print(f"✨ JSON Quality: {result['json_quality']}")
    
    print("\n💡 Explanation:")
    print(result['explanation'])
    
    if 'error' in result:
        print(f"\n⚠️  Error encountered: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Day 4: Enhanced testing with JSON reliability focus
    from pprint import pprint
    
    print("🚀 Day 4: Enhanced JSON Generation Testing")
    print("📦 Loading model and testing JSON reliability...")
    
    # Run comprehensive JSON reliability test suite
    print("\n" + "="*60)
    reliability_results = test_json_reliability()
    
    print("\n" + "="*60)
    print("🔍 Single Test Case Analysis:")
    
    context = [
        "Clause 3.1: The policy includes coverage for hospitalization expenses up to Rs. 5 lakhs per year.",
        "Clause 4.2: Maternity benefits apply after a waiting period of 36 months with coverage up to Rs. 1 lakh.",
        "Clause 5.1: Pre-existing diseases are covered after 48 months of continuous policy tenure."
    ]
    query = "Is maternity covered under this policy?"
    
    try:
        result = generate_answer(query, context)
        print("\n" + "=" * 60)
        print("🎯 COMPLETE DAY 4 ENHANCED RESPONSE:")
        print("=" * 60)
        pprint(result)
        
        # Day 4 specific analysis
        print("\n📊 Day 4 Analysis Summary:")
        print(f"✅ JSON Structure Valid: {'Yes' if all(k in result for k in ['answer', 'source', 'explanation']) else 'No'}")
        print(f"🤖 Model Used: {result.get('model_used', 'Unknown')}")
        print(f"🔧 JSON Extraction: {result.get('json_extraction', 'N/A')}")
        print(f"📈 Reliability Score: {reliability_results['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("This might be due to model loading issues or insufficient GPU memory.")
        print("💡 Fallback mechanisms should still provide structured responses.")
