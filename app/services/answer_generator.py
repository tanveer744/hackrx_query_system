"""
Answer Generator Module - Member 4 (Prompt Engineer)
====================================================

This module integrates LLaMA 3 (Meta-LLaMA-3-8B-Instruct) for generating 
structured answers from context chunks using Hugging Face Transformers.

Author: Nehal (Member 4 - Prompt Engineer)
Day: 3 - LLaMA 3 Integration for Real LLM Responses
"""

import json
import logging
from typing import Dict, Any, Optional, List
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
    Constructs a structured prompt for LLaMA 3 that includes context and user question.
    
    Args:
        query (str): The user's question
        context_chunks (List[str]): List of relevant document context chunks
        
    Returns:
        str: Formatted prompt for LLaMA 3
    """
    combined_context = "\n---\n".join(context_chunks) if context_chunks else "No context provided"
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert insurance policy analyst. Based on the provided context, answer the user's question accurately and provide structured information.

Your response must be in JSON format with exactly these fields:
- "answer": A clear, direct answer to the question
- "source": The specific clause or section that supports your answer
- "explanation": A detailed explanation of how you reached this conclusion based on the context

Context:
{combined_context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def _parse_llama_response(response_text: str, query: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Parses LLaMA 3 response and extracts structured JSON data.
    
    Args:
        response_text (str): Raw response from LLaMA 3
        query (str): Original query for fallback
        context_chunks (List[str]): Context chunks for fallback
        
    Returns:
        Dict[str, Any]: Structured response with answer, source, and explanation
    """
    try:
        # Try to extract JSON from the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            parsed_response = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["answer", "source", "explanation"]
            if all(field in parsed_response for field in required_fields):
                # Add metadata
                parsed_response.update({
                    "confidence": 0.90,  # High confidence for LLaMA responses
                    "query_processed": query,
                    "context_chunks_count": len(context_chunks),
                    "model_used": "LLaMA-3-8B-Instruct"
                })
                return parsed_response
    
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse LLaMA response as JSON: {e}")
    
    # Fallback response if parsing fails
    return {
        "answer": f"Based on the provided context, here's my response to: {query}",
        "source": "Multiple clauses from the policy document",
        "explanation": f"LLaMA 3 generated response: {response_text[:500]}...",
        "confidence": 0.75,  # Lower confidence for fallback
        "query_processed": query,
        "context_chunks_count": len(context_chunks),
        "model_used": "LLaMA-3-8B-Instruct",
        "parsing_note": "Response was generated but could not be parsed as structured JSON"
    }


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
    Generates structured answers using LLaMA 3 (Meta-LLaMA-3-8B-Instruct) model with fallbacks.
    
    Args:
        query (str): The user's question/query
        context_chunks (List[str]): List of relevant document context chunks
        
    Returns:
        Dict[str, Any]: Structured response with answer, source, and explanation
    """
    logger.info(f"Processing query with LLaMA 3: {query[:50]}...")
    
    try:
        # Get model and tokenizer
        model, tokenizer = _get_model_and_tokenizer()
        
        if model is None or tokenizer is None:
            logger.warning("🔄 Models unavailable, using simple text analysis")
            return _simple_text_based_answer(query, context_chunks)
        
        # Construct the prompt (adjust for fallback model if needed)
        if USE_FALLBACK:
            # Simpler but more structured prompt for smaller models
            combined_context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)]) if context_chunks else "No context available"
            prompt = f"""You are an insurance policy analyst. Answer the question based on the context.

Context:
{combined_context}

Question: {query}

Answer: Based on the context provided,"""
        else:
            # Full LLaMA 3 prompt
            prompt = _construct_llama_prompt(query, context_chunks)
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            )
        
        # Decode the response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text (remove the prompt)
        prompt_length = len(prompt)
        generated_text = response_text[prompt_length:].strip()
        
        logger.info(f"Generated response: {generated_text[:100]}...")
        
        # Parse and structure the response
        model_name = FALLBACK_MODEL if USE_FALLBACK else MODEL_NAME
        structured_response = _parse_llama_response(generated_text, query, context_chunks)
        structured_response["model_used"] = model_name
        
        return structured_response
        
    except Exception as e:
        logger.error(f"Error in model generation: {e}")
        logger.info("🔄 Falling back to simple text analysis")
        
        # Ultimate fallback to simple text analysis
        fallback_response = _simple_text_based_answer(query, context_chunks)
        fallback_response["error"] = str(e)
        fallback_response["fallback_reason"] = "Model generation failed"
        
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


# Test function for development
def test_answer_generator():
    """
    Test function to verify LLaMA 3 integration works with context chunks.
    """
    test_query = "Is maternity covered under this policy?"
    test_context_chunks = [
        "Clause 3.1: The policy includes coverage for hospitalization expenses up to Rs. 5 lakhs.",
        "Clause 4.2: Maternity benefits apply after a waiting period of 36 months with coverage up to Rs. 1 lakh.",
        "Clause 5.1: Pre-existing diseases are covered after 48 months of continuous policy tenure."
    ]
    
    print("🧪 Testing LLaMA 3 Integration - Day 3")
    print("=" * 50)
    print(f"Query: {test_query}")
    print(f"Context Chunks: {len(test_context_chunks)} chunks provided")
    print("\nGenerating answer with LLaMA 3...")
    
    result = generate_answer(test_query, test_context_chunks)
    
    print("\n✅ LLaMA 3 Test Result:")
    print(f"🤖 Model Used: {result.get('model_used', 'Unknown')}")
    print(f"📝 Query: {result['query_processed']}")
    print(f"💬 Answer: {result['answer']}")
    print(f"📄 Source: {result['source']}")
    print(f"🎯 Confidence: {result['confidence']}")
    print(f"📊 Context chunks count: {result['context_chunks_count']}")
    print("\n💡 Explanation:")
    print(result['explanation'])
    
    if 'error' in result:
        print(f"\n⚠️  Error encountered: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Run test when file is executed directly
    from pprint import pprint
    
    print("🚀 Starting LLaMA 3 Integration Test")
    print("📦 Loading model and generating response...")
    
    context = [
        "Clause 3.1: The policy includes coverage for hospitalization expenses up to Rs. 5 lakhs per year.",
        "Clause 4.2: Maternity benefits apply after a waiting period of 36 months with coverage up to Rs. 1 lakh.",
        "Clause 5.1: Pre-existing diseases are covered after 48 months of continuous policy tenure."
    ]
    query = "Is maternity covered under this policy?"
    
    try:
        result = generate_answer(query, context)
        print("\n" + "=" * 60)
        print("🎯 COMPLETE LLAMA 3 RESPONSE:")
        print("=" * 60)
        pprint(result)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("This might be due to model loading issues or insufficient GPU memory.")
