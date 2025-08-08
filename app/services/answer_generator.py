"""
Answer Generator Module - Member 4 (Prompt Engineer)
====================================================

This module uses the Gemini API to generate structured answers
from context chunks via HTTP API calls.

Author: Nehal (Member 4 - Prompt Engineer)
Day: Updated for Gemini API integration
"""

import json
import logging
import os
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDsUnQFJuG9goHrhrLNBGoW3ew-nOohcXk")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"


def validate_query(query: str) -> bool:
    """
    Validates if the query is appropriate for processing.

    Args:
        query (str): The input query to validate

    Returns:
        bool: True if query is valid, False otherwise
    """
    return bool(query and len(query.strip()) >= 3)


def format_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Formats the prompt by combining context chunks and the question.
    
    Args:
        query (str): The user query
        context_chunks (List[str]): List of document context chunks

    Returns:
        str: Formatted prompt string
    """
    combined_context = "\n---\n".join(context_chunks) if context_chunks else "No context provided"
    prompt = f"""Based on the following context, please answer the question and provide your response in valid JSON format only.

Context: {combined_context}

Question: {query}

Please respond with only a JSON object with these exact fields:
{{
  "answer": "your detailed answer here",
  "source": "reference to relevant clause or section",
  "explanation": "reasoning based on context"
}}

Do not include any text before or after the JSON object."""
    return prompt.strip()


def _simple_text_based_answer(query: str, context_chunks: List[str], error: str = "") -> Dict[str, Any]:
    """
    Enhanced fallback answer generator when the API call fails or no model available.
    Uses context analysis to provide better answers.

    Args:
        query (str): User's query
        context_chunks (List[str]): Contextual document chunks
        error (str): Optional error message to include

    Returns:
        Dict[str, Any]: Rule-based structured answer with context analysis.
    """
    combined_context = "\n".join(context_chunks) if context_chunks else ""
    query_lower = query.lower()
    context_lower = combined_context.lower()

    # Enhanced context-aware fallback logic
    if "maternity" in query_lower or "pregnancy" in query_lower:
        if "maternity" in context_lower:
            answer = "Based on the policy context, maternity coverage appears to be mentioned. Please review the specific terms and waiting periods."
            source = "Maternity coverage section in policy"
        else:
            answer = "Maternity coverage details should be checked in your policy terms and conditions."
            source = "Policy maternity clause"
    
    elif "claim" in query_lower:
        if "claim" in context_lower or "procedure" in context_lower:
            answer = "Claim procedures are outlined in the provided policy documentation. Please follow the specified claim process."
            source = "Claims procedure section"
        else:
            answer = "For claim procedures, please refer to the claims section of your policy document."
            source = "Claims section"
    
    elif "coverage" in query_lower or "cover" in query_lower:
        if "coverage" in context_lower or "cover" in context_lower:
            answer = "Coverage details are specified in the policy context provided. Review the terms for specific coverage limits."
            source = "Coverage terms and conditions"
        else:
            answer = "Coverage details depend on the specific policy terms and conditions outlined in your document."
            source = "Coverage clauses"
    
    elif "premium" in query_lower or "payment" in query_lower:
        if "premium" in context_lower or "payment" in context_lower:
            answer = "Premium information is mentioned in the policy context. Check for payment schedules and amounts."
            source = "Premium and payment section"
        else:
            answer = "Premium information is specified in your policy schedule and payment terms."
            source = "Premium section"
    
    elif "waiting period" in query_lower or "wait" in query_lower:
        if "waiting" in context_lower or "period" in context_lower:
            answer = "Waiting period information is available in the policy context. Check specific waiting periods for different benefits."
            source = "Waiting period clauses"
        else:
            answer = "Waiting periods may apply to certain benefits. Please check your policy terms for specific waiting period requirements."
            source = "Policy terms regarding waiting periods"
    
    elif context_chunks and len(combined_context) > 50:
        # Generic but context-aware answer
        answer = f"Based on the policy documentation provided, your query about '{query}' relates to the policy terms. Please review the relevant sections for detailed information."
        source = "Policy documentation analysis"
    
    else:
        # Default fallback
        answer = f"For your query about '{query}', please refer to the relevant sections in your policy document for accurate information."
        source = "General policy terms"

    # Include relevant context preview if available
    context_preview = combined_context[:300] + "..." if len(combined_context) > 300 else combined_context
    explanation = f"This is a rule-based analysis of your query. Context chunks analyzed: {len(context_chunks)}.\n\nRelevant context preview:\n{context_preview}" if context_chunks else "No context chunks were provided for analysis."

    fallback_response = {
        "answer": answer,
        "source": source,
        "explanation": explanation,
        "confidence": 0.7 if context_chunks else 0.5,  # Higher confidence if context is available
        "query_processed": query,
        "context_chunks_count": len(context_chunks),
        "model_used": "Enhanced Fallback Analysis",
        "fallback_reason": "Model/API unavailable - using context-aware rules",
    }
    
    if error:
        fallback_response["error"] = error
        fallback_response["fallback_reason"] += f" (Error: {error})"

    return fallback_response


def generate_answer(query: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Generate structured answer by sending prompt to Gemini API and parse response.
    Falls back to simple text-based answer if API fails.

    Args:
        query (str): User question
        context_chunks (List[str]): Relevant document context chunks

    Returns:
        Dict[str, Any]: Structured answer including answer, source, explanation, confidence, and metadata
    """
    logger.info(f"Generating answer for query: {query[:50]}...")

    if not validate_query(query):
        return {
            "answer": "Invalid or too short query.",
            "source": None,
            "explanation": "",
            "confidence": 0.0,
            "query_processed": query,
            "context_chunks_count": len(context_chunks) if context_chunks else 0,
            "model_used": "Validation",
        }

    # Check if API key is available
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        logger.warning("Gemini API key not configured, using fallback")
        return _simple_text_based_answer(query, context_chunks, error="API key not configured")

    prompt = format_prompt(query, context_chunks)

    headers = {
        "Content-Type": "application/json",
    }

    # Correct Gemini API payload format
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 512,
        }
    }

    try:
        logger.info("Making request to Gemini API...")
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"Gemini API returned status {response.status_code}: {response.text}")
            return _simple_text_based_answer(query, context_chunks, error=f"API error {response.status_code}")
        
        result = response.json()

        # Extract generated text from Gemini response format
        generated_text = ""
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    generated_text = parts[0]["text"]

        if not generated_text:
            logger.error("No text generated from Gemini API response")
            return _simple_text_based_answer(query, context_chunks, error="No response from API")

        logger.info(f"Generated text: {generated_text[:100]}...")

        # Attempt to parse JSON output
        try:
            # Clean the response - sometimes it includes markdown formatting
            clean_text = generated_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            
            parsed_json = json.loads(clean_text)
            required_fields = {"answer", "source", "explanation"}
            if not required_fields.issubset(parsed_json):
                raise ValueError("Missing required fields in JSON")
            confidence = 0.9
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            parsed_json = {
                "answer": generated_text or "Unable to generate a meaningful answer.",
                "source": "Generated by Gemini API",
                "explanation": f"The answer is based on the provided context from {len(context_chunks)} document chunks.",
            }
            confidence = 0.75

        response = {
            **parsed_json,
            "confidence": confidence,
            "query_processed": query,
            "context_chunks_count": len(context_chunks) if context_chunks else 0,
            "model_used": "Gemini API",
        }
        logger.info("Successfully generated answer using Gemini API")
        return response

    except requests.RequestException as e:
        logger.error(f"Gemini API request failed: {e}")
        return _simple_text_based_answer(query, context_chunks, error=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during answer generation: {e}")
        return _simple_text_based_answer(query, context_chunks, error=f"Unexpected error: {str(e)}")


# Optional test function to verify connectivity
def test_answer_generator():
    test_query = "Is maternity covered under this policy?"
    test_context_chunks = [
        "Clause 3.1: The policy includes coverage for hospitalization expenses up to Rs. 5 lakhs.",
        "Clause 4.2: Maternity benefits apply after a waiting period of 36 months with coverage up to Rs. 1 lakh.",
    ]

    print("🧪 Testing Gemini API integration")
    response = generate_answer(test_query, test_context_chunks)

    print(f"Query: {response['query_processed']}")
    print(f"Answer: {response['answer']}")
    print(f"Source: {response['source']}")
    print(f"Confidence: {response['confidence']}")
    print(f"Context chunks: {response['context_chunks_count']}")
    print(f"Model used: {response['model_used']}")
    print(f"Explanation:\n{response.get('explanation', '')}")

    if "error" in response:
        print(f"Error: {response['error']}")

    return response


if __name__ == "__main__":
    from pprint import pprint

    context = [
        "Clause 3.1: The policy includes coverage for hospitalization expenses up to Rs. 5 lakhs per year.",
        "Clause 4.2: Maternity benefits apply after a waiting period of 36 months with coverage up to Rs. 1 lakh.",
    ]
    query = "Is maternity covered under this policy?"
    pprint(generate_answer(query, context))
