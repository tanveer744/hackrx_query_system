"""
Answer Generator Module - Member 4 (Prompt Engineer)
====================================================

This module provides a dummy implementation for generating answers 
using a simulated LLM response format. This serves as a placeholder
for the actual prompt engineering implementation.

Author: Nehal (Member 4 - Prompt Engineer)
Day: 1 - Initial Setup
"""

from typing import Dict, Any, Optional


def generate_answer(query: str, context: str) -> Dict[str, Any]:
    """
    Dummy implementation that simulates GPT answer output.
    
    Args:
        query (str): The user's question/query
        context (str): The relevant document context/chunks
        
    Returns:
        Dict[str, Any]: Standardized response format with answer, source, and explanation
    """
    return {
        "answer": f"This is a dummy answer for: {query}",
        "source": "Clause X.Y from the document.",
        "explanation": "This is a placeholder explanation without using an actual LLM.",
        "confidence": 0.85,
        "query_processed": query,
        "context_length": len(context) if context else 0
    }


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


def format_prompt(query: str, context: str, max_context_length: int = 1000) -> str:
    """
    Formats the prompt for the LLM (dummy implementation).
    
    Args:
        query (str): The user query
        context (str): Document context
        max_context_length (int): Maximum length of context to include
        
    Returns:
        str: Formatted prompt string
    """
    # Truncate context if too long
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    prompt = f"""
    Based on the following context, please answer the question:
    
    Context: {context}
    
    Question: {query}
    
    Please provide a comprehensive answer with source references.
    """
    
    return prompt.strip()


# Test function for development
def test_answer_generator():
    """
    Simple test function to verify the answer generator works.
    """
    test_query = "What are the payment terms?"
    test_context = "The payment terms are Net 30 days from invoice date..."
    
    result = generate_answer(test_query, test_context)
    print("Test Result:")
    print(f"Query: {result['query_processed']}")
    print(f"Answer: {result['answer']}")
    print(f"Source: {result['source']}")
    print(f"Confidence: {result['confidence']}")
    
    return result


if __name__ == "__main__":
    # Run test when file is executed directly
    test_answer_generator()
