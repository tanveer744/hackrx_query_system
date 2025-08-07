"""
Answer Generator Module - Member 4 (Prompt Engineer)
====================================================

This module provides a dummy implementation for generating answers 
using a simulated LLM response format. This serves as a placeholder
for the actual prompt engineering implementation.

Author: Nehal (Member 4 - Prompt Engineer)
Day: 2 - Refactored to accept context chunks
"""

from typing import Dict, Any, Optional, List


def generate_answer(query: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Simulates a GPT-generated answer using input context chunks.
    Returns a JSON-style dict with answer, source clause, and explanation.
    
    Args:
        query (str): The user's question/query
        context_chunks (List[str]): List of relevant document context chunks
        
    Returns:
        Dict[str, Any]: Standardized response format with answer, source, and explanation
    """
    # Combine context chunks
    combined_context = "\n---\n".join(context_chunks) if context_chunks else ""
    
    return {
        "answer": f"Yes. This is a dummy answer to: '{query}'",
        "source": "Simulated Clause 4.2 - Benefit Inclusion",
        "explanation": f"The answer is based on the following context:\n{combined_context}",
        "confidence": 0.85,
        "query_processed": query,
        "context_chunks_count": len(context_chunks) if context_chunks else 0
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


def format_prompt(query: str, context_chunks: List[str], max_context_length: int = 1000) -> str:
    """
    Formats the prompt for the LLM using context chunks.
    
    Args:
        query (str): The user query
        context_chunks (List[str]): List of document context chunks
        max_context_length (int): Maximum length of combined context to include
        
    Returns:
        str: Formatted prompt string
    """
    # Combine and truncate context if too long
    combined_context = "\n---\n".join(context_chunks) if context_chunks else ""
    if len(combined_context) > max_context_length:
        combined_context = combined_context[:max_context_length] + "..."
    
    prompt = f"""
    Based on the following context, please answer the question:
    
    Context: {combined_context}
    
    Question: {query}
    
    Please provide a comprehensive answer with source references.
    """
    
    return prompt.strip()


# Test function for development
def test_answer_generator():
    """
    Simple test function to verify the answer generator works with context chunks.
    """
    test_query = "Is maternity covered under this policy?"
    test_context_chunks = [
        "Clause 3.1: The policy includes coverage for hospitalization expenses.",
        "Clause 4.2: Maternity benefits apply after a waiting period of 36 months."
    ]
    
    result = generate_answer(test_query, test_context_chunks)
    print("Test Result:")
    print(f"Query: {result['query_processed']}")
    print(f"Answer: {result['answer']}")
    print(f"Source: {result['source']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Context chunks count: {result['context_chunks_count']}")
    print("Explanation:")
    print(result['explanation'])
    
    return result


if __name__ == "__main__":
    # Run test when file is executed directly
    from pprint import pprint
    
    context = [
        "Clause 3.1: The policy includes coverage for hospitalization expenses.",
        "Clause 4.2: Maternity benefits apply after a waiting period of 36 months."
    ]
    query = "Is maternity covered under this policy?"
    result = generate_answer(query, context)
    pprint(result)
