"""
Services Package - HackRX Query System
======================================

This package contains all the service modules for the query system.

Modules:
- answer_generator: Handles answer generation using LLM prompts
"""

from .answer_generator import generate_answer, validate_query, format_prompt

__all__ = ["generate_answer", "validate_query", "format_prompt"]
