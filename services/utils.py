import re
from typing import Optional
from unidecode import unidecode

def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and fixing encoding.
    
    Args:
        text: Input text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to ASCII, replacing special characters
    text = unidecode(text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters except basic punctuation
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    
    return text.strip()

def remove_headers_and_footers(text: str, min_line_length: int = 20) -> str:
    """
    Remove common headers and footers from text.
    
    Args:
        text: Input text
        min_line_length: Minimum length of line to be considered content (not header/footer)
        
    Returns:
        str: Text with headers and footers removed
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that are too short and contain common header/footer words
        if (len(line) > min_line_length or 
            not any(word in line.lower() for word in 
                   ['page', 'confidential', 'copyright', '©', 'www.', 'http'])):
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def strip_whitespace_lines(text: str) -> str:
    """
    Remove empty or whitespace-only lines from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with empty lines removed
    """
    if not text:
        return ""
    
    lines = [line for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def clean_document_text(text: str) -> str:
    """
    Apply all cleaning functions to document text.
    
    Args:
        text: Input document text
        
    Returns:
        str: Cleaned document text
    """
    if not text:
        return ""
        
    text = normalize_text(text)
    text = remove_headers_and_footers(text)
    text = strip_whitespace_lines(text)
    
    return text