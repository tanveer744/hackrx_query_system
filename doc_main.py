import sys
import os
from services.document_loader import DocumentLoader
from services.chunker import DocumentChunker
from services.utils import clean_document_text

def process_document(file_path: str):
    """Process a single document and return the chunks."""
    # Initialize components
    loader = DocumentLoader()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)  # Changed parameter name here

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return []

    # Process the document
    try:
        if file_path.lower().endswith('.pdf'):
            text_blocks = loader.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            text_blocks = loader.extract_text_from_docx(file_path)
        else:
            print("Error: Unsupported file format. Please use .pdf or .docx files.")
            return []

        # Clean text blocks
        for block in text_blocks:
            block['text'] = clean_document_text(block['text'])

        # Chunk the text
        chunks = chunker.chunk_text(text_blocks)
        return chunks

    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python doc_main.py <path_to_document>")
        print("Example: python doc_main.py documents/example.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    chunks = process_document(file_path)
    
    if chunks:
        print(f"\nSuccessfully processed {len(chunks)} chunks from {file_path}")
        print("\nSample chunk:")
        print(chunks[0])  # Print first chunk as sample