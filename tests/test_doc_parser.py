# test_doc_parser.py
import sys
import os
from pathlib import Path

# Add the services directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

from document_loader import DocumentLoader

def test_document_parsing():
    """Test document parsing functionality"""
    loader = DocumentLoader()
    
    # Test with a sample DOCX file (we created this for testing)
    sample_file = "sample_policy.docx"
    
    if not os.path.exists(sample_file):
        print(f"❌ Sample file '{sample_file}' not found!")
        print("Please ensure you have a sample PDF or DOCX file in the current directory.")
        print("Available files in current directory:")
        for file in os.listdir("."):
            if file.endswith(('.pdf', '.docx')):
                print(f"  - {file}")
        return
    
    try:
        print(f"🔍 Testing document parsing with: {sample_file}")
        
        # Determine file type and parse accordingly
        if sample_file.lower().endswith('.pdf'):
            result = loader.extract_text_from_pdf(sample_file)
            print("✅ PDF parsing successful!")
        elif sample_file.lower().endswith('.docx'):
            result = loader.extract_text_from_docx(sample_file)
            print("✅ DOCX parsing successful!")
        else:
            print("❌ Unsupported file format. Only PDF and DOCX are supported.")
            return
        
        print(f"\n📄 Document parsed successfully!")
        print(f"   Document ID: {result[0]['doc_id'] if result else 'N/A'}")
        print(f"   Number of pages: {len(result)}")
        
        # Display first page content (truncated)
        if result:
            first_page = result[0]['text']
            print(f"\n📝 First page content (first 500 characters):")
            print("-" * 50)
            print(first_page[:500] + "..." if len(first_page) > 500 else first_page)
            print("-" * 50)
        
        return result
        
    except Exception as e:
        print(f"❌ Error parsing document: {str(e)}")
        return None

if __name__ == "__main__":
    result = test_document_parsing()
    if result:
        print(f"\n✅ Test completed successfully! Extracted {len(result)} pages.")
    else:
        print("\n❌ Test failed!")
