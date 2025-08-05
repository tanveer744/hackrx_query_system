import os
import uuid
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from docx import Document
from docx.document import Document as DocxDocument

# Load environment variables
load_dotenv()

class DocumentLoader:
    """Handles loading and processing of PDF and DOCX documents using Azure Document Intelligence."""
    
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_key = os.getenv("AZURE_KEY")
        
    def get_doc_id(self, file_path: str) -> str:
        """
        Generate a unique document ID using UUID and filename.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Unique document ID in format "uuid-filename"
        """
        filename = Path(file_path).stem
        return f"{uuid.uuid4()}-{filename}"

    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """
        Extract text from PDF using Azure Document Intelligence.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List[Dict]: List of dictionaries containing doc_id, page number, and text
        """
        if not all([self.azure_endpoint, self.azure_key]):
            raise ValueError("Azure credentials not properly configured")
            
        doc_id = self.get_doc_id(file_path)
        result = []
        
        try:
            document_analysis_client = DocumentAnalysisClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(self.azure_key)
            )
            
            with open(file_path, "rb") as f:
                poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-layout", document=f
                )
                layout = poller.result()
                
                for page_num, page in enumerate(layout.pages, 1):
                    page_text = ""
                    for line in page.lines:
                        page_text += f"{line.content}\n"
                    
                    if page_text.strip():
                        result.append({
                            "doc_id": doc_id,
                            "page": page_num,
                            "text": page_text.strip()
                        })
                        
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")
            
        return result

    def extract_text_from_docx(self, file_path: str) -> List[Dict]:
        """
        Extract text from DOCX using python-docx.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List[Dict]: List of dictionaries containing doc_id, page number, and text
        """
        doc_id = self.get_doc_id(file_path)
        result = []
        
        try:
            doc = Document(file_path)
            full_text = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            
            # Join all paragraphs and split by page breaks
            full_text = "\n\n".join(full_text)
            pages = full_text.split('\x0c')  # Form feed character for page breaks
            
            for page_num, page_text in enumerate(pages, 1):
                if page_text.strip():
                    result.append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "text": page_text.strip()
                    })
                    
        except Exception as e:
            raise Exception(f"Error processing DOCX {file_path}: {str(e)}")
            
        return result