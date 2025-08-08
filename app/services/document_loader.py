import os
import uuid
import requests
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from docx import Document

# Try to import PyPDF2 as fallback
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

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

    def extract_text_from_pdf_fallback(self, file_path: str) -> List[Dict]:
        """
        Fallback PDF extraction using PyPDF2.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List[Dict]: List of dictionaries containing doc_id, page number, and text
        """
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 not available for fallback PDF extraction")
        
        doc_id = self.get_doc_id(file_path)
        result = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"📄 PyPDF2 detected {len(pdf_reader.pages)} pages in the document")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            result.append({
                                "doc_id": doc_id,
                                "page": page_num,
                                "text": page_text.strip()
                            })
                            print(f"   Page {page_num}: {len(page_text.strip()):,} characters extracted")
                        else:
                            print(f"   Page {page_num}: Empty page, skipped")
                    except Exception as e:
                        print(f"   Page {page_num}: Error extracting text - {e}")
                        
                print(f"✅ PyPDF2 fallback extracted {len(result)} pages with content")
                
        except Exception as e:
            raise Exception(f"PyPDF2 fallback error processing PDF {file_path}: {str(e)}")
            
        return result

    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """
        Main PDF extraction method that prioritizes PyPDF2 for complete page coverage,
        with Azure as a backup for enhanced text recognition.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List[Dict]: List of dictionaries containing doc_id, page number, and text
        """
        doc_id = self.get_doc_id(file_path)
        
        # Try PyPDF2 first as it extracts all pages reliably
        if PYPDF2_AVAILABLE:
            print("🚀 Starting with PyPDF2 extraction (recommended for complete coverage)...")
            try:
                pypdf2_result = self.extract_text_from_pdf_fallback(file_path)
                if len(pypdf2_result) > 2:  # If we got good results
                    print(f"✅ PyPDF2 successfully extracted {len(pypdf2_result)} pages")
                    
                    # Optionally try Azure for text quality comparison on first few pages
                    if all([self.azure_endpoint, self.azure_key]):
                        try:
                            print("� Comparing with Azure text quality on first 2 pages...")
                            azure_result = self.extract_text_from_pdf_azure_original(file_path, doc_id)
                            if len(azure_result) >= 2 and len(pypdf2_result) >= 2:
                                # Compare text quality (Azure might have better OCR)
                                if len(azure_result[0]['text']) > len(pypdf2_result[0]['text']) * 1.2:
                                    print("🎯 Azure has significantly better text quality, using hybrid approach")
                                    # Use Azure for first few pages, PyPDF2 for the rest
                                    hybrid_result = azure_result[:2] + pypdf2_result[2:]
                                    return hybrid_result
                        except Exception as e:
                            print(f"⚠️ Azure quality check failed: {e}")
                    
                    return pypdf2_result
                    
            except Exception as e:
                print(f"❌ PyPDF2 failed: {e}")
        
        # Fallback to Azure if PyPDF2 is not available or failed
        if all([self.azure_endpoint, self.azure_key]):
            print("🔄 Falling back to Azure Document Intelligence...")
            return self.extract_text_from_pdf_azure_original(file_path, doc_id)
        else:
            raise Exception("No PDF extraction methods available. Install PyPDF2 or configure Azure credentials.")

    def extract_text_from_pdf_azure_enhanced(self, file_path: str, doc_id: str) -> List[Dict]:
        """
        Enhanced Azure Document Intelligence extraction with multiple model attempts.
        
        Args:
            file_path: Path to the PDF file
            doc_id: Document ID for tracking
            
        Returns:
            List[Dict]: List of dictionaries containing doc_id, page number, and text
        """
        result = []
        models_to_try = [
            ("prebuilt-layout", "Layout model (best for general documents)"),
            ("prebuilt-document", "Document model (good for mixed content)"),
            ("prebuilt-read", "Read model (optimized for text extraction)")
        ]

        try:
            document_analysis_client = DocumentAnalysisClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(self.azure_key)
            )

            with open(file_path, "rb") as f:
                file_size = os.path.getsize(file_path)
                print(f"📄 Processing PDF with Azure Enhanced: {Path(file_path).name} ({file_size:,} bytes)")
                
                for model_id, model_desc in models_to_try:
                    try:
                        print(f"🔄 Trying {model_desc}...")
                        
                        # Reset file position
                        f.seek(0)
                        
                        # Try different approaches to get all pages
                        poller = document_analysis_client.begin_analyze_document(
                            model_id, 
                            document=f,
                            # Add parameters to potentially improve extraction
                            pages="1-999" if model_id != "prebuilt-read" else None,  # Read model doesn't support pages parameter
                            locale="en-US" if model_id == "prebuilt-layout" else None  # Only layout model supports locale
                        )
                        layout = poller.result()

                        print(f"📊 {model_desc} detected {len(layout.pages)} pages in the document")
                        
                        current_result = []
                        for page_num, page in enumerate(layout.pages, 1):
                            page_text = ""
                             
                            # Different extraction methods based on model
                            if model_id == "prebuilt-read":
                                # Read model has different structure
                                for line in layout.pages[page_num-1].lines if hasattr(layout.pages[page_num-1], 'lines') else []:
                                    page_text += f"{line.content}\n"
                            else:
                                # Layout and document models
                                for line in page.lines:
                                    page_text += f"{line.content}\n"

                            if page_text.strip():
                                current_result.append({
                                    "doc_id": doc_id,
                                    "page": page_num,
                                    "text": page_text.strip()
                                })
                                print(f"   Page {page_num}: {len(page_text.strip()):,} characters extracted")
                            else:
                                print(f"   Page {page_num}: Empty page, skipped")
                        
                        print(f"✅ {model_desc} extracted {len(current_result)} pages with content")
                        
                        # If this model extracted more pages, use it
                        if len(current_result) > len(result):
                            result = current_result
                            print(f"🎯 Using {model_desc} as it extracted more pages ({len(current_result)})")
                             
                            # If we got a reasonable number of pages, stop trying other models
                            if len(current_result) >= 10:
                                break
                        
                    except Exception as model_error:
                        print(f"❌ {model_desc} failed: {str(model_error)}")
                        continue
                
                if result:
                    print(f"✅ Azure enhanced extraction completed: {len(result)} pages")
                else:
                    print("❌ All Azure models failed to extract content")

        except Exception as e:
            print(f"❌ Azure Document Intelligence enhanced extraction failed: {str(e)}")
        
        return result

    def extract_text_from_pdf_azure_original(self, file_path: str, doc_id: str) -> List[Dict]:
        """
        Extract text from PDF using Azure Document Intelligence with PyPDF2 fallback.

        Args:
            file_path: Path to the PDF file

        Returns:
            List[Dict]: List of dictionaries containing doc_id, page number, and text
        """
        if not all([self.azure_endpoint, self.azure_key]):
            print("⚠️ Azure credentials not configured, using PyPDF2 fallback")
            return self.extract_text_from_pdf_fallback(file_path)

        doc_id = self.get_doc_id(file_path)
        result = []

        try:
            document_analysis_client = DocumentAnalysisClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(self.azure_key)
            )

            with open(file_path, "rb") as f:
                # Check file size
                file_size = os.path.getsize(file_path)
                print(f"📄 Processing PDF with Azure: {file_path} ({file_size:,} bytes)")
                
                # Try different approaches to get all pages
                poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-layout", 
                    document=f,
                    # Add parameters to potentially improve extraction
                    pages="1-999",  # Explicitly request all pages
                    locale="en-US"  # Set locale for better text recognition
                )
                layout = poller.result()

                print(f"📊 Azure detected {len(layout.pages)} pages in the document")
                
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
                        print(f"   Page {page_num}: {len(page_text.strip()):,} characters extracted")
                    else:
                        print(f"   Page {page_num}: Empty page, skipped")
                
                print(f"✅ Azure successfully extracted {len(result)} pages with content")
                
                # If Azure extracted very few pages, try PyPDF2 fallback
                if len(result) <= 2 and PYPDF2_AVAILABLE:
                    print("⚠️ Azure extracted very few pages, trying PyPDF2 fallback...")
                    fallback_result = self.extract_text_from_pdf_fallback(file_path)
                    if len(fallback_result) > len(result):
                        print(f"✅ PyPDF2 extracted more pages ({len(fallback_result)} vs {len(result)}), using fallback")
                        return fallback_result

        except Exception as e:
            print(f"❌ Azure Document Intelligence failed: {str(e)}")
            if PYPDF2_AVAILABLE:
                print("🔄 Trying PyPDF2 fallback...")
                return self.extract_text_from_pdf_fallback(file_path)
            else:
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

    def parse_document(self, doc_url_or_path: str) -> List[Dict]:
        """
        Parse document from URL or local path. Download PDF if URL is provided; else use local file.
        Supports PDF and DOCX files.

        Args:
            doc_url_or_path: URL starting with 'http' or local file path

        Returns:
            List[Dict]: Extracted text data (doc_id, page, text)
        """
        if doc_url_or_path.lower().startswith("http"):
            # Currently only support downloading PDFs from URL
            extension = Path(doc_url_or_path).suffix.lower()
            if extension == ".pdf":
                temp_file = "temp.pdf"
                response = requests.get(doc_url_or_path)
                response.raise_for_status()
                with open(temp_file, "wb") as f:
                    f.write(response.content)
                doc_path = temp_file
                result = self.extract_text_from_pdf_azure_original(doc_path, self.get_doc_id(doc_path))
                # Optionally, remove the temp file after processing
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
                return result
            else:
                raise ValueError("Currently, only PDF URLs are supported for remote documents.")
        else:
            # Local file path: determine file type from extension
            extension = Path(doc_url_or_path).suffix.lower()
            if extension == ".pdf":
                return self.extract_text_from_pdf_azure_original(doc_url_or_path, self.get_doc_id(doc_url_or_path))
            elif extension == ".docx":
                return self.extract_text_from_docx(doc_url_or_path)
            else:
                raise ValueError("Unsupported file type. Only PDF or DOCX files are supported.")
