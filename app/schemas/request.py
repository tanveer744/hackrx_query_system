from pydantic import BaseModel, Field
from typing import List, Optional, Union

class QueryRequest(BaseModel):
    """
    Enhanced Query Request supporting multiple document input methods:
    1. Local filename (existing behavior)
    2. Document URL for download
    3. File upload via multipart/form-data (separate endpoint)
    """
    documents: str = Field(
        ..., 
        description="Document filename (if local) or URL (if remote). For file uploads, use /upload-query endpoint."
    )
    questions: List[str] = Field(
        ..., 
        description="List of questions to answer from the document"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://example.com/policy.pdf",
                "questions": ["What are the maternity benefits?"]
            }
        }

class UploadQueryRequest(BaseModel):
    """
    Query request for uploaded files with questions as JSON
    """
    questions: List[str] = Field(
        ..., 
        description="List of questions to answer from the uploaded document"
    )
