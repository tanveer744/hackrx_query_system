"""
HackRx Query System - Main FastAPI Application
Simple schema-based implementation for testing
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse

# Load environment variables
load_dotenv()

app = FastAPI(
    title="HackRx Query System",
    description="Semantic search and retrieval system for policy documents",
    version="1.0.0"
)

@app.post("/hackrx/run", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """
    Main HackRx endpoint with proper schema validation.
    Returns placeholder answers for now.
    """
    return QueryResponse(
        answers=[
            {
                "answer": "This is a placeholder answer.",
                "source": "No source",
                "explanation": "Dummy explanation"
            }
        ]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Complex retrieval system code commented out for future integration
# Original complex code with retrieval system integration can be restored later
