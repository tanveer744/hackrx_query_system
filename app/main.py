"""
HackRx Query System - Main FastAPI Application
Demonstrates integration of the embedding & retrieval system.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx Query System",
    description="Semantic search and retrieval system for policy documents",
    version="1.0.0"
)

# Global retrieval system instance
retrieval_system = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class DocumentRequest(BaseModel):
    documents: List[Dict[str, Any]]

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    context: str
    total_results: int

@app.on_event("startup")
async def startup_event():
    """Initialize the retrieval system on startup."""
    global retrieval_system

    try:
        from services.retrieval_pipeline import HackRxRetrieval
        retrieval_system = HackRxRetrieval()
        logger.info("✓ Retrieval system initialized successfully")

        # Load sample data if available
        stats = retrieval_system.pipeline.get_stats()
        if stats["total_vectors"] > 0:
            logger.info(f"✓ Loaded existing index with {stats['total_vectors']} documents")
        else:
            logger.info("ℹ No existing documents found. Use /documents endpoint to add documents.")

    except Exception as e:
        logger.error(f"✗ Failed to initialize retrieval system: {e}")
        logger.error("Make sure to install dependencies: pip install -r requirements.txt")

@app.get("/")
async def root():
    """Root endpoint with system information."""
    if retrieval_system:
        stats = retrieval_system.pipeline.get_stats()
        return {
            "message": "HackRx Query System is running",
            "status": "ready",
            "stats": stats
        }
    else:
        return {
            "message": "HackRx Query System",
            "status": "initialization_failed",
            "error": "Retrieval system not available"
        }

@app.post("/documents", response_model=dict)
async def add_documents(request: DocumentRequest):
    """Add documents to the retrieval system."""
    if not retrieval_system:
        raise HTTPException(status_code=500, detail="Retrieval system not initialized")

    try:
        retrieval_system.process_documents(request.documents)
        stats = retrieval_system.pipeline.get_stats()

        return {
            "message": f"Successfully added {len(request.documents)} documents",
            "total_documents": stats["total_vectors"]
        }

    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the retrieval system for relevant documents."""
    if not retrieval_system:
        raise HTTPException(status_code=500, detail="Retrieval system not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Get search results
        results = retrieval_system.query(request.query, top_k=request.top_k)

        # Get formatted context for LLM
        context = retrieval_system.get_context_for_llm(request.query, max_chunks=min(request.top_k, 3))

        return QueryResponse(
            query=request.query,
            results=results,
            context=context,
            total_results=len(results)
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

@app.post("/hackrx/run")
async def hackrx_run(request: dict):
    """
    Main HackRx endpoint (placeholder for full integration).
    This is where Member 1 would integrate with the LLM for final answer generation.
    """
    if not retrieval_system:
        raise HTTPException(status_code=500, detail="Retrieval system not initialized")

    try:
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Step 1: Retrieve relevant context (Member 3's work - COMPLETED)
        context = retrieval_system.get_context_for_llm(query, max_chunks=5)
        search_results = retrieval_system.query(query, top_k=5)

        # Step 2: Generate answer with LLM (Member 1's work - TO BE INTEGRATED)
        # This is where Member 1 would call their LLM service
        # answer = llm_service.generate_answer(query, context)

        # For now, return the retrieved context
        return {
            "query": query,
            "context": context,
            "search_results": search_results,
            "answer": "LLM integration pending - this is the retrieved context that would be passed to the LLM",
            "status": "retrieval_complete"
        }

    except Exception as e:
        logger.error(f"Error in hackrx_run: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not retrieval_system:
        return {"error": "Retrieval system not initialized"}

    return retrieval_system.pipeline.get_stats()

@app.delete("/clear")
async def clear_index():
    """Clear all documents from the index."""
    if not retrieval_system:
        raise HTTPException(status_code=500, detail="Retrieval system not initialized")

    try:
        retrieval_system.pipeline.clear_index()
        return {"message": "Index cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
def run_query(request: QueryRequest):
    return {"answers": ["This is a dummy response for now."]}
