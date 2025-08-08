"""
HackRx Query System - Main FastAPI Application
Complete pipeline implementation using existing service functions
"""

import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse, AnswerItem
from app.services.document_loader import DocumentLoader
from app.services.chunker import DocumentChunker
from app.services.embedder import get_embeddings
from app.services.vector_store import FAISSIndex
from app.services.answer_generator import generate_answer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx Query System",
    description="Semantic search and retrieval system for policy documents",
    version="1.0.0"
)

@app.post("/hackrx/run", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """
    Main HackRx endpoint that processes documents and answers questions.
    Uses existing service classes and functions with proper error handling.
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document source: {request.documents[:100]}...")
        
        # 1. Parse document using DocumentLoader (supports URLs and local files)
        logger.info("Step 1: Loading document...")
        loader = DocumentLoader()
        text_blocks = loader.parse_document(request.documents)
        logger.info(f"Extracted {len(text_blocks)} text blocks")
        
        if not text_blocks:
            raise ValueError("No text blocks extracted from document")

        # 2. Chunk text using DocumentChunker
        logger.info("Step 2: Chunking text...")
        chunker = DocumentChunker()
        chunk_dicts = chunker.chunk_text(text_blocks)
        chunks = [chunk_dict['chunk'] for chunk_dict in chunk_dicts if chunk_dict.get('chunk', '').strip()]
        logger.info(f"Generated {len(chunks)} valid chunks")
        
        if not chunks:
            raise ValueError("No valid chunks generated from document")

        # 3. Embed chunks using existing get_embeddings function
        logger.info("Step 3: Generating embeddings...")
        chunk_embeddings = get_embeddings(chunks)
        logger.info(f"Generated embeddings with dimension {len(chunk_embeddings[0])}")
        
        if not chunk_embeddings or not chunk_embeddings[0]:
            raise ValueError("Failed to generate embeddings")

        # 4. Store in FAISS using existing FAISSIndex
        logger.info("Step 4: Building FAISS index...")
        index = FAISSIndex(len(chunk_embeddings[0]))
        metadata = [{"index": i} for i in range(len(chunks))]
        index.add(chunk_embeddings, chunks, metadata)
        logger.info("FAISS index built successfully")

        # 5. Answer each question
        logger.info("Step 5: Processing questions...")
        answers = []
        for i, q in enumerate(request.questions):
            if not q or not q.strip():
                continue  # Skip empty questions
            
            logger.info(f"Processing question {i+1}: {q[:50]}...")
            q_embedding = get_embeddings([q])[0]
            search_results = index.search(q_embedding)
            top_chunks = [c["text"] for c in search_results if "text" in c]
            
            logger.info(f"Found {len(top_chunks)} relevant chunks for question")
            result = generate_answer(q, top_chunks)
            
            # Ensure all required fields are present
            answer_item = {
                "answer": result.get("answer", "No answer generated"),
                "source": result.get("source", "Unknown"),
                "explanation": result.get("explanation", "No explanation provided"),
                "confidence": result.get("confidence"),
                "query_processed": result.get("query_processed"),
                "context_chunks_count": result.get("context_chunks_count"),
                "model_used": result.get("model_used")
            }
            answers.append(AnswerItem(**answer_item))

        logger.info(f"Successfully processed all questions, returning {len(answers)} answers")
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Return error as answer for debugging
        error_answer = AnswerItem(
            answer=f"Error processing request: {str(e)}",
            source="System Error", 
            explanation="An error occurred during processing. Please check your input and try again."
        )
        return {"answers": [error_answer]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Complex retrieval system code commented out for future integration
# Original complex code with retrieval system integration can be restored later
