"""
HackRx Query System - Main FastAPI Application
Complete pipeline implementation using existing service functions
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse, AnswerItem
from app.services.document_loader import DocumentLoader
from app.services.chunker import DocumentChunker
from app.services.embedder import get_embeddings
from app.services.vector_store import FAISSIndex
from app.services.answer_generator import generate_answer

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
    Main HackRx endpoint that processes documents and answers questions.
    Uses existing service classes and functions.
    """
    # 1. Parse document using DocumentLoader
    loader = DocumentLoader()
    if request.documents.lower().endswith('.pdf'):
        text_blocks = loader.extract_text_from_pdf(request.documents)
    elif request.documents.lower().endswith('.docx'):
        text_blocks = loader.extract_text_from_docx(request.documents)
    else:
        raise ValueError(f"Unsupported file type: {request.documents}")
    
    # Extract text strings from the blocks
    full_text = [block['text'] for block in text_blocks if block['text'].strip()]

    # 2. Chunk text using DocumentChunker
    chunker = DocumentChunker()
    # Convert text to the format expected by chunker
    text_blocks_for_chunker = [
        {"doc_id": f"doc_{i}", "page": i+1, "text": text}
        for i, text in enumerate(full_text)
    ]
    chunk_dicts = chunker.chunk_text(text_blocks_for_chunker)
    chunks = [chunk_dict['chunk'] for chunk_dict in chunk_dicts if chunk_dict['chunk'].strip()]

    # 3. Embed chunks using existing get_embeddings function
    chunk_embeddings = get_embeddings(chunks)

    # 4. Store in FAISS using existing FAISSIndex
    index = FAISSIndex(len(chunk_embeddings[0]))
    metadata = [{"index": i} for i in range(len(chunks))]
    index.add(chunk_embeddings, chunks, metadata)

    # 5. Answer each question
    answers = []
    for q in request.questions:
        q_embedding = get_embeddings([q])[0]
        top_chunks = [c["text"] for c in index.search(q_embedding)]
        result = generate_answer(q, top_chunks)  # Correct parameter order
        answers.append(AnswerItem(**result))

    return {"answers": answers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Complex retrieval system code commented out for future integration
# Original complex code with retrieval system integration can be restored later
