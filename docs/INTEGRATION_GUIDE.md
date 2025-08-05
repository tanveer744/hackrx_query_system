# HackRx Embedding & Retrieval System - Integration Guide

This guide explains how to integrate the embedding and retrieval system into the main HackRx FastAPI application.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (for OpenAI)

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Basic Usage

```python
from app.services.retrieval_pipeline import HackRxRetrieval

# Initialize the retrieval system
retrieval = HackRxRetrieval()

# Add documents (from parser output)
documents = [
    {
        "text": "Policy text chunk...",
        "source": "policy.pdf",
        "page": 1,
        "section": "Coverage"
    }
]
retrieval.process_documents(documents)

# Query for relevant information
results = retrieval.query("What is covered?", top_k=3)

# Get formatted context for LLM
context = retrieval.get_context_for_llm("What is covered?", max_chunks=3)
```

## FastAPI Integration

### Integration with `/hackrx/run` Endpoint

```python
from fastapi import FastAPI
from app.services.retrieval_pipeline import HackRxRetrieval

app = FastAPI()

# Initialize retrieval system (do this once at startup)
retrieval_system = HackRxRetrieval()

@app.post("/hackrx/run")
async def hackrx_run(request: QueryRequest):
    """Main HackRx endpoint with integrated retrieval."""
    
    # Step 1: Parse documents (Member 2's work)
    # documents = parse_documents(request.files)
    
    # Step 2: Add to retrieval system (Member 3's work - THIS IS YOUR PART)
    # retrieval_system.process_documents(documents)
    
    # Step 3: Retrieve relevant context (Member 3's work - THIS IS YOUR PART)
    relevant_chunks = retrieval_system.get_context_for_llm(
        request.query, 
        max_chunks=5
    )
    
    # Step 4: Generate answer with LLM (Member 1's work)
    # answer = generate_answer(request.query, relevant_chunks)
    
    return {"answer": "Generated answer with retrieved context"}
```

## Component APIs

### 1. Embedding Service

```python
from app.services.embedder import create_embedder

# Create embedder
embedder = create_embedder("openai")  # or "huggingface"

# Single text
embedding = embedder.embed_single("Some text")

# Batch processing
embeddings = embedder.embed_batch(["text1", "text2", "text3"])

# Query embedding
query_embedding = embedder.embed_query("What is the policy?")
```

### 2. Vector Store Service

```python
from app.services.vector_store import DocumentVectorStore

# Initialize
vector_store = DocumentVectorStore(embedding_dimension=1536, data_dir="data")

# Add documents
vector_store.add_documents(
    texts=["text1", "text2"],
    embeddings=embeddings_array,
    sources=["doc1.pdf", "doc2.pdf"],
    pages=[1, 2]
)

# Search
results = vector_store.search_documents(query_embedding, k=5)
```

### 3. Reranker Service

```python
from app.services.reranker import create_reranker

# Create reranker
reranker = create_reranker("cross_encoder")  # or "hybrid"

# Rerank search results
reranked = reranker.rerank("query", search_results)
```

### 4. Complete Pipeline

```python
from app.services.retrieval_pipeline import create_retrieval_pipeline

# Create pipeline
pipeline = create_retrieval_pipeline(
    embedding_provider="openai",
    use_reranker=True,
    reranker_strategy="hybrid"
)

# Add documents
pipeline.add_documents(documents)

# Search
results = pipeline.search("query", k=5)

# Get text chunks for LLM
chunks = pipeline.get_relevant_chunks("query", k=3)
```

## Configuration Options

### Embedding Providers

1. **OpenAI** (Recommended for quality)
   - Model: `text-embedding-3-small`
   - Requires API key
   - Higher quality embeddings

2. **HuggingFace** (Free, local)
   - Model: `BAAI/bge-small-en`
   - No API key required
   - Good quality, runs locally

### Reranking Strategies

1. **Cross-Encoder**: Uses semantic similarity scoring
2. **Hybrid**: Combines vector similarity + cross-encoder (recommended)

## Data Persistence

The system automatically saves and loads:
- `data/faiss_index.bin` - Vector index
- `data/metadata.json` - Document metadata

## Testing

Run the comprehensive test suite:

```bash
python test_embedding_system.py
```

## Performance Considerations

### Embedding Generation
- Batch processing is more efficient than single embeddings
- OpenAI has rate limits (consider batching)
- HuggingFace models run locally (no rate limits)

### Vector Search
- FAISS IndexFlatL2 is fast for up to ~1M vectors
- Memory usage: ~4 bytes per dimension per vector
- Search is very fast (milliseconds)

### Reranking
- Adds latency but improves accuracy
- Cross-encoder is slower than vector search
- Consider using only for top-k results

## Error Handling

The system gracefully handles:
- Missing API keys (falls back to HuggingFace)
- Missing dependencies (disables optional features)
- Empty queries and documents
- File I/O errors

## Integration Checklist

- [ ] Install dependencies from `requirements.txt`
- [ ] Set `OPENAI_API_KEY` environment variable (optional)
- [ ] Import `HackRxRetrieval` in your FastAPI app
- [ ] Initialize retrieval system at startup
- [ ] Call `process_documents()` with parsed documents
- [ ] Call `get_context_for_llm()` for query processing
- [ ] Pass retrieved context to LLM for answer generation

## Example Full Integration

```python
from fastapi import FastAPI, HTTPException
from app.services.retrieval_pipeline import HackRxRetrieval
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# Initialize at startup
retrieval_system = None

@app.on_event("startup")
async def startup_event():
    global retrieval_system
    try:
        retrieval_system = HackRxRetrieval()
        logger.info("Retrieval system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize retrieval system: {e}")
        raise

@app.post("/hackrx/run")
async def hackrx_run(request: dict):
    if not retrieval_system:
        raise HTTPException(500, "Retrieval system not initialized")
    
    try:
        # Get query from request
        query = request.get("query", "")
        
        # Retrieve relevant context
        context = retrieval_system.get_context_for_llm(query, max_chunks=5)
        
        # TODO: Pass context to LLM for answer generation
        # answer = your_llm_function(query, context)
        
        return {
            "query": query,
            "context": context,
            "answer": "Generated answer would go here"
        }
        
    except Exception as e:
        logger.error(f"Error in hackrx_run: {e}")
        raise HTTPException(500, f"Internal server error: {e}")
```

## Support

For issues with the embedding and retrieval system:
1. Check the logs for detailed error messages
2. Run `test_embedding_system.py` to validate setup
3. Verify all dependencies are installed
4. Check API key configuration for OpenAI

The system is designed to be robust and will fall back to local models if cloud services are unavailable.
