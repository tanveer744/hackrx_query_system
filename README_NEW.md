# HackRx Query System 🚀

A comprehensive document processing and query system built for the HackRx competition. This system provides intelligent document parsing, embedding generation, vector search, and answer generation capabilities.

## 📁 Project Structure

```
hackrx_query_system/
├── app/                          # Main application
│   ├── main.py                   # FastAPI application entry point
│   ├── dependencies/             # Dependency injection
│   ├── routers/                  # API route handlers
│   ├── schemas/                  # Pydantic models and schemas
│   └── services/                 # Core business logic
│       ├── __init__.py
│       ├── answer_generator.py   # Answer generation service
│       ├── chunker.py           # Document chunking utilities
│       ├── document_loader.py   # PDF/DOCX document processing
│       ├── embedder.py          # Text embedding generation
│       ├── reranker.py          # Search result reranking
│       ├── retrieval_pipeline.py # End-to-end retrieval pipeline
│       ├── utils.py             # Utility functions
│       └── vector_store.py      # FAISS vector database
├── data/                        # Data storage directory
│   └── README.md               # Data directory documentation
├── docs/                       # Documentation
│   ├── INTEGRATION_GUIDE.md    # Integration instructions
│   └── MEMBER3_COMPLETION_REPORT.md # Development reports
├── scripts/                    # Utility scripts
│   ├── doc_main.py            # Document processing script
│   └── example_usage.py       # Usage examples
├── tests/                     # Test suite
│   └── test_embedding_system.py # Comprehensive system tests
├── .gitignore                 # Git ignore rules
├── LICENSE                    # Project license
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohammedmaaz1786/hackrx_query_system.git
   cd hackrx_query_system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables** (create `.env` file)
   ```env
   AZURE_ENDPOINT=your_azure_endpoint
   AZURE_KEY=your_azure_key
   OPENAI_API_KEY=your_openai_key  # Optional
   ```

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Access the API**
   - API: http://127.0.0.1:8000
   - Documentation: http://127.0.0.1:8000/docs

## 🧪 Testing

Run the comprehensive test suite:
```bash
python tests/test_embedding_system.py
```

## 🔧 Core Components

### 1. Document Processing (`document_loader.py`)
- **PDF Processing**: Azure Document Intelligence integration
- **DOCX Processing**: python-docx based text extraction
- **Chunking**: Intelligent document segmentation

### 2. Embeddings (`embedder.py`)
- **OpenAI Embeddings**: text-embedding-3-small model
- **HuggingFace Embeddings**: Local BAAI/bge-small-en model
- **Batch Processing**: Efficient embedding generation

### 3. Vector Search (`vector_store.py`)
- **FAISS Integration**: High-performance similarity search
- **Persistent Storage**: Index saving and loading
- **Metadata Management**: Document source tracking

### 4. Answer Generation (`answer_generator.py`)
- **LLM Integration**: GPT-based answer generation
- **Context-Aware**: Uses retrieved documents for answers
- **Structured Output**: JSON response format

### 5. Reranking (`reranker.py`)
- **Cross-Encoder**: Advanced relevance scoring
- **Hybrid Scoring**: Combines similarity and relevance
- **Configurable**: Adjustable scoring weights

## 📡 API Endpoints

### Main Query Endpoint
```http
POST /hackrx/run
Content-Type: application/json

{
  "documents": "document_text_here",
  "questions": ["What is the grace period?", "What are the coverage limits?"]
}
```

### Response Format
```json
{
  "answers": [
    {
      "question": "What is the grace period?",
      "answer": "The grace period is 30 days for premium payment.",
      "source": "health_policy.pdf, Page 3",
      "confidence": 0.95
    }
  ]
}
```

## 🚀 Performance Features

- **Batch Processing**: Efficient embedding generation
- **FAISS Indexing**: Sub-millisecond search times
- **Memory Optimization**: Chunked document processing
- **Caching**: Embedded results caching
- **Async Support**: Non-blocking API operations

## 🛡️ Error Handling

- Comprehensive exception handling
- Graceful fallbacks for missing dependencies
- Detailed error logging and reporting
- Input validation and sanitization

## 📊 Supported File Formats

- **PDF**: Via Azure Document Intelligence
- **DOCX**: Via python-docx
- **TXT**: Direct text processing
- **Future**: Excel, PowerPoint, HTML

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 HackRx Competition

This system was developed for the HackRx competition, showcasing advanced document processing and AI-powered query capabilities.

---

**Built with ❤️ for HackRx Competition**
