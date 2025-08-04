# Data Directory

This directory contains the FAISS vector index and associated metadata for the HackRx Query System.

## Files

### `faiss_index.bin`
- FAISS vector index file containing document embeddings
- Binary format, created and managed by the vector store service
- Contains dense vector representations of document chunks

### `metadata.json`
- JSON file containing metadata for each vector in the index
- Includes text content, source document, page numbers, sections, etc.
- Maintains 1:1 correspondence with vectors in the FAISS index

## Structure

```
data/
├── README.md           # This file
├── faiss_index.bin     # FAISS vector index (created when documents are added)
├── metadata.json       # Document metadata (created when documents are added)
└── backups/           # Optional backup directory
```

## Usage

The vector store service automatically manages these files:

1. **Adding Documents**: When documents are processed and embedded, the service:
   - Adds vectors to `faiss_index.bin`
   - Appends metadata to `metadata.json`
   - Automatically saves both files

2. **Loading**: On startup, the service attempts to load existing files:
   - Reads the FAISS index from `faiss_index.bin`
   - Loads metadata from `metadata.json`
   - Validates consistency between index and metadata

3. **Searching**: During query processing:
   - Uses FAISS index for fast similarity search
   - Returns corresponding metadata for matched vectors

## File Formats

### Metadata JSON Structure
```json
[
  {
    "text": "Document chunk text content",
    "source": "document_name.pdf",
    "page": 1,
    "section": "Section Title",
    "chunk_id": 0
  },
  ...
]
```

### FAISS Index
- Binary format managed by FAISS library
- Contains float32 vectors with L2 distance metric
- Optimized for fast similarity search

## Backup and Recovery

To backup your vector store:
1. Copy both `faiss_index.bin` and `metadata.json`
2. Store in a safe location
3. Both files are required for complete restoration

To restore:
1. Place both files back in the data directory
2. The vector store service will automatically load them on startup

## Performance Notes

- Index size grows linearly with number of documents
- Search performance is optimized for up to millions of vectors
- Memory usage depends on embedding dimension and document count
- For large datasets, consider using FAISS IVF indices for better performance

## Troubleshooting

### Index/Metadata Mismatch
If you see warnings about mismatched counts:
1. Check that both files are from the same processing run
2. Delete both files to start fresh if corrupted
3. Re-process your documents

### File Permissions
Ensure the application has read/write permissions to this directory.

### Disk Space
Monitor disk usage as the index files can grow large with many documents.
