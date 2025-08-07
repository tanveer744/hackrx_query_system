#!/usr/bin/env python3
"""
Document Processing Pipeline Test
Tests the complete pipeline: PDF processing -> Text extraction -> Chunking
Usage: python tests/test_document_pipeline.py
"""

print("🧪 Testing Document Processing Pipeline")
print("=" * 50)

try:
    # Import services
    from app.services.document_loader import DocumentLoader
    from app.services.chunker import DocumentChunker
    print("✅ Imported document processing services")
    
    # Initialize services  
    loader = DocumentLoader()
    chunker = DocumentChunker()
    print("✅ Initialized services")
    
    # Process the PDF (equivalent to parse_document("sample.pdf"))
    pdf_file = "Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf" 
    print(f"📄 Processing: {pdf_file}")
    
    text = loader.extract_text_from_pdf(pdf_file)  # This is parse_document equivalent
    print(f"✅ Extracted text from {len(text)} blocks")
    
    # Chunk the text (equivalent to chunk_text(text))
    chunks = chunker.chunk_text(text)  # This is chunk_text equivalent
    print(f"✅ Created {len(chunks)} chunks")
    
    # Display results exactly as requested
    print(f"\n📝 Displaying chunks:")
    for i, c in enumerate(chunks):
        chunk_text = c['chunk']  # Extract the chunk text from the dictionary
        print(f"\nChunk {i+1}:\n", chunk_text[:200], "...\n")
        
        # Only show first 5 chunks to avoid too much output
        if i >= 4:
            print(f"... and {len(chunks) - 5} more chunks")
            break
    
    print("🎉 Test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
