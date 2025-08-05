from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunker:
    """
    Handles chunking of document text into smaller semantic pieces using token-based splitting.
    
    This class ensures complete document coverage by processing all input blocks without 
    truncation or artificial limits. Uses LangChain's tiktoken-based splitter for accurate
    token counting and semantic preservation.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the DocumentChunker with token-based text splitting.
        
        Args:
            chunk_size (int): Maximum size of each chunk in tokens (default: 500)
            chunk_overlap (int): Number of tokens to overlap between chunks (default: 100)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use tiktoken-based text splitter for accurate token counting
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",  # GPT-4 tokenizer
            model_name="gpt-4",
        )
    
    def chunk_text(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Split document text into token-based chunks with semantic overlap.
        
        This function ensures 100% document coverage by:
        - Fully traversing all input blocks without limits
        - Processing both short (1 page) and long (100+ page) documents
        - Handling empty blocks gracefully without skipping
        - Using token-based splitting for accurate chunking
        - Preserving semantic context through overlap
        
        Args:
            text_blocks (List[Dict]): List of text blocks from document_loader.py
                Expected format: [{"doc_id": str, "page": int, "text": str}, ...]
                
        Returns:
            List[Dict]: Complete list of chunk dictionaries with metadata
                Format: [{"doc_id": str, "page": int, "chunk": str}, ...]
                
        Note:
            - No artificial limits on number of chunks or document length
            - Empty or whitespace-only blocks are handled gracefully
            - All input text is guaranteed to be processed and chunked
        """
        chunks = []
        total_input_chars = 0
        total_processed_chars = 0
        
        print(f"🔍 Starting chunking process for {len(text_blocks)} text blocks...")
        
        # Process every single block in the input list - no limits or early returns
        for block_index, block in enumerate(text_blocks):
            # Extract metadata from each block
            doc_id = block.get("doc_id", "unknown")
            page_num = block.get("page", 1)
            text = block.get("text", "")
            
            # Track input text length
            input_length = len(text)
            total_input_chars += input_length
            
            print(f"   Processing block {block_index + 1}/{len(text_blocks)} (Page {page_num}): {input_length} chars")
            
            # Handle empty blocks gracefully - but still process them
            if not text or not text.strip():
                print(f"     ⚠️ Empty block detected, adding placeholder chunk")
                # Create an entry for empty blocks to maintain document structure
                chunks.append({
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk": ""  # Empty chunk for empty blocks
                })
                continue
            
            # Use tiktoken-based splitter to chunk the text into tokens
            try:
                # Split text into token-limited chunks with semantic preservation
                doc_chunks = self.text_splitter.split_text(text)
                
                print(f"     ✅ Generated {len(doc_chunks)} chunks from this block")
                
                # Process every chunk generated - no truncation or limits
                for chunk_index, chunk_text in enumerate(doc_chunks):
                    # Clean whitespace but preserve the chunk content
                    cleaned_chunk = chunk_text.strip()
                    
                    # Track processed text length
                    total_processed_chars += len(cleaned_chunk)
                    
                    # Add chunk to results (including empty chunks for completeness)
                    chunks.append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "chunk": cleaned_chunk
                    })
                    
                    print(f"       Chunk {chunk_index + 1}: {len(cleaned_chunk)} chars")
                    
            except Exception as e:
                # Handle any chunking errors gracefully while ensuring no data loss
                print(f"❌ Warning: Error chunking block {block_index} from page {page_num}: {str(e)}")
                # Fallback: add the original text as a single chunk
                fallback_chunk = text.strip()
                total_processed_chars += len(fallback_chunk)
                
                chunks.append({
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk": fallback_chunk
                })
                print(f"     🔄 Fallback: Added original text as single chunk ({len(fallback_chunk)} chars)")
        
        # Calculate coverage statistics
        coverage_percent = (total_processed_chars / total_input_chars * 100) if total_input_chars > 0 else 0
        
        print(f"📊 Chunking Summary:")
        print(f"   • Input text blocks: {len(text_blocks)}")
        print(f"   • Output chunks: {len(chunks)}")
        print(f"   • Total input characters: {total_input_chars:,}")
        print(f"   • Total processed characters: {total_processed_chars:,}")
        print(f"   • Coverage: {coverage_percent:.1f}%")
        
        if coverage_percent < 95:
            print(f"⚠️ WARNING: Only {coverage_percent:.1f}% coverage - potential data loss!")
        else:
            print("✅ Excellent coverage - full document processed!")
        
        # Return all chunks - ensuring 100% document coverage
        return chunks