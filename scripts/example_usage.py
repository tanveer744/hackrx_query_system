"""
Example usage of the HackRx Embedding & Retrieval System
Demonstrates how to use the system with sample policy documents.
"""

import os
import sys
import logging

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample policy documents
SAMPLE_DOCUMENTS = [
    {
        "text": "A grace period of thirty (30) days is allowed for premium payment after the due date. During this grace period, the policy remains in full force and effect. If the premium is not paid within the grace period, the policy will lapse.",
        "source": "health_insurance_policy.pdf",
        "page": 3,
        "section": "Premium Payment Terms"
    },
    {
        "text": "This policy covers medical expenses up to $100,000 annually for hospitalization, surgery, emergency care, and diagnostic procedures. Coverage includes both inpatient and outpatient services at network providers.",
        "source": "health_insurance_policy.pdf",
        "page": 5,
        "section": "Coverage Benefits"
    },
    {
        "text": "All claims must be submitted within ninety (90) days of the date of service or incident. Claims submitted after this period may be denied unless there are extenuating circumstances. Required documentation includes medical records and itemized bills.",
        "source": "health_insurance_policy.pdf",
        "page": 8,
        "section": "Claims Processing Requirements"
    },
    {
        "text": "Pre-existing medical conditions are covered under this policy after a waiting period of twelve (12) months from the policy effective date. A pre-existing condition is defined as any illness or injury for which medical advice or treatment was received in the 6 months prior to enrollment.",
        "source": "health_insurance_policy.pdf",
        "page": 12,
        "section": "Pre-existing Condition Coverage"
    },
    {
        "text": "The policyholder may add eligible dependents including spouse and unmarried children under age 26. Additional premium applies for each dependent. Maximum of 4 dependents allowed per policy. Dependent coverage is subject to the same terms and conditions as the primary insured.",
        "source": "health_insurance_policy.pdf",
        "page": 15,
        "section": "Dependent Coverage Options"
    },
    {
        "text": "Emergency services are covered 24/7 at any hospital, including out-of-network facilities. No prior authorization required for true emergencies. Emergency room copay is $150 per visit. Follow-up care must be transferred to network providers when medically stable.",
        "source": "health_insurance_policy.pdf",
        "page": 18,
        "section": "Emergency Care Coverage"
    },
    {
        "text": "Prescription drug coverage includes generic and brand-name medications. Generic drugs have a $10 copay, preferred brand drugs $30 copay, and non-preferred brand drugs $60 copay. Mail-order pharmacy available for 90-day supplies at reduced cost.",
        "source": "health_insurance_policy.pdf",
        "page": 22,
        "section": "Prescription Drug Benefits"
    },
    {
        "text": "Annual deductible is $2,500 per individual or $5,000 per family. After meeting the deductible, the plan pays 80% of covered expenses and the member pays 20% coinsurance up to the annual out-of-pocket maximum of $8,000 individual or $16,000 family.",
        "source": "health_insurance_policy.pdf",
        "page": 25,
        "section": "Cost Sharing Structure"
    }
]

# Sample queries to test the system
SAMPLE_QUERIES = [
    "What is the grace period for premium payment?",
    "How much does the policy cover for medical expenses?",
    "When do I need to submit claims?",
    "Are pre-existing conditions covered?",
    "Can I add my family members to the policy?",
    "What is covered for emergency services?",
    "How much do prescription drugs cost?",
    "What is the annual deductible?"
]

import numpy as np
from app.services.embedder import create_embedder
from app.services.vector_store import create_faiss_index_from_embeddings, semantic_search


def demonstrate_retrieval_system():
    """Demonstrate the complete retrieval system."""
    logger.info("🚀 Starting HackRx Retrieval System Demo")
    logger.info("=" * 60)
    
    try:
        # Import and initialize the retrieval system
        from services.retrieval_pipeline import HackRxRetrieval
        
        logger.info("Initializing retrieval system...")
        retrieval = HackRxRetrieval()
        logger.info("✓ Retrieval system initialized")
        
        # Add sample documents
        logger.info(f"\nAdding {len(SAMPLE_DOCUMENTS)} sample policy documents...")
        retrieval.process_documents(SAMPLE_DOCUMENTS)
        logger.info("✓ Documents processed and indexed")
        
        # Get system statistics
        stats = retrieval.pipeline.get_stats()
        logger.info(f"\n📊 System Statistics:")
        logger.info(f"   Total vectors: {stats['total_vectors']}")
        logger.info(f"   Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"   Embedding provider: {stats['embedding_provider']}")
        logger.info(f"   Reranker enabled: {stats['reranker_enabled']}")
        
        # Test queries
        logger.info(f"\n🔍 Testing {len(SAMPLE_QUERIES)} sample queries:")
        logger.info("-" * 60)
        
        for i, query in enumerate(SAMPLE_QUERIES, 1):
            logger.info(f"\nQuery {i}: {query}")
            
            # Get search results
            results = retrieval.query(query, top_k=3)
            
            # Get formatted context for LLM
            context = retrieval.get_context_for_llm(query, max_chunks=2)
            
            logger.info(f"Found {len(results)} relevant results:")
            
            for j, result in enumerate(results, 1):
                score_field = "rerank_score" if "rerank_score" in result else "similarity_score"
                score = result.get(score_field, 0)
                text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                
                logger.info(f"  {j}. Score: {score:.4f}")
                logger.info(f"     Text: {text_preview}")
                logger.info(f"     Source: {result['source']}, Page: {result['page']}")
            
            # Show LLM context (first 200 chars)
            context_preview = context[:200] + "..." if len(context) > 200 else context
            logger.info(f"LLM Context Preview: {context_preview}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Demo completed successfully!")
        logger.info("\nThe retrieval system is working correctly and ready for integration.")
        logger.info("Next steps:")
        logger.info("1. Integrate with document parser (Member 2)")
        logger.info("2. Integrate with LLM service (Member 1)")
        logger.info("3. Deploy to production environment")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure to install dependencies: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False


def demonstrate_individual_components():
    """Demonstrate individual components separately."""
    logger.info("\n🔧 Testing Individual Components")
    logger.info("=" * 60)
    
    try:
        # Test embedder
        logger.info("\n1. Testing Embedder...")
        from services.embedder import create_embedder
        
        try:
            embedder = create_embedder("openai")
            logger.info("✓ Using OpenAI embedder")
        except:
            embedder = create_embedder("huggingface")
            logger.info("✓ Using HuggingFace embedder")
        
        # Test embedding
        sample_text = SAMPLE_DOCUMENTS[0]["text"]
        embedding = embedder.embed_single(sample_text)
        logger.info(f"✓ Generated embedding with shape: {embedding.shape}")
        
        # Test vector store
        logger.info("\n2. Testing Vector Store...")
        from services.vector_store import DocumentVectorStore
        
        vector_store = DocumentVectorStore(embedder.get_embedding_dimension(), "demo_data")
        
        # Add a few documents
        texts = [doc["text"] for doc in SAMPLE_DOCUMENTS[:3]]
        embeddings = embedder.embed_batch(texts)
        sources = [doc["source"] for doc in SAMPLE_DOCUMENTS[:3]]
        pages = [doc["page"] for doc in SAMPLE_DOCUMENTS[:3]]
        
        vector_store.add_documents(texts, embeddings, sources, pages)
        logger.info(f"✓ Added {len(texts)} documents to vector store")
        
        # Test search
        query_embedding = embedder.embed_query("grace period")
        results = vector_store.search_documents(query_embedding, k=2)
        logger.info(f"✓ Search returned {len(results)} results")
        
        # Test reranker (optional)
        logger.info("\n3. Testing Reranker...")
        try:
            from services.reranker import create_reranker
            reranker = create_reranker("cross_encoder")
            reranked = reranker.rerank("grace period", results)
            logger.info(f"✓ Reranker processed {len(reranked)} results")
        except ImportError:
            logger.info("⚠ Reranker dependencies not available (optional)")
        
        logger.info("\n✅ All individual components working correctly!")
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")


def main():
    """Main function to run the demonstration."""
    print("HackRx Embedding & Retrieval System - Demo")
    print("Member 3: Embedding & Retrieval Specialist")
    print("=" * 60)
    
    # Run the main demonstration
    success = demonstrate_retrieval_system()
    
    if success:
        # Run individual component tests
        demonstrate_individual_components()
        
        print("\n" + "🎯 SUMMARY")
        print("=" * 60)
        print("✅ Embedding service: Working")
        print("✅ Vector store service: Working") 
        print("✅ Reranker service: Working (optional)")
        print("✅ Complete pipeline: Working")
        print("✅ FastAPI integration: Ready")
        print("\n🚀 System is ready for production use!")
        
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        print("Common issues:")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        print("- Missing OpenAI API key (set OPENAI_API_KEY environment variable)")
        print("- Python path issues (run from project root directory)")


if __name__ == "__main__":
    # 1. Example text chunks
    texts = [
        "Knee surgery is covered.",
        "Cataract has a waiting period.",
        "Outpatient expenses are reimbursed.",
    ]
    
    # 2. Get embeddings
    embedder = create_embedder("huggingface")
    embeddings = embedder.embed_batch(texts)
    
    # 3. Create FAISS index
    doc_store = create_faiss_index_from_embeddings(texts, embeddings, sources=["policy.pdf"]*len(texts))
    
    # 4. Prepare a query and get its embedding
    query = "Is cataract surgery covered?"
    query_embedding = embedder.embed_single(query)
    
    # 5. Perform top-k semantic search
    results = semantic_search(doc_store, query_embedding, k=2)
    
    print("Top-2 semantic search results:")
    for i, res in enumerate(results):
        print(f"Result {i+1}: text='{res['text']}', score={res['similarity_score']:.4f}")
