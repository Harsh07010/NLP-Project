import time
import uuid
from datetime import datetime, timedelta
from langchain_core.documents import Document
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
    TimeWeightedVectorStoreRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_core.prompts import PromptTemplate



class AdvancedRetrievalMethods:
    def __init__(self):
        print("Loading local embedding model (HuggingFace)...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model loaded.")
        
        try:
            self.llm = ChatOllama(model="llama3", temperature=0)
            print("LLM (Ollama) initialized.")
        except Exception as e:
            self.llm = None
            print(f"Could not initialize Ollama LLM. MultiQuery and SelfQuery will not work. Error: {e}")

    def ensemble_retriever(self, query, documents):
        print("\n--- Running Ensemble Retriever ---")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 2
        faiss_vectorstore = FAISS.from_documents(documents, self.embedding_model)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        return ensemble_retriever.invoke(query)

    def mmr_retriever(self, query, documents):
        print("\n--- Running MMR Retriever ---")
        vectorstore = FAISS.from_documents(documents, self.embedding_model)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'fetch_k': 10})
        return retriever.invoke(query)

    def timeweighted_retriever(self, query, time_docs):
        print("\n--- Running Time-Weighted Retriever ---")
        vectorstore = FAISS.from_documents(time_docs, self.embedding_model)
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, decay_rate=0.0001, k=2
        )
        return retriever.invoke(query)

    def multiquery_retriever(self, query, documents):
        if not self.llm:
            return "Skipped: Ollama LLM not available."
        print("\n--- Running Multi-Query Retriever ---")
        vectorstore = FAISS.from_documents(documents, self.embedding_model)
        base_retriever = vectorstore.as_retriever()
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=self.llm
        )
        return retriever.invoke(query)
    
    def contextual_compression_retriever(self, query, documents):
        if not self.llm:
            return "Skipped: Ollama LLM not available for LLMChainFilter."
        print("\n--- Running Contextual Compression Retriever ---")
        base_retriever = FAISS.from_documents(documents, self.embedding_model).as_retriever()

        llm_filter = LLMChainFilter.from_llm(self.llm)
    
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=llm_filter, base_retriever=base_retriever
        )
        return compression_retriever.invoke(query)

    def parent_document_retriever(self, parent_docs):
        print("\n--- Running Parent Document Retriever ---")
        store = InMemoryStore()
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        vectorstore = Chroma(collection_name="split_parents", embedding_function=self.embedding_model)
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
        retriever.add_documents(parent_docs)
        return retriever.invoke("What are the key themes in AI development?")
    

    def multivector_retriever(self, parent_docs):
        print("\n--- Running Multi-Vector Retriever ---")
        store = InMemoryStore()
        doc_ids = [str(uuid.uuid4()) for _ in parent_docs]
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        sub_docs = []
        for i, doc in enumerate(parent_docs):
            _id = doc_ids[i]
            _sub_docs = child_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata["doc_id"] = _id
            sub_docs.extend(_sub_docs)
        vectorstore = Chroma.from_documents(sub_docs, self.embedding_model)
        store.mset(list(zip(doc_ids, parent_docs)))
        retriever = MergerRetriever(retrievers=[vectorstore.as_retriever()])
        return retriever.invoke("What is the role of large language models?")

    def long_context_reorder_retriever(self, query, documents):
        print("\n--- Running Long Context Reorder ---")
        retriever = FAISS.from_documents(documents, self.embedding_model).as_retriever(search_kwargs={"k": 5})
        initial_results = retriever.invoke(query)
        reorderer = LongContextReorder()
        reordered_docs = reorderer.transform_documents(initial_results)
        return reordered_docs

if __name__ == '__main__':
    advanced_retrieval = AdvancedRetrievalMethods()
    
    docs = [
        Document(page_content="The cat sat on the mat."),
        Document(page_content="The cat chased the dog."),
        Document(page_content="The bird flew over the dog."),
        Document(page_content="AI is transforming industries."),
        Document(page_content="Deep learning is a subset of AI."),
    ]
    query = "cat" 

    # 1. Ensemble
    ensemble_results = advanced_retrieval.ensemble_retriever(query, docs)
    print("Ensemble Results:", [doc.page_content for doc in ensemble_results])

    # 2. MMR
    mmr_results = advanced_retrieval.mmr_retriever(query, docs)
    print("MMR Results:", [doc.page_content for doc in mmr_results])
    
    # 3. Time-Weighted
    time_docs = [
        Document(page_content="AI summit happened last week", metadata={"last_accessed_at": datetime.now() - timedelta(days=1)}),
        Document(page_content="Old AI research paper from 2022", metadata={"last_accessed_at": datetime.now() - timedelta(days=730)}),
        Document(page_content="Recent breakthrough in AI", metadata={"last_accessed_at": datetime.now() - timedelta(hours=2)}),
    ]
    tw_results = advanced_retrieval.timeweighted_retriever("recent AI news", time_docs)
    print("Time-Weighted Results:", [doc.page_content for doc in tw_results])

    # 4. MultiQuery
    mq_results = advanced_retrieval.multiquery_retriever("What is artificial intelligence?", docs)
    if isinstance(mq_results, str): print(mq_results)
    else: print("Multi-Query Results:", [doc.page_content for doc in mq_results])
    
    # 5. Contextual Compression
    cc_results = advanced_retrieval.contextual_compression_retriever(query, docs)
    print("Contextual Compression Results:", [doc.page_content for doc in cc_results])
    
    # 6. Parent Document
    parent_docs = [
        Document(page_content="AI development is rapidly accelerating. Large language models (LLMs) like GPT are a key theme, capable of generating human-like text."),
        Document(page_content="Another major trend is generative art. Models like Stable Diffusion can create stunning visuals from simple text prompts, raising questions about creativity."),
        Document(page_content="Ethical considerations are paramount. Ensuring fairness, transparency, and accountability in AI systems is a challenge that researchers and policymakers are actively addressing.")
    ]
    pd_results = advanced_retrieval.parent_document_retriever(parent_docs)
    print("Parent Document Results:", [doc.page_content for doc in pd_results])

    # 7. Multi-Vector
    mv_results = advanced_retrieval.multivector_retriever(parent_docs)
    print("Multi-Vector Results:", [doc.page_content for doc in mv_results])
    
    # 8. Long Context Reorder
    lcr_results = advanced_retrieval.long_context_reorder_retriever(query, docs)
    print("Long Context Reorder Results:", [doc.page_content for doc in lcr_results])

