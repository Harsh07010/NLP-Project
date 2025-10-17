from langchain_community.retrievers import BM25Retriever, SVMRetriever, TFIDFRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, DocArrayInMemorySearch, Chroma

class RetrievalMethods:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model loaded.")


    def bm25_retriever(self, query, documents):
        retriever = BM25Retriever.from_texts(documents)
        return retriever.invoke(query)
    
    def tfidf_retriever(self, query, documents):
        retriever = TFIDFRetriever.from_texts(documents)
        return retriever.invoke(query)

    def svm_retriever(self, query, documents):
        retriever = SVMRetriever.from_texts(documents, self.embedding_model)
        return retriever.invoke(query)
    
    def faiss_retriever(self, query, documents):
        vectorstore = FAISS.from_texts(documents, self.embedding_model)
        retriever = vectorstore.as_retriever()
        return retriever.invoke(query)

    def docarray_retriever(self, query, documents):
        vectorstore = DocArrayInMemorySearch.from_texts(documents, self.embedding_model)
        retriever = vectorstore.as_retriever()
        return retriever.invoke(query)

    def chroma_retriever(self, query, documents):
        vectorstore = Chroma.from_texts(documents, self.embedding_model)
        retriever = vectorstore.as_retriever()
        return retriever.invoke(query)


if __name__ == '__main__':
    retrieval = RetrievalMethods()
    docs = [
        "The cat sat on the mat.",
        "The cat chased the dog.",
        "The bird flew over the dog.",
        "dog can run fast as lion",
        "the bird sat on the cat",
        "cat was running fast as dog"
    ]
    user_query = "cat"

    print("--- BM25 Results ---")
    bm25_results = retrieval.bm25_retriever(user_query, docs)
    clean_bm25_results = [doc.page_content for doc in bm25_results]
    print("BM25 Result:", clean_bm25_results)

    print("\n--- TF-IDF Results ---")
    tfidf_results = retrieval.tfidf_retriever(user_query, docs)
    clean_tfidf_results = [doc.page_content for doc in tfidf_results]
    print("TF-IDF Results:", clean_tfidf_results)

    print("\n--- SVM Results ---")
    svm_results = retrieval.svm_retriever(user_query, docs)
    clean_svm_results = [doc.page_content for doc in svm_results]
    print("SVM_Results:", clean_svm_results)

    print("\n--- FAISS Results ---")
    faiss_results = retrieval.faiss_retriever(user_query, docs)
    clean_faiss_results = [doc.page_content for doc in faiss_results]
    print("FAISS Result:", clean_faiss_results)

    print("\n--- DocArrayInMemorySearch Results ---")
    docarray_results = retrieval.docarray_retriever(user_query, docs)
    clean_docarray_results = [doc.page_content for doc in docarray_results]
    print("DocArrayInMemorySearch Result:", clean_docarray_results)
        
    print("\n--- Chroma Results ---")
    chroma_results = retrieval.chroma_retriever(user_query, docs)
    clean_chroma_results = [doc.page_content for doc in chroma_results]
    print("Chroma Results:", clean_chroma_results)


    


    
