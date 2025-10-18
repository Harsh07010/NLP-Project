import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import weaviate
import weaviate.auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_elasticsearch import ElasticsearchStore
from langchain_astradb import AstraDBVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever,WikipediaRetriever,ArxivRetriever,TavilySearchAPIRetriever
from langchain_community.vectorstores import Zilliz

class RetrievalMethods:
    def __init__(self):
        print("Loading local embedding model (HuggingFace)...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model loaded.")

    def pinecone_retriever(self, query, documents, index_name="my-langchain-index"):
        api_key = "api"
        os.environ["PINECONE_API_KEY"] = api_key
        
        print(f"\n--- Running Pinecone Retriever (Index: {index_name}) ---")
        vectorstore = PineconeVectorStore.from_texts(
            texts=documents,
            embedding=self.embedding_model,
            index_name=index_name
        )
            
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}
        )
            
        return retriever.invoke(query)


    def weaviate_retriever(self, query, documents, index_name="MyLangChainIndex"):
        weaviate_url = "url"
        weaviate_api_key = "api"

        auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=auth_config
        )

        print(f"\n--- Running Weaviate Retriever (Index: {index_name}) ---")
        
        vectorstore = WeaviateVectorStore.from_texts(
            client=client,
            texts=documents,
            embedding=self.embedding_model,
            index_name=index_name,
            by_text=False 
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
        results = retriever.invoke(query)
        client.close()
        return results

    def elasticsearch_retriever(self, query, documents, index_name="my-langchain-index"):
        elastic_url = "url"
        elastic_api_key = "api"

        print(f"\n--- Running Elasticsearch Retriever (Index: {index_name}) ---")
        vectorstore = ElasticsearchStore.from_texts(
            texts=documents,
            embedding=self.embedding_model,
            es_url=elastic_url,
            es_api_key=elastic_api_key,
            index_name=index_name
        )
            
        retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
        return retriever.invoke(query)
    

    def cohere_rerank_retriever(self, query, documents):
        cohere_api_key = "api"
        os.environ["COHERE_API_KEY"] = cohere_api_key
        
        print("\n--- Running Cohere Rerank Retriever ---")
        base_retriever = BM25Retriever.from_texts(documents)
            
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=4)
            
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

        return compression_retriever.invoke(query)
    
    def astra_db_retriever(self, query, documents, collection_name="my_langchain_collection"):
        api_endpoint = "url"
        token = "api" 
        
        print(f"\n--- Running Astra DB Retriever (Collection: {collection_name}) ---")
        vector_store = AstraDBVectorStore(
            embedding=self.embedding_model,
            collection_name=collection_name,
            api_endpoint=api_endpoint,
            token=token,
        )
        vector_store.add_texts(documents)
        results = vector_store.similarity_search(query, k=4)
        return results

    def zilliz_retriever(self, query, documents, collection_name="my_langchain_collection"):
        cloud_uri = "uri" 
        api_key = "api"

        print(f"\n--- Running Zilliz Cloud Retriever (Collection: {collection_name}) ---")

        connection_args = {'uri': cloud_uri, 'token': api_key}
        
        vector_store = Zilliz.from_texts(
            texts=documents,
            embedding=self.embedding_model,
            collection_name=collection_name,
            connection_args=connection_args,
            auto_id=True  
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': 4})
        results = retriever.invoke(query)
        return results
    
    def wikipedia_retriever(self, query):
        print("\n--- Running Wikipedia Retriever ---")
        retriever = WikipediaRetriever(top_k_results=2, doc_content_chars_max=1000)
        return retriever.invoke(query)

    def arxiv_retriever(self, query):
        print("\n--- Running Arxiv Retriever ---")
        retriever = ArxivRetriever(top_k_results=2, load_max_docs=3)
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

    pinecone_results = retrieval.pinecone_retriever(user_query, docs)
    clean_pinecone_results = [doc.page_content for doc in pinecone_results]
    print("--- Pinecone Result ---")
    print("Pinecone Result:", clean_pinecone_results)
    
    weaviate_results = retrieval.weaviate_retriever(user_query, docs)
    clean_weaviate_results = [doc.page_content for doc in weaviate_results]
    print("--- Weaviate Result ---")
    print("Weaviate Result:", clean_weaviate_results)
    
    elasticsearch_results = retrieval.elasticsearch_retriever(user_query, docs)
    clean_elasticsearch_results = [doc.page_content for doc in elasticsearch_results]
    print("--- Elasticsearch_results ---")
    print("Elasticsearch_results:", clean_elasticsearch_results)

    cohere_results = retrieval.cohere_rerank_retriever(user_query, docs)
    clean_cohere_results = [doc.page_content for doc in cohere_results]
    print("--- Cohere_Results ---")
    print("Cohere_Results:", clean_cohere_results)

    astra_results = retrieval.astra_db_retriever(user_query, docs)
    clean_astra_results = [doc.page_content for doc in astra_results]
    print("\n--- Astra DB Result ---")
    print(clean_astra_results)

    zilliz_results = retrieval.zilliz_retriever(user_query, docs)
    clean_zilliz_results = [doc.page_content for doc in zilliz_results]
    print("\n--- Zilliz Cloud Result ---\n", clean_zilliz_results)

    wiki_query = "History of the Persian cat"
    wikipedia_results = retrieval.wikipedia_retriever(wiki_query)
    clean_wiki_results = [doc.page_content for doc in wikipedia_results]
    print("\n--- Wikipedia Result ---", clean_wiki_results)

    arxiv_query = "Transformers in machine learning"
    arxiv_results = retrieval.arxiv_retriever(arxiv_query)
    clean_arxiv_results = [doc.page_content for doc in arxiv_results]
    print("\n--- Arxiv Result ---", clean_arxiv_results)
