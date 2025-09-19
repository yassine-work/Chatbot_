from load_documents import VECTOR_DB
from sentence_transformers import SentenceTransformer

def retrieve_relevant_docs(query, top_k=2):
    if VECTOR_DB is None:
        print("Error: VECTOR_DB not initialized.")
        return []
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = model.encode([query])[0].tolist()
        print(f"Query: {query}, Embedding shape: {len(query_emb)}")
        results = VECTOR_DB.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        print(f"Retrieval results: {results}")
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"Error in retrieve_relevant_docs: {e}")
        return []