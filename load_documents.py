import chromadb
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import os
from config import CHROMA_PATH

def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

    sections = re.split(r'\n##+\s', content)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split('\n')
        header = lines[0].strip() if lines else ""
        if not header:
            continue
        current_chunk = [header]
        current_length = len(header)
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            line_length = len(line)
            if current_length + line_length + 1 < 200:
                current_chunk.append(line)
                current_length += line_length + 1
            else:
                chunk_text = '\n'.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                current_chunk = [header, line]
                current_length = len(header) + line_length + 1
        chunk_text = '\n'.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)

    chunks = [chunk for chunk in chunks if chunk and isinstance(chunk, str)]
    print(f"Created {len(chunks)} chunks: {chunks[:2]}...")
    return chunks

def create_embeddings(texts):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [text for text in texts if text and isinstance(text, str)]
        if not texts:
            print("Warning: No valid texts to encode.")
            return np.array([])
        embeddings = model.encode(texts)
        return embeddings
    except Exception as e:
        print(f"Error in create_embeddings: {e}")
        return np.array([])

def initialize_vector_db(knowledge_base):
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        # Delete existing collection to ensure clean state
        try:
            client.delete_collection("knowledge_base")
            print("Deleted existing ChromaDB collection for rebuild.")
        except:
            pass
        collection = client.create_collection("knowledge_base")
        print("Created new ChromaDB collection.")
        embeddings = create_embeddings(knowledge_base)
        if len(embeddings) == 0:
            print("Error: No embeddings created.")
            return collection
        for i, (text, emb) in enumerate(zip(knowledge_base, embeddings)):
            try:
                collection.add(
                    documents=[text],
                    embeddings=[emb.tolist()],
                    ids=[str(i)]
                )
            except Exception as e:
                print(f"Error storing document {i}: {e}")
                continue
        print(f"Stored {len(knowledge_base)} documents in ChromaDB.")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return None

KNOWLEDGE_BASE = load_knowledge_base('carbonjar_knowledgebase.txt')
VECTOR_DB = initialize_vector_db(KNOWLEDGE_BASE)