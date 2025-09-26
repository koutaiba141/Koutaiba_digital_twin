# utils/retrieval.py

from sentence_transformers import SentenceTransformer
import chromadb
import os
import json

# -----------------------------
# 1. Load embedding model
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 2. Load ChromaDB persistent storage
# -----------------------------
PERSIST_DIR = r"C:\Users\USER\Desktop\Digital_Twin\embeddings\chroma_db"
client = chromadb.PersistentClient(path=PERSIST_DIR)

# -----------------------------
# 3. Check collections and create if missing
# -----------------------------
collections = client.list_collections()
print("Available collections:", [c.name for c in collections])

if "personal_data" in [c.name for c in collections]:
    collection = client.get_collection("personal_data")
else:
    collection = client.create_collection("personal_data")
    print("Created collection 'personal_data'")

# -----------------------------
# 4. Load personal_data.json
# -----------------------------
DATA_FILE = r"C:\Users\USER\Desktop\Digital_Twin\data\personal_data.json"
with open(DATA_FILE, "r", encoding="utf-8") as f:
    personal_data = json.load(f)

# -----------------------------
# 5. Add new documents to collection
# -----------------------------
existing_ids = set(collection.get()["ids"])
new_docs = []
new_ids = []
new_embeddings = []

for idx, item in enumerate(personal_data):
    doc_id = str(idx)
    if doc_id not in existing_ids:
        doc_text = item.get("output", "").strip()
        if doc_text:
            new_docs.append(doc_text)
            new_ids.append(doc_id)
            new_embeddings.append(embed_model.encode(doc_text).tolist())

if new_docs:
    collection.add(documents=new_docs, embeddings=new_embeddings, ids=new_ids)
    print(f"Added {len(new_docs)} documents to 'personal_data' collection.")
else:
    print("All documents already exist in collection.")

# -----------------------------
# 6. Function to retrieve documents
# -----------------------------
def retrieve_docs(query, top_k=3):
    query_embedding = embed_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

# -----------------------------
# 7. Test
# -----------------------------
if __name__ == "__main__":
    question = "where do you study"
    docs = retrieve_docs(question, top_k=10)
    print("\nRetrieved documents:\n", docs)
