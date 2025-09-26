# utils/embedding_utils.py

import json
from sentence_transformers import SentenceTransformer
import chromadb
import os

# -----------------------------
# 1. Load embedding model
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 2. Initialize ChromaDB with persistence
# -----------------------------
PERSIST_DIR = "embeddings/chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=PERSIST_DIR)

# Check if collection exists, else create
COLLECTION_NAME = "personal_data"
existing_collections = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing_collections:
    collection = client.get_collection(COLLECTION_NAME)
else:
    collection = client.create_collection(COLLECTION_NAME)

# -----------------------------
# 3. Load personal data
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "personal_data.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    personal_data = json.load(f)

# -----------------------------
# 4. Add new documents if not already in DB
# -----------------------------
existing_ids = set(collection.get()["ids"])
new_docs = []
new_ids = []
new_embeddings = []

for idx, item in enumerate(personal_data):
    doc_id = str(idx)
    if doc_id not in existing_ids:
        doc_text = item.get("output", "").strip()
        if doc_text:  # skip empty
            new_docs.append(doc_text)
            new_ids.append(doc_id)
            new_embeddings.append(embed_model.encode(doc_text).tolist())

if new_docs:
    collection.add(documents=new_docs, embeddings=new_embeddings, ids=new_ids)
    print(f"✅ Added {len(new_docs)} new documents to ChromaDB.")
else:
    print("ℹ️ All documents are already embedded in ChromaDB.")

# -----------------------------
# 5. Function to add new data later
# -----------------------------
def add_new_data(new_data_list):
    """
    new_data_list: list of strings
    """
    existing_ids = set(collection.get()["ids"])
    docs_to_add = []
    ids_to_add = []
    embeddings_to_add = []

    start_idx = max([int(i) for i in existing_ids]) + 1 if existing_ids else 0

    for i, doc in enumerate(new_data_list):
        doc_id = str(start_idx + i)
        docs_to_add.append(doc)
        ids_to_add.append(doc_id)
        embeddings_to_add.append(embed_model.encode(doc).tolist())

    if docs_to_add:
        collection.add(documents=docs_to_add, embeddings=embeddings_to_add, ids=ids_to_add)
        print(f"✅ Added {len(docs_to_add)} new documents to ChromaDB.")
    else:
        print("ℹ️ No new documents to add.")
