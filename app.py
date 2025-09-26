# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
from fastapi.responses import FileResponse


# -----------------------------
# 1. FastAPI app
# -----------------------------
app = FastAPI(title="Personal RAG Digital Twin")

# -----------------------------
# 2. Load embedding model & ChromaDB
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

PERSIST_DIR = r"C:\Users\USER\Desktop\Digital_Twin\embeddings\chroma_db"
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Check / create collection
if "personal_data" in [c.name for c in client.list_collections()]:
    collection = client.get_collection("personal_data")
else:
    collection = client.create_collection("personal_data")

# -----------------------------
# 3. Load personal_data.json & add docs if missing
# -----------------------------
DATA_FILE = r"C:\Users\USER\Desktop\Digital_Twin\data\personal_data.json"
with open(DATA_FILE, "r", encoding="utf-8") as f:
    personal_data = json.load(f)

existing_ids = set(collection.get()["ids"])
new_docs, new_ids, new_embeddings = [], [], []

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
    print(f"Added {len(new_docs)} documents to collection.")

# -----------------------------
# 4. Load Flan-T5 Base
# -----------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# -----------------------------
# 5. Define retrieval + generation
# -----------------------------
def retrieve_docs(query, top_k=3):
    query_embedding = embed_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]


def answer_question(question, top_k=3, max_length=300):
    docs = retrieve_docs(question, top_k=top_k)
    context = " ".join(docs)

    input_text = (
        f"You are Koutaiba Diab's personal digital twin. "
        f"Use the retrieved context to answer the question accurately. "
        f"Select the most relevant information from the context and combine answers if needed. "
        f"Remove irrelevant information that does not answer the question. "
        f"If the question is about general knowledge not mentioned in the context, "
        f"answer based on your general knowledge and combine with Koutaiba-specific info if relevant. "
        f"If the question asks 'Who are you?', start with 'I am Koutaiba...' and continue naturally. "
        f"If the question asks 'What is your name?' or 'Whats your name?' and the context contains only an email, "
        f"extract the first and last name from the email and answer naturally. "
        f"Context: {context} "
        f"Question: {question} "
        f"Answer in a natural, clear, first-person way, giving the best possible answer."
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(**inputs, max_length=max_length)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer


# -----------------------------
# 6. FastAPI endpoint
# -----------------------------
class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    answer = answer_question(query.question, top_k=3)
    return {"question": query.question, "answer": answer}


# -----------------------------
# 7. Test route
# -----------------------------
@app.get("/")
def serve_frontend():
    return FileResponse("page.html")  # put your HTML file here