from fastapi import FastAPI
from pydantic import BaseModel
from chromadb import PersistentClient
import ollama
from fastapi.middleware.cors import CORSMiddleware

PERSIST_DIR = "../chroma_db"
COLLECTION_NAME = "documents"
EMBED_MODEL = "embeddinggemma:300m"
LLM_MODEL = "mistral:latest"

client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

def embed(text: str):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]

def retrieve(query: str, k: int = 4):
    query_embedding = embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents"],
    )
    docs = results["documents"][0]
    return "\n".join(docs) if docs else ""

@app.post("/chat")
def chat_rag(req: ChatRequest):
    docs = retrieve(req.query)

    prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant.
Use ONLY the information from the context below to answer.
If the answer is not found in the context, say:
"I don't know based on the stored documents."

Context:
{docs}

Question: {req.query}
Answer:
"""
    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt
    )
    print(f"RAG Response: {response['response']}")
    return {"response": response["response"]}
