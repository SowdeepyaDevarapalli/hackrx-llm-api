########################### 1. Imports & Configs ###########################
import os
import re
import json
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.llms import OpenAI
from pinecone import Pinecone  # Updated import for serverless
import requests
from pdfminer.high_level import extract_text as pdf_extract
import docx
from email import policy
from email.parser import BytesParser

# --- Config ---
PINECONE_API_KEY = "<your-pinecone-api-key>"
PINECONE_INDEX = "grand-dogwood"  # Your actual index name
OPENAI_API_KEY = "<your-openai-api-key>"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Pinecone (serverless - no environment needed)
pc = Pinecone(api_key=PINECONE_API_KEY)

########################### 2. FastAPI Models ###########################

class QueryInput(BaseModel):
    documents: str   # Blob URL
    questions: List[str]

class QueryOutput(BaseModel):
    answers: List[str]

########################### 3. Utility: Document Loaders ###########################

def download_file(url: str, filename: str) -> str:
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)
    return filename

def extract_pdf_text(filepath: str) -> str:
    return pdf_extract(filepath)

def extract_docx_text(filepath: str) -> str:
    doc = docx.Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_email_text(filepath: str) -> str:
    with open(filepath, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return msg.get_body(preferencelist=('plain')).get_content()

def load_and_extract_text(blob_url: str) -> str:
    # Choose filename based on extension
    ext = blob_url.split("?")[0].split(".")[-1].lower()
    tmp_name = str(uuid.uuid4()) + "." + ext
    download_file(blob_url, tmp_name)
    if ext == "pdf":
        return extract_pdf_text(tmp_name)
    elif ext == "docx":
        return extract_docx_text(tmp_name)
    elif ext in ("eml", "msg"):
        return extract_email_text(tmp_name)
    else:
        raise Exception("Unsupported file format: " + ext)

########################### 4. Chunking, Embedding & Indexing ###########################

def chunk_and_embed(
    text: str, 
    max_chunk_size: int = 600, 
    overlap: int = 60
) -> List[Dict[str, Any]]:
    # Split text into overlapping chunks for context preservation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=overlap
    )
    chunks = splitter.split_text(text)
    return [
        {"id": str(uuid.uuid4()), "text": chunk.strip(), "metadata": {}} 
        for chunk in chunks if chunk.strip()
    ]

def pinecone_upsert_chunks(chunks: List[Dict], namespace: str = "default"):
    # Initialize embeddings with text-embedding-3-large
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"  # Full 3072 dimensions to match your index
    )
    
    # Get your index (serverless)
    index = pc.Index(PINECONE_INDEX)
    
    to_upsert = []
    for ch in chunks:
        emb = embeddings.embed_query(ch["text"])
        to_upsert.append((ch["id"], emb, {"text": ch["text"], **ch.get("metadata", {})}))
        if len(to_upsert) == 100:
            index.upsert(to_upsert, namespace=namespace)
            to_upsert = []
    if to_upsert:
        index.upsert(to_upsert, namespace=namespace)

def pinecone_retrieve(
    query: str, 
    namespace: str = "default",
    top_k: int = 7
) -> List[Dict]:
    # Initialize embeddings with text-embedding-3-large
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"  # Full 3072 dimensions to match your index
    )
    
    index = pc.Index(PINECONE_INDEX)
    emb = embeddings.embed_query(query)
    results = index.query(vector=emb, namespace=namespace, top_k=top_k, include_metadata=True)
    return [
        {"id": m["id"], "text": m["metadata"]["text"], "score": m["score"]} 
        for m in results["matches"]
    ]

########################### 5. Explainable Clause Matching & Logic ###########################

def extract_structured_query(question: str) -> str:
    # Use LLM to extract core query intent ("operation") and constraints
    prompt = (
        "Given the following user query, extract the key decision point and any conditions being inquired about, phrased as a single decision (e.g., 'policy covers knee surgery under these conditions'). "
        f"Query: {question}\nSummary:"
    )
    llm = OpenAI(model="gpt-3.5-turbo-instruct")
    return llm(prompt).strip()

def score_and_select_clauses(
    query: str, 
    retrieved_chunks: List[Dict], 
    top_n: int = 3
) -> List[Dict]:
    # Run similarity scoring and pick top_n candidates for rationale building
    # (In production, a cross-encoder/LLM reranker can further enhance quality!)
    return sorted(retrieved_chunks, key=lambda x: -x["score"])[:top_n]

def llm_decision_and_explanation(question: str, clauses: List[Dict]) -> str:
    # Provide the answer, rationale, and traceability (with clause excerpts)
    prompt = (
        "A user asks the following question about an insurance/legal/HR policy:\n"
        f"Question: \"{question}\"\n"
        "Relevant Policy Clauses:\n"
        + "\n---\n".join([f'Clause {i+1}: "{cl["text"]}"' for i, cl in enumerate(clauses)])
        + "\n\n"
        "Based on the above, provide a direct answer, referencing which clause supports your answer, and include a rationale in clear natural language. End with a numbered list of traced clause references."
    )
    llm = OpenAI(temperature=0.2, max_tokens=384, model="gpt-4")
    return llm(prompt).strip()

########################### 6. Pipeline & API ###########################

app = FastAPI(
    title="LLM-powered Query–Retrieval System",
    description="Handles complex insurance/legal queries with full explainability, clause matching, and efficient token usage."
)

@app.post("/api/v1/hackrx/run", response_model=QueryOutput)
def run_query(
    payload: QueryInput, 
    Authorization: str = Header(None)
):
    # Auth
    if Authorization != "Bearer 52270cf7813e09f935c6ec5fb06e9607b4f3aace553501f05ffc4acfa840a654":
        raise HTTPException(status_code=401, detail="Invalid or missing token.")

    # Ingest & Index (done freshly per run for demo; use persistent DB in prod)
    document_text = load_and_extract_text(payload.documents)
    chunks = chunk_and_embed(document_text)
    namespace = str(uuid.uuid5(uuid.NAMESPACE_URL, payload.documents))
    pinecone_upsert_chunks(chunks, namespace=namespace)

    answers = []
    for question in payload.questions:
        core_query = extract_structured_query(question)  # LLM statement
        retrieved = pinecone_retrieve(core_query, namespace=namespace)
        top_clauses = score_and_select_clauses(core_query, retrieved, top_n=3)
        answer = llm_decision_and_explanation(question, top_clauses)
        answers.append(answer)

    return QueryOutput(answers=answers)

########################### 7. If running standalone ###########################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)