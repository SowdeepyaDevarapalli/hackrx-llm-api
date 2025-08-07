import os
import io
import json
import tempfile
import fitz         # PyMuPDF
import pdfplumber
import docx
import requests
from typing import List
import tiktoken
import nltk

from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

import openai
from pinecone import Pinecone

# Download NLTK punkt tokenizer on first run
nltk.download("punkt", quiet=True)

# --- Initialization ---
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# Use your existing Pinecone index name here
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "grand-dogwood")

# Instantiate Pinecone client (new SDK)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Use existing index - no creation here
pc_index = pc.Index(PINECONE_INDEX)


# --- FastAPI ---
app = FastAPI(title="HackRx LLM Query-Retrieval System")


# --- Pydantic Models ---
class HackRxQuery(BaseModel):
    documents: str
    questions: List[str]


class HackRxResponse(BaseModel):
    answers: List[str]


# --- Utilities ---
def download_file(url: str) -> bytes:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content


def read_pdf(content: bytes) -> str:
    import os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(content)
        temp_path = tf.name
    # File is closed after exiting the 'with' block, so we can open it safely
    try:
        with pdfplumber.open(temp_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    finally:
        # Cleanup temp file after reading
        os.remove(temp_path)
    return text

def read_docx(content: bytes) -> str:
    import os
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
        tf.write(content)
        temp_path = tf.name
    try:
        doc = docx.Document(temp_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    finally:
        os.remove(temp_path)
    return text

def read_email(content: bytes) -> str:
    from email import message_from_bytes
    msg = message_from_bytes(content)
    body = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                body.append(part.get_payload(decode=True).decode(errors="ignore"))
    else:
        body.append(msg.get_payload(decode=True).decode(errors="ignore"))
    return "\n".join(body)


def parse_document(url: str) -> str:
    content = download_file(url)
    if url.lower().endswith(".pdf"):
        return read_pdf(content)
    elif url.lower().endswith(".docx"):
        return read_docx(content)
    elif url.lower().endswith(".eml"):
        return read_email(content)
    else:
        # Fallback to PDF parsing
        return read_pdf(content)


# Advanced chunking for retrieval - by clause, heading, list, etc.
def smart_chunk(text: str, max_chunk_tokens: int = 200, overlap: int = 40) -> List[str]:
    """Break text into semi-clauses, favoring headings/lists, for semantic retrieval."""
    paras = [p for p in text.split("\n") if p.strip()]
    chunks = []
    curr = ""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for p in paras:
        if (
            curr
            and (
                p.strip().startswith(tuple("1234567890-•*."))
                or p.isupper()
                or len(enc.encode(curr + p)) > max_chunk_tokens
            )
        ):
            chunks.append(curr.strip())
            curr = ""
        curr += " " + p.strip()
    if curr.strip():
        chunks.append(curr.strip())
    final_chunks = []
    for chunk in chunks:
        sents = nltk.sent_tokenize(chunk)
        buf = ""
        for sent in sents:
            temp = (buf + " " + sent).strip()
            if len(enc.encode(temp)) <= max_chunk_tokens:
                buf = temp
            else:
                if buf:
                    final_chunks.append(buf.strip())
                buf = sent
        if buf:
            final_chunks.append(buf.strip())
    overlapped = []
    for i, ch in enumerate(final_chunks):
        if i > 0:
            prev = final_chunks[i - 1]
            joined = " ".join([*prev.split()[-overlap:], *ch.split()[:overlap]])
            overlapped.append(joined)
        overlapped.append(ch)
    # Unique only
    return list(dict.fromkeys(final_chunks + overlapped))


# Embedding wrapper - using your index's embedding model "text-embedding-3-large"
def get_embeddings(texts: List[str]) -> List[List[float]]:
    emb_model = "text-embedding-3-large"
    resp = openai.embeddings.create(input=texts, model=emb_model)
    return [d.embedding for d in resp.data]



def upsert_chunks(chunks: List[str]) -> List[str]:
    ids = [f"clause-{i}" for i in range(len(chunks))]
    embs = get_embeddings(chunks)
    records = list(zip(ids, embs, [{"text": c} for c in chunks]))
    BATCH_SIZE = 32
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        pc_index.upsert(batch)
    return ids


def semantic_search(query: str, top_k: int = 4):
    emb = get_embeddings([query])[0]
    res = pc_index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [(m["metadata"]["text"], m["id"], m["score"]) for m in res["matches"]]


# OpenAI GPT-4o answer synthesis with clause tracing for explainability
def llm_answer(query: str, context_chunks: List[str]) -> str:
    content = "\n\n".join([f"Clause {i + 1}: {txt.strip()}" for i, txt in enumerate(context_chunks)])
    prompt = (
        "You are a compliance assistant. Use the provided policy/document clauses "
        "to answer the user query factually and briefly. If answer isn't present, say 'Not covered.'\n"
        f"Query:\n{query}\n\n"
        f"Relevant Clauses:\n{content}\n\n"
        "Answer and briefly cite the clause(s)."
    )
    chat_resp = openai.chat.completions.create(
        model="gpt-4o",  # Best combo of cost, speed, accuracy
        messages=[{"role": "system", "content": prompt}],
        temperature=0.1,
        max_tokens=256,
    )
    return chat_resp.choices[0].message.content.strip()


@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(req: Request, authorization: str = Header(None)):
    if not authorization or "Bearer" not in authorization:
        raise HTTPException(401, "Missing or invalid Authorization header")
    payload = await req.json()
    try:
        data = HackRxQuery(**payload)
    except Exception as e:
        raise HTTPException(400, f"Invalid schema: {e}")

    # Parse the document, chunk it smartly, and upsert to Pinecone index
    text = parse_document(data.documents)
    chunks = smart_chunk(text)
    upsert_chunks(chunks)

    # Query Pinecone and generate LLM answers
    answers = []
    for q in data.questions:
        rel_chunks = semantic_search(q, top_k=4)
        ctx_chunks = [c[0] for c in rel_chunks]
        answer = llm_answer(q, ctx_chunks)
        answers.append(answer)

    return {"answers": answers}


# --- For local debugging only ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
