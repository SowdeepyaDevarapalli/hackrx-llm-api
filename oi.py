import os
import uuid
import tempfile
import json
from typing import List

import spacy
import PyPDF2
import docx2txt
import openai
import pinecone
from pydantic import BaseModel, ValidationError

# ==== Configuration ====

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = "doc-query-index"

openai.api_key = OPENAI_API_KEY

# Create or connect to Pinecone index
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=1536)  # text-embedding-ada-002 dimension

index = pinecone.Index(INDEX_NAME)

# Load spaCy for NLP chunking
nlp = spacy.load("en_core_web_sm")

# ==== Data Models ====

class QueryResponse(BaseModel):
    decision: str
    amount: float
    justification: str
    references: List[str]


# ==== Utils: Parse Documents ====

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    text = docx2txt.process(file_path)
    return text.strip()

def chunk_text(text: str, max_chunk_tokens=500) -> List[str]:
    """
    Splits text into smaller chunks (~max_chunk_tokens tokens)
    on sentence boundaries using spaCy.
    """
    doc = nlp(text)
    chunks = []
    chunk = ""
    for sent in doc.sents:
        if len(chunk.split()) + len(sent.text.split()) > max_chunk_tokens:
            if chunk:
                chunks.append(chunk.strip())
            chunk = sent.text
        else:
            chunk += " " + sent.text
    if chunk:
        chunks.append(chunk.strip())
    return chunks


# ==== Utils: Embeddings & Pinecone ====

def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = response['data'][0]['embedding']
    return embedding

def upsert_chunks(doc_id: str, chunks: List[str]) -> None:
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        vectors.append(
            (f"{doc_id}_chunk_{i}", emb, {"text": chunk, "doc_id": doc_id, "chunk_index": i})
        )
    index.upsert(vectors)
    print(f"Upserted {len(chunks)} chunks for doc_id {doc_id}")


# ==== Retrieval ====

def query_index(doc_id: str, question: str, top_k=5) -> List[str]:
    q_embed = get_embedding(question)
    results = index.query(q_embed, top_k=top_k, filter={"doc_id": doc_id}, include_metadata=True)
    chunks = []
    for match in results['matches']:
        chunks.append(match['metadata']['text'])
    return chunks


# ==== Prompt Engineering & LLM Interaction ====


PROMPT_TEMPLATE = """
You are an intelligent assistant that answers questions based on the provided document text chunks.

Question:
{question}

Text Chunks:
{context}

Rules:
- Answer strictly in JSON with these fields: 
  - decision (Approved/Rejected), 
  - amount (float), 
  - justification (string), 
  - references (list of strings).
- Use exact clauses or sections from the text chunks in justification and references.
- If unsure, answer 'Rejected' with a clear reason.

Provide only JSON output, no additional text.
"""


def generate_answer(question: str, chunks: List[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a precise and logical assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
    )
    answer = response['choices'][0]['message']['content']
    return answer


def parse_llm_response(response: str) -> QueryResponse:
    try:
        data = json.loads(response)
        validated = QueryResponse.parse_obj(data)
        return validated
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid LLM response: {e}\nResponse text: {response}")


# ==== Full Pipeline Functions ====

def ingest_document(file_path: str) -> str:
    """
    Parse document, chunk and embed it into Pinecone.
    Returns unique document ID.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if not text:
        raise ValueError("No text extracted from document")

    chunks = chunk_text(text)
    
    doc_id = str(uuid.uuid4())
    upsert_chunks(doc_id, chunks)
    return doc_id

def answer_query(doc_id: str, question: str) -> QueryResponse:
    chunks = query_index(doc_id, question)
    if not chunks:
        raise ValueError("No relevant document chunks found")

    llm_raw = generate_answer(question, chunks)
    result = parse_llm_response(llm_raw)
    return result

# ==== Command-line testing example ====

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Document Query System")
    parser.add_argument("--file", type=str, help="Path to document file (pdf/docx) to ingest")
    parser.add_argument("--question", type=str, help="Question to ask on the last ingested doc")
    parser.add_argument("--doc_id", type=str, default=None, help="Optional doc_id to query")
    args = parser.parse_args()

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("ERROR: Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables.")
        return

    # Ingest document if provided
    if args.file:
        print(f"[*] Ingesting document: {args.file}")
        doc_id = ingest_document(args.file)
        print(f"Document ingested with doc_id: {doc_id}")
    else:
        doc_id = args.doc_id
        if not doc_id:
            print("ERROR: Must provide either --file to ingest or --doc_id to query.")
            return

    # Query if question provided
    if args.question:
        print(f"[*] Querying document ID {doc_id} with question:\n{args.question}")
        try:
            response = answer_query(doc_id, args.question)
            print("\n[+] Answer:")
            print(response.json(indent=2))
        except Exception as e:
            print(f"[!] Query error: {e}")
    else:
        print("No question provided, exiting.")

if __name__ == "__main__":
    main()
