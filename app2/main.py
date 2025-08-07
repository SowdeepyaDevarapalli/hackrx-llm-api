from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="LLM-Powered Document Query API")

class QueryRequest(BaseModel):
    doc_id: str
    question: str

class QueryResponse(BaseModel):
    decision: str
    amount: float
    justification: str
    references: List[str]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # TODO: implement document ingestion
    return {"doc_id": "sample-doc-id"}

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    # TODO: implement query handling
    return {
        "decision": "Approved",
        "amount": 100000.0,
        "justification": "Sample justification referencing clause 5.2",
        "references": ["doc_id:clause_5.2"]
    }
