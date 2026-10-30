import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.pipeline import AdvancedRAGPipeline

app = FastAPI(
    title="NexusDocs Advanced RAG API",
    description="Enterprise-grade RAG with Hybrid Search, Semantic Cache, and Pre-filtering.",
    version="2.0.0"
)

pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = AdvancedRAGPipeline()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    filters_applied: dict
    answer: str
    sources: list

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "NexusDocs V2"}

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")
        
    os.makedirs("./temp_uploads", exist_ok=True)
    temp_path = f"./temp_uploads/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        num_chunks = pipeline.ingest(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"filename": file.filename, "status": "indexed", "new_chunks_indexed": num_chunks}

@app.post("/query", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return pipeline.query(request.query)