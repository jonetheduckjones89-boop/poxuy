from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os
import uuid
from services.ai_agent import analyze_document, chat_with_document, rewrite_text
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Shadows Medical AI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup absolute paths to avoid CWD issues on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print(f"Starting Shadows Backend...")
print(f"Base Directory: {BASE_DIR}")
print(f"Upload Directory: {UPLOAD_DIR}")

# In-memory storage for results (replace with DB in production)
RESULTS_DB = {}

class ChatPayload(BaseModel):
    jobId: str
    message: str
    history: List[dict]

class RewritePayload(BaseModel):
    text: str
    style: str

@app.get("/")
def read_root():
    return {"status": "Shadows AI Backend is running"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file and triggers AI analysis.
    """
    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Trigger AI Analysis
    # In a real app, this should be a background task (Celery/BullMQ)
    try:
        analysis_result = analyze_document(file_path, file.filename, job_id)
        RESULTS_DB[job_id] = {
            "status": "processed",
            "data": analysis_result,
            "file_path": file_path
        }
        return {"jobId": job_id, "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def get_status(jobId: str):
    if jobId not in RESULTS_DB:
        return {"status": "not_found"}
    return {"status": "processed", "percent": 100}

@app.get("/api/results")
def get_results(jobId: str):
    if jobId not in RESULTS_DB:
        raise HTTPException(status_code=404, detail="Job not found")
    return RESULTS_DB[jobId]["data"]

@app.post("/api/chat")
def chat(payload: ChatPayload):
    if payload.jobId not in RESULTS_DB:
        raise HTTPException(status_code=404, detail="Job not found")
    
    file_path = RESULTS_DB[payload.jobId]["file_path"]
    response = chat_with_document(payload.message, payload.history, file_path)
    
    return {
        "reply": response,
        "sources": ["Document Analysis"]
    }

@app.post("/api/rewrite")
def rewrite(payload: RewritePayload):
    rewritten = rewrite_text(payload.text, payload.style)
    return {"text": rewritten}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
