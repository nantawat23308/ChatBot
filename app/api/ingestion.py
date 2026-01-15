from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from src.services.llm_services import RAGService
import os
import uuid
import shutil

router = APIRouter(prefix="/ingest", tags=["ingestion"])
rag_service = RAGService()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(rag_service.ingest_document, file_path)
    return {
        "status": "Processing started",
        "filename": file.filename,
        "message": "Your document is being embedded and indexed.",
    }
