from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from pathlib import Path
from .ingestion import DOCUMENT_PATH
from .vector_store import index_directory
from .graph import answer_question
import logging
import os
import uuid
from .monitoring import get_metrics
from .retrieval import refresh_bm25_cache

app = FastAPI(title="SentinelAI API")
logger = logging.getLogger(__name__)

@app.on_event("startup")
def initialize_document_index():

    logger.info("Initializing SentinelAI document index...")

    # Synchronize files with Chroma on application startup.
    index_directory(DOCUMENT_PATH.parent)
    refresh_bm25_cache()
    # Load the indexed chunks into the BM25 cache.
    count = refresh_bm25_cache()

    logger.info(
        "SentinelAI ready. BM25 chunks loaded: %d",
        count,
    )
logger = logging.getLogger(__name__)

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    question: str
    answer: list[dict]
    sources: list[dict]



@app.get("/health")
def health():
    return {"status": "ok"}

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.get("/metrics")
def metrics():
    return get_metrics()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, and .pdf files are supported.",
        )

    documents_dir = DOCUMENT_PATH.parent
    documents_dir.mkdir(parents=True, exist_ok=True)

    destination = documents_dir / filename
    if destination.exists():
        destination = documents_dir / (
            f"{destination.stem}_{uuid.uuid4().hex[:8]}{extension}"
        )

    # Write to a temporary file first; don't hold the whole upload in memory.
    temporary_path = documents_dir / f".upload_{uuid.uuid4().hex}.tmp"
    total_bytes = 0

    try:
        with temporary_path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB at a time
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File must be 10 MB or smaller.",
                    )

                output.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        os.replace(temporary_path, destination)
        index_directory(documents_dir)

    except HTTPException:
        temporary_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise

    except Exception:
        temporary_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        logger.exception("Document upload or indexing failed")
        raise HTTPException(
            status_code=500,
            detail="The document could not be saved and indexed. Check the server logs.",
        )

    finally:
        await file.close()

    return {
        "message": "Document uploaded and indexed successfully.",
        "filename": destination.name,
    }

@app.get("/documents")
def list_documents( limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),):
    documents_dir = DOCUMENT_PATH.parent
    supported_extensions = {".txt", ".md", ".pdf"}
    all_files = sorted(
        path for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )
    # files = [
    #     {
    #         "filename": path.name,
    #         "extension": path.suffix.lower(),
    #         "size_bytes": path.stat().st_size,
    #     }
    #     for path in documents_dir.iterdir()
    #     if path.is_file() and path.suffix.lower() in supported_extensions
    # ]

    page = all_files[offset : offset + limit]

    return {
        "total": len(all_files),
        "limit": limit,
        "offset": offset,
        "documents": [
            {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
            for path in page
        ],
    }

from app.vector_store import delete_document

@app.delete("/documents/{filename}")
def remove_document(filename: str):
    safe_name = Path(filename).name
    documents_dir = DOCUMENT_PATH.parent
    file_path = documents_dir / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        delete_document(str(file_path))
        file_path.unlink()
    except Exception:
        logger.exception("Document deletion failed")
        raise HTTPException(
            status_code=500,
            detail="Could not delete the document. Check server logs.",
        )

    return {"message": f"Deleted {safe_name} and its indexed chunks."}

@app.post("/ask",response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = answer_question(request.question)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("SentinelAI failed while answering a question")
        raise HTTPException(
            status_code=500,
            detail="Unable to answer the question. Check the server logs.",
        )

# def ask(request: AskRequest):
#     try:
#         ans=answer_question(request.question)['answer'][0]['text'] if answer_question(request.question)['answer'][0] and isinstance(answer_question(request.question)['answer'][0], dict) else answer_question(request.question)
#         return ans
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc))
#     except Exception:
#         logger.exception("SentinelAI failed while answering a question")
#         raise HTTPException(status_code=500, detail="Unable to answer the question.")