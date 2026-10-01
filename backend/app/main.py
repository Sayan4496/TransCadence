import logging

from fastapi import FastAPI

from backend.app.config import gemini_api_key_status
from backend.app.api.upload import router as upload_router
from backend.app.api.transcribe import router as transcribe_router
from backend.app.api.translate import router as translate_router
from backend.app.api.jobs import router as jobs_router


logger = logging.getLogger(__name__)
logger.info("Gemini API key configuration: %s", gemini_api_key_status())


app = FastAPI(
    title="TransCadence API",
    description=(
        "AI Automated Video Lecture Dubbing "
        "& Synchronized Subtitle System"
    ),
    version="0.1.0",
)


app.include_router(upload_router)
app.include_router(transcribe_router)
app.include_router(translate_router)
app.include_router(jobs_router)



@app.get("/")
def root():
    return {
        "name": "TransCadence",
        "status": "running",
        "message": "AI Video Dubbing API",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }