from pathlib import Path
from uuid import uuid4
import json

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.services.asr_service import transcribe_audio
from backend.app.services.audio_service import extract_audio
from backend.app.services.video_service import validate_video


router = APIRouter(
    prefix="/api",
    tags=["Video Upload"],
)


UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIO_DIR = Path("storage/audio")
TRANSCRIPT_DIR = Path("storage/transcripts")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".webm"}


def _looks_like_video_signature(header: bytes, extension: str) -> bool:
    if not header:
        return False
    if extension == ".mp4":
        return b"ftyp" in header[:64] or header.startswith(b"\x00\x00\x00")
    if extension == ".webm":
        return header.startswith(b"\x1aE\xdf\xa3") or b"EBML" in header[:32]
    return False


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):
    """Upload and validate a video against both extension and content signatures."""
    raw_filename = file.filename or ""
    original_name = Path(raw_filename).name or ""
    if not original_name:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file name.",
        )

    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .mp4 and .webm files are supported.",
        )

    await file.seek(0)
    header = await file.read(512)
    if not _looks_like_video_signature(header, extension):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file does not match a valid MP4 or WebM video signature.",
        )
    await file.seek(0)

    job_id = str(uuid4())
    safe_filename = f"{job_id}{extension}"
    output_path = UPLOAD_DIR.resolve(strict=False) / safe_filename
    if output_path.exists():
        output_path.unlink(missing_ok=True)

    total_bytes = 0
    try:
        with output_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
                            "size limit for local MVP processing."
                        ),
                    )
                buffer.write(chunk)
    except HTTPException:
        output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save upload: {exc}",
        ) from exc

    try:
        metadata = validate_video(output_path)
    except ValueError as exc:
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {exc}",
        ) from exc

    audio_path = AUDIO_DIR / f"{job_id}.wav"
    transcript_path = TRANSCRIPT_DIR / f"{job_id}.json"

    try:
        audio_path = await run_in_threadpool(
            extract_audio,
            output_path,
            job_id,
        )
    except Exception as exc:
        audio_path.unlink(missing_ok=True)
        transcript_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Audio extraction failed: {exc}",
        ) from exc

    try:
        transcript = await run_in_threadpool(
            transcribe_audio,
            audio_path,
        )

        with transcript_path.open("w", encoding="utf-8") as transcript_file:
            json.dump(
                transcript,
                transcript_file,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as exc:
        audio_path.unlink(missing_ok=True)
        transcript_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        ) from exc

    return {
        "job_id": job_id,
        "status": "transcribed",
        "filename": original_name,
        "stored_filename": safe_filename,
        "duration": metadata["duration"],
        "metadata": metadata["metadata"],
        "transcript": transcript,
    }