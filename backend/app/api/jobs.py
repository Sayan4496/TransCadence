"""Job processing, status, and safe final-video download routes."""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.services.job_processing_service import (
    JobProcessingError,
    get_job_status,
    process_job,
)
from backend.app.services.video_generation_service import OUTPUT_ROOT


router = APIRouter(prefix="/api/jobs", tags=["Job Processing"])


def _canonical_job_id(job_id: str) -> str:
    try:
        return str(UUID(job_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail="Job not found.") from None


@router.post("/{job_id}/process")
def process_job_route(job_id: str) -> dict:
    safe_job_id = _canonical_job_id(job_id)
    try:
        return process_job(safe_job_id)
    except JobProcessingError as exc:
        detail = {"stage": exc.stage, "message": exc.message}
        if exc.segment_id is not None:
            detail["segment_id"] = exc.segment_id
        raise HTTPException(status_code=500, detail=detail) from None
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found.") from None


@router.get("/{job_id}/status")
def job_status_route(job_id: str) -> dict:
    safe_job_id = _canonical_job_id(job_id)
    try:
        return get_job_status(safe_job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found.") from None
    except Exception:
        raise HTTPException(status_code=500, detail="Job status is unavailable.") from None


@router.get("/{job_id}/download")
def download_job_video(job_id: str) -> FileResponse:
    safe_job_id = _canonical_job_id(job_id)
    try:
        job_status = get_job_status(safe_job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Final video not found.") from None
    except Exception:
        raise HTTPException(status_code=500, detail="Job status is unavailable.") from None
    if job_status.get("status") != "completed" or not job_status.get("output_ready"):
        raise HTTPException(status_code=404, detail="Final video not found.")

    job_directory = (OUTPUT_ROOT / safe_job_id).resolve(strict=False)
    expected_job_directory = OUTPUT_ROOT.resolve(strict=False) / safe_job_id

    if job_directory != expected_job_directory:
        raise HTTPException(status_code=404, detail="Final video not found.")
    try:
        job_directory.relative_to(OUTPUT_ROOT.resolve(strict=False))
    except ValueError:
        raise HTTPException(status_code=404, detail="Final video not found.") from None

    output_path = job_directory / "hindi_dubbed.mp4"
    try:
        resolved_output = output_path.resolve(strict=True)
        resolved_output.relative_to(job_directory)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=404, detail="Final video not found.") from None

    if not resolved_output.is_file():
        raise HTTPException(status_code=404, detail="Final video not found.")

    return FileResponse(
        resolved_output,
        media_type="video/mp4",
        filename="hindi_dubbed.mp4",
    )