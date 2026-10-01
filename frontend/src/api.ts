export type SyncResult = {
  segment_id?: number;
  target_duration_seconds?: number;
  actual_audio_duration_seconds?: number;
  sync_error_ms?: number;
  absolute_sync_error_ms?: number;
  tolerance_ms?: number;
  status?: string;
};

export type JobStatus = {
  job_id: string;
  status: string;
  language?: string;
  progress?: number;
  output_ready: boolean;
  download_url?: string | null;
  sync_validation?: SyncResult[];
};

export type UploadResponse = {
  job_id: string;
  status: string;
  filename: string;
  duration: number;
};

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

function safeMessage(status: number, detail: unknown): string {
  if (status === 429) return 'Translation service is temporarily unavailable. Please try again.';
  if (status === 413) return 'This video is too large for the upload service.';
  if (status === 404) return 'This job could not be found. Upload the video again to continue.';
  if (status >= 500) return 'The server could not complete this request. Please try again.';
  if (status === 400 || status === 422) {
    const candidate = typeof detail === 'string'
      ? detail
      : detail && typeof detail === 'object' && 'message' in detail && typeof detail.message === 'string'
        ? detail.message
        : undefined;
    if (candidate && candidate.length < 220 && !/[\r\n]|Traceback|[A-Za-z]:\\/.test(candidate)) return candidate;
    return 'The request was not accepted. Check the selected video and try again.';
  }
  return 'The request could not be completed. Please try again.';
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError('Could not connect to TransCadence. Check that the backend is running.');
  }
  if (!response.ok) {
    let detail: unknown;
    try {
      detail = (await response.json() as { detail?: unknown }).detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(safeMessage(response.status, detail), response.status);
  }
  try {
    return await response.json() as T;
  } catch {
    throw new ApiError('The server returned an invalid response. Please try again.');
  }
}

export function uploadVideo(file: File): Promise<UploadResponse> {
  const body = new FormData();
  body.append('file', file);
  return request<UploadResponse>('/upload', { method: 'POST', body });
}

export function translateJob(jobId: string): Promise<{ job_id: string; status: string; target_language: string }> {
  return request(`/jobs/${encodeURIComponent(jobId)}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_language: 'hi' }),
  });
}

export function processJob(jobId: string): Promise<Record<string, unknown>> {
  return request(`/jobs/${encodeURIComponent(jobId)}/process`, { method: 'POST' });
}

export function getJobStatus(jobId: string): Promise<JobStatus> {
  return request(`/jobs/${encodeURIComponent(jobId)}/status`);
}

export function getDownloadUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/download`;
}

export function isJobStatus(value: unknown, jobId: string): value is JobStatus {
  if (!value || typeof value !== 'object') return false;
  const status = value as Partial<JobStatus>;
  return status.job_id === jobId
    && typeof status.status === 'string'
    && typeof status.output_ready === 'boolean'
    && (status.progress === undefined || (typeof status.progress === 'number' && Number.isFinite(status.progress)))
    && (status.sync_validation === undefined || Array.isArray(status.sync_validation));
}