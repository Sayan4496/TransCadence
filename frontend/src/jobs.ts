import type { JobStatus, SyncResult } from './api';

const STORAGE_KEY = 'transcadence_jobs';

export type SavedJob = {
  job_id: string;
  filename: string;
  duration: number;
  status: string;
  progress?: number;
  output_ready: boolean;
  download_url?: string | null;
  created_at: string;
  sync_validation?: SyncResult[];
};

function validJob(value: unknown): value is SavedJob {
  if (!value || typeof value !== 'object') return false;
  const job = value as Partial<SavedJob>;
  return typeof job.job_id === 'string'
    && typeof job.filename === 'string'
    && typeof job.duration === 'number'
    && typeof job.status === 'string'
    && typeof job.output_ready === 'boolean'
    && typeof job.created_at === 'string';
}

export function readJobs(): SavedJob[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return Array.isArray(parsed) ? parsed.filter(validJob) : [];
  } catch {
    return [];
  }
}

export function saveJob(job: SavedJob): SavedJob[] {
  const jobs = [job, ...readJobs().filter((saved) => saved.job_id !== job.job_id)];
  localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs));
  return jobs;
}

export function updateFromStatus(job: SavedJob, status: JobStatus): SavedJob {
  return {
    ...job,
    status: status.status,
    output_ready: status.output_ready,
    ...(typeof status.progress === 'number' ? { progress: status.progress } : {}),
    ...(typeof status.download_url === 'string' || status.download_url === null ? { download_url: status.download_url } : {}),
    ...(Array.isArray(status.sync_validation) ? { sync_validation: status.sync_validation } : {}),
  };
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return 'Duration unavailable';
  const wholeSeconds = Math.floor(seconds);
  return `${Math.floor(wholeSeconds / 60).toString().padStart(2, '0')}:${(wholeSeconds % 60).toString().padStart(2, '0')}`;
}

export function friendlyStatus(status: string): string {
  const labels: Record<string, string> = {
    uploaded: 'Uploaded',
    transcribed: 'Ready to translate',
    translated: 'Translated',
    processing: 'Processing',
    starting: 'Processing',
    transcribing: 'Processing',
    translating: 'Processing',
    generating_tts: 'Processing',
    adjusting_audio: 'Processing',
    generating_subtitles: 'Processing',
    generating_video: 'Processing',
    validating_sync: 'Processing',
    completed: 'Completed',
    failed: 'Failed',
  };
  return labels[status] || 'Status unavailable';
}