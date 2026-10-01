import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { NavLink, Route, Routes, useNavigate } from 'react-router-dom';
import {
  ArrowDownToLine, ArrowRight, Bell, Check, CheckCircle2, ChevronDown, Clock3, CloudUpload,
  FileVideo2, FolderClosed, Globe2, House, Languages, LoaderCircle, Mic2, Moon, Play,
  Search, Settings, ShieldCheck, Sparkles, Sun, Video, XCircle,
} from 'lucide-react';
import { ApiError, getDownloadUrl, getJobStatus, isJobStatus, processJob, translateJob, uploadVideo, type JobStatus, type SyncResult } from './api';
import { formatDuration, friendlyStatus, readJobs, saveJob, updateFromStatus, type SavedJob } from './jobs';

type Theme = 'light' | 'dark' | 'system';
const navigation = [
  { label: 'Home', path: '/', Icon: House },
  { label: 'Translate Video', path: '/translate', Icon: Video },
  { label: 'My Projects', path: '/projects', Icon: FolderClosed },
  { label: 'Voice Library', path: '/voices', Icon: Mic2 },
  { label: 'History', path: '/history', Icon: Clock3 },
  { label: 'Settings', path: '/settings', Icon: Settings },
];

function Sidebar() {
  return <aside className="sidebar">
    <NavLink className="brand" to="/" aria-label="TransCadence home">
      <img className="brand-logo" src="/assets/logo.png" alt="TransCadence logo" />
    </NavLink>
    <nav className="side-nav" aria-label="Main navigation">
      {navigation.map(({ label, path, Icon }) => <NavLink key={path} to={path} end={path === '/'} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
        <Icon size={19} strokeWidth={1.9} /><span>{label}</span>
      </NavLink>)}
    </nav>
    <div className="workspace-note"><span className="workspace-icon"><ShieldCheck size={16} /></span><strong>Private workspace</strong><p>Jobs are saved in this browser.</p></div>
  </aside>;
}

function TopBar({ theme, onToggleTheme, search, setSearch }: { theme: Theme; onToggleTheme: () => void; search: string; setSearch: (value: string) => void }) {
  return <header className="topbar">
    <label className="global-search"><Search size={17} /><input aria-label="Search saved jobs" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search your projects, languages, or videos..." /><kbd>Search</kbd></label>
    <div className="top-actions">
      <button className="icon-button" aria-label={`Theme: ${theme}. Toggle light and dark`} onClick={onToggleTheme} type="button" title="Toggle theme">{theme === 'dark' ? <Moon size={20} /> : <Sun size={20} />}</button>
      <button className="icon-button" aria-label="Notifications unavailable" type="button" title="Notifications are not available"><Bell size={20} /></button>
      <span className="user-avatar" aria-hidden="true">TC</span><strong className="user-name">Workspace</strong><ChevronDown size={15} className="chevron" />
    </div>
  </header>;
}

function PageHeader({ step, title, highlight, description }: { step: string; title: string; highlight?: string; description: string }) {
  return <div className="page-header"><span className="eyebrow">{step}</span><h1>{title}{highlight && <> <span className="gradient-text">{highlight}</span></>}</h1><p>{description}</p></div>;
}

function StatusBadge({ job }: { job: SavedJob }) {
  const statusClass = job.status === 'completed' && job.output_ready ? 'success' : job.status === 'failed' ? 'failed' : 'working';
  const Icon = statusClass === 'success' ? CheckCircle2 : statusClass === 'failed' ? XCircle : Clock3;
  return <span className={`status-badge ${statusClass}`}><Icon size={13} />{friendlyStatus(job.status === 'completed' && !job.output_ready ? 'processing' : job.status)}</span>;
}

function EmptyState({ title, detail, action }: { title: string; detail: string; action?: ReactNode }) {
  return <div className="empty-state"><span className="empty-icon"><FileVideo2 size={24} /></span><h3>{title}</h3><p>{detail}</p>{action}</div>;
}

function SyncCard({ results }: { results?: SyncResult[] }) {
  if (!results?.length) return <div className="sync-card muted-sync"><span><ShieldCheck size={19} /></span><div><strong>Synchronization report unavailable</strong><p>No sync validation data was returned for this job.</p></div></div>;
  const passed = results.every((result) => result.status?.toUpperCase() === 'PASS');
  const tolerances = results.map((result) => result.tolerance_ms).filter((value): value is number => typeof value === 'number');
  const tolerance = tolerances.length ? Math.max(...tolerances) : undefined;
  return <div className={`sync-card ${passed ? 'sync-pass' : 'sync-review'}`}><span><ShieldCheck size={19} /></span><div><strong>{passed ? 'Synchronization validated' : 'Synchronization needs review'}</strong><p>Final video: {passed ? 'PASS' : 'CHECK'}{tolerance !== undefined ? ` · Tolerance: ±${tolerance} ms` : ''} · {results.length} measured segment{results.length === 1 ? '' : 's'}</p></div></div>;
}

function JobCard({ job, onOpen }: { job: SavedJob; onOpen: (job: SavedJob) => void }) {
  return <article className="project-card">
    <button className="project-visual" type="button" onClick={() => onOpen(job)} aria-label={`Open ${job.filename}`}><span className="project-play"><Play size={14} fill="currentColor" /></span><span className="project-duration">{formatDuration(job.duration)}</span></button>
    <div className="project-info"><div className="project-title-row"><div><h3>{job.filename}</h3><p>{new Date(job.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })} · Hindi</p></div><StatusBadge job={job} /></div>
      <div className="project-bottom"><span className="language-chip">English</span><ArrowRight size={14} /><span className="language-chip hindi-chip">हिन्दी</span><button type="button" className="text-link" onClick={() => onOpen(job)}>Open</button></div>
    </div>
  </article>;
}

function HomePage({ jobs, onUploaded, onOpen }: { jobs: SavedJob[]; onUploaded: (file: File, job: SavedJob) => void; onOpen: (job: SavedJob) => void }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  async function handleFile(candidate?: File) {
    if (!candidate) return;
    setError('');
    const extension = candidate.name.toLowerCase().split('.').pop();
    const expectedType = extension === 'mp4' ? 'video/mp4' : extension === 'webm' ? 'video/webm' : '';
    if (!expectedType || (candidate.type && candidate.type !== expectedType)) {
      setError('Choose an MP4 or WebM video.');
      return;
    }
    const file = candidate.type ? candidate : new File([candidate], candidate.name, { type: expectedType });
    let objectUrl = '';
    try {
      objectUrl = URL.createObjectURL(file);
      const duration = await new Promise<number>((resolve, reject) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = () => resolve(video.duration);
        video.onerror = () => reject(new Error('Could not read video duration.'));
        video.src = objectUrl;
      });
      URL.revokeObjectURL(objectUrl);
      objectUrl = '';
      if (!Number.isFinite(duration) || duration <= 0) throw new Error('Could not read video duration. Please choose a playable MP4 or WebM.');
      if (duration > 120) throw new Error('Videos must be 2 minutes or shorter.');
      setUploading(true);
      const response = await uploadVideo(file);
      if (typeof response.job_id !== 'string' || !response.job_id || typeof response.duration !== 'number') throw new Error('The server returned an invalid upload response.');
      const job: SavedJob = { job_id: response.job_id, filename: response.filename || file.name, duration: response.duration, status: response.status, output_ready: false, created_at: new Date().toISOString() };
      onUploaded(file, job);
      navigate('/translate');
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : reason instanceof Error ? reason.message : 'Upload failed. Please try again.');
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      setUploading(false);
    }
  }
  const latestJobs = jobs.slice(0, 3);
  return <>
    <section className="home-hero">
      <div className="hero-copy"><span className="eyebrow"><Sparkles size={13} /> AI-powered localization</span><h1>Make Every Video<br />Speak to <span className="gradient-text">the World</span></h1><p>Translate, dub, and subtitle your videos with AI.<br className="desktop-break" /> Keep your stories moving across languages.</p>
        <div className="feature-list"><span><Languages />Translate<br />Speech</span><span><FileVideo2 />Generate<br />Subtitles</span><span><Mic2 />Hindi Voice<br />Dubbing</span><span><Globe2 />Share beyond<br />language borders</span></div>
      </div>
      <div className="hero-art" aria-label="TransCadence language localization artwork"><img src="/assets/bg.png" alt="" /><span className="art-language english">🇺🇸 &nbsp; English</span><span className="art-language hindi">🇮🇳 &nbsp; हिन्दी</span><span className="art-caption">One story<br />in another language</span><span className="orbit orbit-one" /><span className="orbit orbit-two" /></div>
    </section>
    <section className="home-workspace">
      <div className={`upload-card${dragging ? ' dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); void handleFile(event.dataTransfer.files[0]); }}>
        <input id="video-upload" type="file" accept="video/mp4,video/webm,.mp4,.webm" className="visually-hidden" disabled={uploading} onChange={(event) => { void handleFile(event.target.files?.[0]); event.currentTarget.value = ''; }} />
        <CloudUpload size={42} className="upload-icon" /><h2>{uploading ? 'Uploading your video...' : 'Drop your video here'}</h2><p>or click to choose a file</p><small>MP4 or WebM · Up to 2 minutes</small>
        <label htmlFor="video-upload" className={`primary-button${uploading ? ' disabled' : ''}`} aria-disabled={uploading}><ArrowDownToLine size={16} />{uploading ? 'Uploading' : 'Upload Video'}</label>
        {error && <p className="inline-error" role="alert">{error}</p>}
      </div>
      <div className="recent-panel"><div className="section-heading"><div><span className="eyebrow">YOUR WORKSPACE</span><h2>Recent videos</h2></div><button type="button" className="text-link" onClick={() => navigate('/projects')}>See all <ArrowRight size={14} /></button></div>
        {latestJobs.length ? latestJobs.map((job) => <button className="recent-row" type="button" key={job.job_id} onClick={() => onOpen(job)}><span className="recent-thumb"><FileVideo2 size={20} /></span><span className="recent-copy"><strong>{job.filename}</strong><small>{formatDuration(job.duration)} · English → Hindi</small></span><StatusBadge job={job} /><ArrowRight size={15} /></button>) : <div className="sample-empty"><span className="sample-icon"><Play size={15} fill="currentColor" /></span><div><strong>No sample videos available</strong><p>Upload your own video to start a real translation.</p></div></div>}
      </div>
    </section>
  </>;
}

function TranslatePage({ job, videoUrl, updateJob }: { job?: SavedJob; videoUrl?: string; updateJob: (job: SavedJob) => void }) {
  const [busy, setBusy] = useState(false);
  const [hindiSelected, setHindiSelected] = useState(true);
  const [message, setMessage] = useState('');
  const [status, setStatus] = useState<JobStatus>();
  const processingRef = useRef(false);
  const navigate = useNavigate();
  useEffect(() => {
    let cancelled = false;
    if (job) void getJobStatus(job.job_id).then((current) => {
      if (!cancelled && isJobStatus(current, job.job_id)) {
        setStatus(current);
        updateJob(updateFromStatus(job, current));
      }
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [job?.job_id]);

  async function startDubbing() {
    if (!job || processingRef.current) return;
    processingRef.current = true;
    setBusy(true);
    setMessage('');
    let finalStatus: JobStatus | undefined;
    try {
      const translated = await translateJob(job.job_id);
      if (translated.job_id !== job.job_id || translated.target_language !== 'hi') throw new ApiError('The server returned an unexpected translation response.');
      const translatedJob = { ...job, status: translated.status };
      updateJob(translatedJob);
      let processError: unknown;
      let processSettled = false;
      const processRequest = processJob(job.job_id).then(() => { processSettled = true; }).catch((error: unknown) => { processSettled = true; processError = error; });
      while (!finalStatus || !['completed', 'failed'].includes(finalStatus.status)) {
        const current = await getJobStatus(job.job_id);
        if (!isJobStatus(current, job.job_id)) throw new ApiError('The server returned an invalid job status.');
        finalStatus = current;
        setStatus(current);
        updateJob(updateFromStatus(translatedJob, current));
        if (['completed', 'failed'].includes(current.status)) break;
        if (processSettled && processError) throw processError;
        await new Promise((resolve) => window.setTimeout(resolve, 1800));
      }
      await processRequest;
      if (finalStatus.status === 'failed') setMessage('The backend reported that this video could not be processed. Review the file and try again.');
      else if (processError && finalStatus.status !== 'completed') throw processError;
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'The request could not be completed. Please try again.');
    } finally {
      processingRef.current = false;
      setBusy(false);
    }
  }

  const completed = (status?.status || job?.status) === 'completed' && (status?.output_ready ?? job?.output_ready);
  const failed = (status?.status || job?.status) === 'failed';
  const downloadUrl = job ? getDownloadUrl(job.job_id) : '';
  const progress = status?.progress ?? job?.progress;
  return <>
    <PageHeader step="Step 1 · Translate video" title="Translate Your Video" highlight="in Hindi" description="AI-powered translation, dubbing, and subtitles, while preserving the original video." />
    <div className="translate-layout">
      <section className="video-column">
        <div className="video-frame">
          {completed ? <video key={downloadUrl} controls playsInline preload="metadata" src={downloadUrl} aria-label="Completed Hindi dubbed video" /> : videoUrl ? <video key={videoUrl} controls playsInline preload="metadata" src={videoUrl} aria-label="Uploaded source video" /> : <EmptyState title={job?.filename || 'No video selected'} detail={job ? 'The original upload is available only in the current browser session. You can still process and download the result.' : 'Upload a supported video from Home to begin.'} action={!job && <button className="secondary-button" type="button" onClick={() => navigate('/')}>Go to upload <ArrowRight size={15} /></button>} />}
          {job && <><span className="video-label"><FileVideo2 size={14} />{job.filename}</span><span className="source-label"><Globe2 size={14} />Original: English</span></>}
        </div>
        {busy && <div className="processing-panel" role="status"><LoaderCircle className="spin" size={19} /><div><strong>Processing your Hindi dub</strong><p>Waiting for the latest backend status.</p></div>{typeof progress === 'number' && <span className="real-progress">{Math.round(progress)}%</span>}<div className="progress-track"><span style={{ width: `${typeof progress === 'number' ? Math.max(0, Math.min(100, progress)) : 0}%` }} /></div></div>}
        {completed && <div className="result-details"><SyncCard results={status?.sync_validation || job?.sync_validation} /><div className="result-actions"><span className="result-language"><span>🇮🇳</span> Hindi dubbed video</span><a className="primary-button" href={downloadUrl} download><ArrowDownToLine size={16} />Download MP4</a></div></div>}
        {failed && <div className="failure-panel" role="alert"><XCircle size={19} /><div><strong>Processing failed</strong><p>{message || 'The backend reported that this job failed. No completed video is available.'}</p></div></div>}
        {message && !failed && <div className="inline-error" role="alert">{message}</div>}
      </section>
      <aside className="language-panel"><span className="panel-icon"><Languages size={20} /></span><h2>Select target language</h2><p>Choose a supported language for your video.</p>
        <button className={`language-option${hindiSelected ? ' selected' : ''}`} type="button" aria-pressed={hindiSelected} disabled={busy} onClick={() => setHindiSelected((selected) => !selected)}><span className="flag">🇮🇳</span><span><strong>हिन्दी</strong><small>Hindi · Available</small></span>{hindiSelected && <span className="checked"><Check size={15} /></span>}</button>
        <p className="support-note">Hindi dubbing is currently the only supported target language.</p>
        <button className="primary-button continue-button" type="button" disabled={!job || !hindiSelected || busy || completed} onClick={() => void startDubbing()}>{busy ? <><LoaderCircle className="spin" size={17} />Processing</> : completed ? <><CheckCircle2 size={17} />Complete</> : <>Continue <ArrowRight size={17} /></>}</button>
        {!job && <p className="panel-hint">Upload a video to enable translation.</p>}
      </aside>
    </div>
    <div className="benefit-row"><div><span><Languages /></span><strong>Hindi translation</strong><small>Translated by the connected backend</small></div><div><span><Mic2 /></span><strong>Hindi voice dubbing</strong><small>Generated with the available Piper voice</small></div><div><span><ShieldCheck /></span><strong>Measured sync validation</strong><small>Shown when returned by the backend</small></div></div>
  </>;
}

function ProjectsPage({ jobs, onOpen, search }: { jobs: SavedJob[]; onOpen: (job: SavedJob) => void; search: string }) {
  const navigate = useNavigate();
  const filtered = jobs.filter((job) => job.filename.toLowerCase().includes(search.toLowerCase()));
  return <><PageHeader step="Step 2 · My projects" title="Your Projects," highlight="All in One Place" description="Manage the videos you have uploaded from this browser." />
    <div className="summary-row"><div><FolderClosed /><strong>{jobs.length}</strong><span>Saved jobs</span></div><div><CheckCircle2 /><strong>{jobs.filter((job) => job.status === 'completed' && job.output_ready).length}</strong><span>Completed</span></div><div><Clock3 /><strong>{jobs.filter((job) => !['completed', 'failed'].includes(job.status)).length}</strong><span>Not complete</span></div><button type="button" className="primary-button" onClick={() => navigate('/')}>＋ New video</button></div>
    {filtered.length ? <div className="project-grid">{filtered.map((job) => <JobCard key={job.job_id} job={job} onOpen={onOpen} />)}</div> : <EmptyState title={jobs.length ? 'No matching videos' : 'No projects yet'} detail={jobs.length ? 'Try another search term.' : 'Upload your first video to get started.'} />}
  </>;
}

function HistoryPage({ jobs, onOpen, search }: { jobs: SavedJob[]; onOpen: (job: SavedJob) => void; search: string }) {
  const filtered = jobs.filter((job) => job.filename.toLowerCase().includes(search.toLowerCase()));
  return <><PageHeader step="Step 5 · History" title="Your Translation" highlight="History" description="Your uploaded videos and real Hindi dubbing results, saved on this device." />
    {filtered.length ? <div className="history-table-wrap"><table className="history-table"><thead><tr><th>Video</th><th>Languages</th><th>Type</th><th>Duration</th><th>Date added</th><th>Status</th><th>Actions</th></tr></thead><tbody>{filtered.map((job) => <tr key={job.job_id}><td><button className="history-video" type="button" onClick={() => onOpen(job)}><span className="history-thumb"><FileVideo2 size={17} /></span><span><strong>{job.filename}</strong><small>English · {formatDuration(job.duration)}</small></span></button></td><td><span className="flag">🇺🇸</span> <span className="flag">🇮🇳</span></td><td><span className="type-chip">{job.output_ready ? 'Hindi dub' : 'Video'}</span></td><td>{formatDuration(job.duration)}</td><td>{new Date(job.created_at).toLocaleDateString()}</td><td><StatusBadge job={job} /></td><td className="table-actions"><button className="round-action" type="button" aria-label={`Open ${job.filename}`} onClick={() => onOpen(job)}><Play size={14} /></button>{job.output_ready && job.status === 'completed' ? <a className="round-action" href={getDownloadUrl(job.job_id)} download aria-label={`Download ${job.filename}`}><ArrowDownToLine size={15} /></a> : <button className="round-action disabled-action" type="button" disabled aria-label="Download unavailable"><ArrowDownToLine size={15} /></button>}</td></tr>)}</tbody></table></div> : <EmptyState title={jobs.length ? 'No matching history' : 'No translation history yet'} detail="Videos you upload and process will appear here." />}
  </>;
}

function VoicesPage() {
  return <><PageHeader step="Step 4 · Voice library" title="Hindi Voice" highlight="Library" description="The current dubbing pipeline uses its configured Piper Hindi voice." />
    <section className="voice-card"><span className="voice-art"><Mic2 size={27} /></span><div className="voice-details"><span className="available-tag"><span />Available in pipeline</span><h2>Piper Hindi voice</h2><p>Hindi · Piper text-to-speech</p><p className="voice-description">This is the voice used by the existing backend. Voice selection and preview controls are unavailable because the API does not expose configurable voices.</p></div><span className="locked-control"><Settings size={15} /> Fixed by backend</span></section>
  </>;
}

function SettingsPage({ theme, setTheme }: { theme: Theme; setTheme: (theme: Theme) => void }) {
  return <><PageHeader step="Step 6 · Settings" title="Settings" description="Personalize the local workspace. Account and security controls are not connected." />
    <div className="settings-layout"><nav className="settings-nav" aria-label="Settings sections"><a className="selected" href="#preferences"><Sun size={16} />Preferences <ChevronDown size={15} /></a><a href="#language"><Globe2 size={16} />Translation <ChevronDown size={15} /></a><a href="#unavailable"><ShieldCheck size={16} />Privacy &amp; Security <ChevronDown size={15} /></a></nav>
      <div className="settings-content"><section className="settings-card" id="preferences"><div className="settings-heading"><h2>Preferences</h2><p>Changes are saved in this browser.</p></div><div className="setting-row"><div><strong>Theme</strong><small>Choose how TransCadence looks.</small></div><div className="segmented-control">{(['light', 'dark', 'system'] as Theme[]).map((option) => <button key={option} type="button" className={theme === option ? 'selected' : ''} onClick={() => setTheme(option)}>{option === 'light' ? <Sun size={15} /> : option === 'dark' ? <Moon size={15} /> : <Settings size={15} />}{option[0].toUpperCase() + option.slice(1)}</button>)}</div></div></section>
        <section className="settings-card" id="language"><div className="settings-heading"><h2>Translation</h2><p>Available options from the connected backend.</p></div><div className="setting-row"><div><strong>Target language</strong><small>Hindi is currently the only supported target language.</small></div><span className="fixed-value"><span>🇮🇳</span> हिन्दी <span className="fixed-pill">Fixed</span></span></div></section>
        <section className="settings-card unavailable-card" id="unavailable"><div className="settings-heading"><h2>Account &amp; security</h2><p>Sign-in and account management are not available in this MVP.</p></div><div className="disabled-setting"><ShieldCheck size={19} /><div><strong>Account controls unavailable</strong><small>No authentication or account API is configured.</small></div><button type="button" disabled>Unavailable</button></div></section>
      </div>
    </div>
  </>;
}

function App() {
  const [jobs, setJobs] = useState<SavedJob[]>([]);
  const [activeJobId, setActiveJobId] = useState(() => localStorage.getItem('transcadence_active_job') || '');
  const [videoUrl, setVideoUrl] = useState('');
  const [search, setSearch] = useState('');
  const [theme, setThemeState] = useState<Theme>(() => (localStorage.getItem('transcadence_theme') as Theme) || 'light');
  const navigate = useNavigate();
  const activeJob = useMemo(() => jobs.find((job) => job.job_id === activeJobId) || jobs[0], [jobs, activeJobId]);

  useEffect(() => {
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
    const applyTheme = () => {
      document.documentElement.dataset.theme = theme === 'system' ? (systemTheme.matches ? 'dark' : 'light') : theme;
    };
    applyTheme();
    localStorage.setItem('transcadence_theme', theme);
    systemTheme.addEventListener('change', applyTheme);
    return () => systemTheme.removeEventListener('change', applyTheme);
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    void Promise.all(readJobs().map(async (job) => {
      try {
        const status = await getJobStatus(job.job_id);
        return isJobStatus(status, job.job_id) ? updateFromStatus(job, status) : job;
      } catch {
        return job;
      }
    })).then((updated) => {
      if (cancelled) return;
      localStorage.setItem('transcadence_jobs', JSON.stringify(updated));
      setJobs(updated);
    });
    return () => { cancelled = true; };
  }, []);

  function updateJob(job: SavedJob) {
    setJobs(saveJob(job));
  }

  function activateJob(job: SavedJob) {
    setActiveJobId(job.job_id);
    localStorage.setItem('transcadence_active_job', job.job_id);
    navigate('/translate');
  }

  function uploaded(file: File, job: SavedJob) {
    if (file) setVideoUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(file);
    });
    setJobs(saveJob(job));
    setActiveJobId(job.job_id);
    localStorage.setItem('transcadence_active_job', job.job_id);
  }

  function toggleTheme() {
    setThemeState((current) => current === 'dark' ? 'light' : 'dark');
  }

  return <div className="app-shell"><Sidebar /><main className="main-area"><TopBar theme={theme} onToggleTheme={toggleTheme} search={search} setSearch={setSearch} /><div className="page-content"><Routes>
    <Route path="/" element={<HomePage jobs={jobs} onUploaded={uploaded} onOpen={activateJob} />} />
    <Route path="/translate" element={<TranslatePage job={activeJob} videoUrl={videoUrl} updateJob={updateJob} />} />
    <Route path="/projects" element={<ProjectsPage jobs={jobs} onOpen={activateJob} search={search} />} />
    <Route path="/voices" element={<VoicesPage />} />
    <Route path="/history" element={<HistoryPage jobs={jobs} onOpen={activateJob} search={search} />} />
    <Route path="/settings" element={<SettingsPage theme={theme} setTheme={setThemeState} />} />
    <Route path="*" element={<EmptyState title="Page not found" detail="Choose a page from the navigation." />} />
  </Routes></div></main></div>;
}

export default App;