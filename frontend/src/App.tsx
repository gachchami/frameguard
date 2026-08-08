import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight, Check, ChevronDown, Download, Eye, FileJson, FileText, LoaderCircle,
  LockKeyhole, ScanFace, Settings, ShieldCheck, Sparkles, UploadCloud, Video, X,
} from "lucide-react";

type View = "protect" | "faces" | "results" | "settings";
type Job = { id: string; status: "queued" | "running" | "completed" | "failed"; message: string; error?: string; result?: ResultData };
type Finding = { id: string; type: string; value: string; modality: string; confidence: number; start_ms: number; end_ms: number; action: string };
type ResultData = { run_id: string; assets: Record<string, string>; findings: Finding[]; metrics: Record<string, unknown>; report_preview: Record<string, unknown> };
type Profile = { person_id: string; label: string; portrait_url: string; preview_urls: string[]; observation_count: number; first_seen_ms: number; last_seen_ms: number };
type Gallery = { gallery_id: string; profiles: Profile[]; summary: Record<string, unknown> };
type Config = Record<string, string | number | boolean | string[]>;

const initialConfig: Config = {
  api_base: "http://127.0.0.1:8091/v1", model: "Qwen2.5-Omni-3B", detector_mode: "qwen",
  chunk_seconds: 5, deterministic_sample_interval_ms: 350, face_sample_interval_ms: 200,
  face_score_threshold: .75, face_max_track_gap_ms: 900, face_min_track_observations: 2,
  reference_match_threshold: .363, identity_similarity_threshold: .4, run_log_level: "INFO",
  show_sensitive_values: false, include_raw_model_output: false,
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Request failed (${response.status})`); }
  return response.json();
}

function FileDrop({ file, onFile, accept = "video/*", compact = false }: { file: File | null; onFile: (file: File | null) => void; accept?: string; compact?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  return <div className={`dropzone ${drag ? "drag" : ""} ${compact ? "compact" : ""}`}
    onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
    onDrop={e => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files[0] || null); }} onClick={() => input.current?.click()}>
    <input ref={input} type="file" accept={accept} hidden onChange={e => onFile(e.target.files?.[0] || null)} />
    {file ? <><div className="file-icon"><Video size={22}/></div><div><strong>{file.name}</strong><span>{(file.size / 1048576).toFixed(1)} MB · ready</span></div><button className="icon-button" onClick={e => { e.stopPropagation(); onFile(null); }}><X size={17}/></button></>
      : <><UploadCloud size={compact ? 22 : 28}/><div><strong>{compact ? "Add a reference photo" : "Drop a video here"}</strong><span>{compact ? "JPG, PNG or WebP" : "or click to choose an MP4 or MOV"}</span></div></>}
  </div>;
}

function Toggle({ checked, onChange, label, detail }: { checked: boolean; onChange: (v: boolean) => void; label: string; detail?: string }) {
  return <label className="toggle-row"><span><strong>{label}</strong>{detail && <small>{detail}</small>}</span><button type="button" role="switch" aria-checked={checked} className={`toggle ${checked ? "on" : ""}`} onClick={() => onChange(!checked)}><i /></button></label>;
}

function ProgressCard({ job }: { job: Job }) {
  return <div className={`progress-card ${job.status}`}>
    {job.status === "completed" ? <Check/> : job.status === "failed" ? <X/> : <LoaderCircle className="spin"/>}
    <div><strong>{job.status === "completed" ? "Protection complete" : job.status === "failed" ? "Something went wrong" : "FrameGuard is working"}</strong><span>{job.error || job.message}</span></div>
  </div>;
}

export default function App() {
  const [view, setView] = useState<View>("protect");
  const [config, setConfig] = useState<Config>(initialConfig);
  const [video, setVideo] = useState<File | null>(null);
  const [faceVideo, setFaceVideo] = useState<File | null>(null);
  const [reference, setReference] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<ResultData | null>(null);
  const [error, setError] = useState("");
  const [gallery, setGallery] = useState<Gallery | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [faceMode, setFaceMode] = useState<"review" | "all" | "reference" | "likely_minors">("review");
  const [galleryAction, setGalleryAction] = useState("blur_selected");
  const [ocr, setOcr] = useState(true); const [qr, setQr] = useState(true);

  useEffect(() => { api<Config>("/api/config").then(c => setConfig({...initialConfig, ...c})).catch(() => {}); }, []);
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setTimeout(() => api<Job>(`/api/jobs/${job.id}`).then(next => {
      setJob(next); if (next.status === "completed" && next.result) {
        const payload = next.result as unknown as Gallery;
        if ("gallery_id" in payload) setGallery(payload); else { setResult(next.result); setView("results"); }
      }
    }).catch(e => setError(e.message)), 1200);
    return () => window.clearTimeout(timer);
  }, [job]);

  const setCfg = (key: string, value: string | number | boolean) => setConfig(old => ({...old, [key]: value}));
  const submit = async (path: string, source: File, extra: Record<string, unknown> = {}, photo?: File | null) => {
    setError(""); setResult(null); const form = new FormData(); form.append("video", source); if (photo) form.append("reference_face", photo);
    form.append("config_json", JSON.stringify({...config, ...extra}));
    try { const data = await api<{job_id: string}>(path, {method: "POST", body: form}); setJob({id: data.job_id, status: "queued", message: "Preparing your video"}); }
    catch (e) { setError(e instanceof Error ? e.message : "Request failed"); }
  };
  const scanGallery = () => faceVideo && submit("/api/galleries", faceVideo);
  const renderGallery = async () => {
    if (!gallery) return; setError(""); const form = new FormData();
    form.append("config_json", JSON.stringify({...config, selected_person_ids: selected, gallery_action: galleryAction}));
    try { const data = await api<{job_id: string}>(`/api/galleries/${gallery.gallery_id}/render`, {method: "POST", body: form}); setJob({id: data.job_id, status: "queued", message: "Preparing your video"}); }
    catch (e) { setError(e instanceof Error ? e.message : "Request failed"); }
  };

  return <div className="app-shell">
    <aside>
      <button className="brand" onClick={() => setView("protect")}><span><ShieldCheck size={22}/></span>FrameGuard</button>
      <nav>
        <p>WORKFLOWS</p>
        <button className={view === "protect" ? "active" : ""} onClick={() => setView("protect")}><LockKeyhole/>Sensitive content</button>
        <button className={view === "faces" ? "active" : ""} onClick={() => setView("faces")}><ScanFace/>Face privacy</button>
        <p>OUTPUT</p>
        <button className={view === "results" ? "active" : ""} onClick={() => setView("results")}><Sparkles/>Latest result{result && <i className="nav-dot"/>}</button>
      </nav>
      <div className="aside-bottom"><div className="privacy-note"><ShieldCheck/><span><strong>Runs locally</strong><small>Your media stays on this machine.</small></span></div><button className={view === "settings" ? "active" : ""} onClick={() => setView("settings")}><Settings/>Settings</button></div>
    </aside>

    <main>
      {error && <div className="error-banner"><X size={18}/>{error}<button onClick={() => setError("")}><X size={15}/></button></div>}
      {view === "protect" && <section className="page">
        <header><span className="eyebrow">SENSITIVE CONTENT</span><h1>Protect what shouldn't be seen.</h1><p>Detect and remove exposed secrets from video, on-screen text, QR codes, and spoken audio.</p></header>
        <div className="workflow-grid"><div className="card upload-card"><div className="step"><b>1</b><span><strong>Choose a video</strong><small>FrameGuard processes it entirely on this machine.</small></span></div><FileDrop file={video} onFile={setVideo}/></div>
          <div className="card options-card"><div className="step"><b>2</b><span><strong>Choose what to detect</strong><small>Semantic analysis is always enabled.</small></span></div>
            <Toggle checked={ocr} onChange={setOcr} label="Visible sensitive text" detail="Emails, API keys, IPs, account IDs and private URLs"/>
            <Toggle checked={qr} onChange={setQr} label="QR codes" detail="Locate and obscure readable QR codes"/>
            <div className="always-on"><Sparkles size={17}/><span><strong>Visual + audio understanding</strong><small>Qwen analyzes context across every clip.</small></span><em>Always on</em></div>
          </div></div>
        {job && view === "protect" && <ProgressCard job={job}/>}<button className="primary-action" disabled={!video || job?.status === "running" || job?.status === "queued"} onClick={() => video && submit("/api/jobs/sensitive", video, {deterministic_ocr: ocr, detect_qr_codes: qr})}>Protect this video <ArrowRight/></button>
      </section>}

      {view === "faces" && <section className="page">
        <header><span className="eyebrow">FACE PRIVACY</span><h1>Decide who stays visible.</h1><p>Review detected people or apply a fast automatic privacy rule.</p></header>
        <div className="segmented">{([['review','Review people'],['all','Blur everyone'],['reference','Match a photo'],['likely_minors','Children only']] as const).map(([id,label]) => <button className={faceMode === id ? "active" : ""} onClick={() => setFaceMode(id)} key={id}>{label}</button>)}</div>
        <div className="card"><div className="step"><b>1</b><span><strong>Choose a video</strong><small>Use a clear source for the best face tracking.</small></span></div><FileDrop file={faceVideo} onFile={f => {setFaceVideo(f); setGallery(null); setSelected([]);}}/></div>
        {faceMode === "review" ? <>
          {!gallery ? <button className="primary-action" disabled={!faceVideo || job?.status === "running"} onClick={scanGallery}>Detect people <ScanFace/></button> : <div className="card gallery-card"><div className="gallery-heading"><div><span className="eyebrow">{gallery.profiles.length} PEOPLE FOUND</span><h2>Select people to protect</h2></div><div><button className="text-button" onClick={() => setSelected(gallery.profiles.map(p => p.person_id))}>Select all</button><button className="text-button" onClick={() => setSelected([])}>Clear</button></div></div>
            <div className="profiles">{gallery.profiles.map((p, i) => {const active=selected.includes(p.person_id); const previews=p.preview_urls?.length ? p.preview_urls : [p.portrait_url]; return <button key={p.person_id} aria-pressed={active} className={`profile ${active ? "selected" : ""}`} onClick={() => setSelected(old => active ? old.filter(x => x !== p.person_id) : [...old,p.person_id])}><div className="profile-photos">{previews.slice(0,3).map((url,index)=><img key={url} className={index===0 ? "primary" : ""} src={url} alt={`Person ${i+1}, sighting ${index+1}`}/>)}</div><span>{active && <i><Check size={14}/></i>}<b>Person {i+1}</b><small>{p.observation_count} sightings · {(p.first_seen_ms/1000).toFixed(1)}–{(p.last_seen_ms/1000).toFixed(1)}s</small></span></button>})}</div>
            <label className="select-label">Privacy rule<select value={galleryAction} onChange={e => setGalleryAction(e.target.value)}><option value="blur_selected">Blur selected people</option><option value="keep_selected_visible">Keep selected visible, blur everyone else</option></select><ChevronDown/></label>
            <button className="primary-action" disabled={!selected.length} onClick={renderGallery}>Create protected video ({selected.length} selected) <ArrowRight/></button></div>}
        </> : <div className="card automatic-card"><div className="step"><b>2</b><span><strong>{faceMode === "all" ? "Blur every detected face" : faceMode === "reference" ? "Upload a face to match" : "Blur visually apparent children"}</strong><small>{faceMode === "likely_minors" ? "Experimental visual classification—not a legal age determination." : "Face tracks are followed through the full video."}</small></span></div>
          {faceMode === "reference" && <FileDrop file={reference} onFile={setReference} accept="image/*" compact/>}
          <button className="primary-action" disabled={!faceVideo || (faceMode === "reference" && !reference)} onClick={() => faceVideo && submit("/api/jobs/automatic", faceVideo, {face_redaction_mode: faceMode}, reference)}>Apply face protection <ArrowRight/></button></div>}
        {job && view === "faces" && <ProgressCard job={job}/>} 
      </section>}

      {view === "results" && <section className="page results-page"><header><span className="eyebrow">LATEST RESULT</span><h1>{result ? "Your protected video is ready." : "No result yet."}</h1><p>{result ? `Run ${result.run_id} completed successfully.` : "Complete a privacy workflow and the output will appear here."}</p></header>
        {result ? <><div className="result-hero"><video src={result.assets.video} controls/><div className="result-summary"><div className="success-mark"><Check/></div><h2>Protection complete</h2><p>{result.findings.length} privacy finding{result.findings.length === 1 ? "" : "s"} handled.</p><a className="download-main" href={result.assets.video} download><Download/>Download protected MP4</a><div className="download-links"><a href={result.assets.report} download><FileJson/>Audit report</a><a href={result.assets.log} download><FileText/>Run log</a></div></div></div>
          <div className="card findings-card"><h2>Privacy findings</h2>{result.findings.length ? <div className="table-wrap"><table><thead><tr><th>Type</th><th>Protected value</th><th>Time</th><th>Confidence</th><th>Action</th></tr></thead><tbody>{result.findings.map(f => <tr key={f.id}><td><span className="type-pill">{f.type}</span></td><td>{f.value}</td><td>{(f.start_ms/1000).toFixed(1)}–{(f.end_ms/1000).toFixed(1)}s</td><td>{Math.round(f.confidence*100)}%</td><td>{f.action}</td></tr>)}</tbody></table></div> : <p className="empty">No sensitive content was detected.</p>}</div></> : <div className="empty-result"><Eye/><h2>Nothing to preview</h2><p>Start with Sensitive content or Face privacy.</p><button onClick={() => setView("protect")}>Protect a video</button></div>}
      </section>}

      {view === "settings" && <section className="page settings-page"><header><span className="eyebrow">SETTINGS</span><h1>Processing controls.</h1><p>Defaults are tuned for the included models. Changes apply to the next run.</p></header>
        <SettingsGroup title="Qwen connection"><Field label="API base"><input value={String(config.api_base)} onChange={e=>setCfg('api_base',e.target.value)}/></Field><Field label="Model ID"><input value={String(config.model)} onChange={e=>setCfg('model',e.target.value)}/></Field><Field label="Detector"><select value={String(config.detector_mode)} onChange={e=>setCfg('detector_mode',e.target.value)}><option value="qwen">Qwen endpoint</option><option value="mock">Local smoke test</option></select></Field></SettingsGroup>
        <SettingsGroup title="Detection"><Range label="Video chunk length" value={Number(config.chunk_seconds)} min={3} max={10} unit="s" onChange={v=>setCfg('chunk_seconds',v)}/><Range label="OCR scan interval" value={Number(config.deterministic_sample_interval_ms)} min={200} max={1200} step={50} unit="ms" onChange={v=>setCfg('deterministic_sample_interval_ms',v)}/><Range label="Face confidence" value={Number(config.face_score_threshold)} min={.5} max={.95} step={.01} onChange={v=>setCfg('face_score_threshold',v)}/><Range label="Identity grouping strictness" value={Number(config.identity_similarity_threshold)} min={.3} max={.65} step={.01} onChange={v=>setCfg('identity_similarity_threshold',v)}/><p className="setting-note">Higher grouping strictness keeps similar-looking people separate. Scene cuts are isolated before identity grouping; the default is 0.40.</p></SettingsGroup>
        <SettingsGroup title="Privacy & diagnostics"><Toggle checked={Boolean(config.show_sensitive_values)} onChange={v=>setCfg('show_sensitive_values',v)} label="Show exact detected values" detail="Off by default to avoid revealing secrets in the UI and report."/><Toggle checked={Boolean(config.include_raw_model_output)} onChange={v=>setCfg('include_raw_model_output',v)} label="Include raw model output" detail="Adds the full Qwen response to the audit report."/></SettingsGroup>
      </section>}
    </main>
  </div>;
}

function SettingsGroup({title, children}:{title:string; children:React.ReactNode}) { return <div className="card settings-group"><h2>{title}</h2>{children}</div>; }
function Field({label,children}:{label:string;children:React.ReactNode}) { return <label className="field"><span>{label}</span>{children}</label>; }
function Range({label,value,min,max,step=1,unit="",onChange}:{label:string;value:number;min:number;max:number;step?:number;unit?:string;onChange:(n:number)=>void}) { return <label className="range"><span><strong>{label}</strong><em>{value}{unit}</em></span><input type="range" value={value} min={min} max={max} step={step} onChange={e=>onChange(Number(e.target.value))}/></label>; }
