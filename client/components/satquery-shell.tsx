'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Moon, Sun, Satellite, ArrowRight, Check, Circle, Loader2, Copy, Download, Upload, Layers3, ScanSearch, GitBranch, Database, Cpu, Activity, MapPin, ChevronDown, Clock } from 'lucide-react'
import { useEffect, useState, useRef, useMemo } from 'react'
import gsap from 'gsap'

import { EarthExperience } from './earth-experience'
import { checkHealth, queryAgent, getModels, type AgentResponse, type HealthResponse, type ModelInfo, ApiError } from '@/lib/api'

export function Shell({ children }: { children: React.ReactNode }) {
  return <div className="min-h-screen">
    <header className="sticky top-4 z-50 mx-auto w-full max-w-6xl px-4 sm:px-6">
      <nav className="nav-capsule flex items-center justify-between px-5 py-2.5 sm:px-6 sm:py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight"><span className="brand-mark"><Satellite size={18}/></span><span>SatQuery <span className="text-cyan">AI</span></span></Link>
        <div className="hidden items-center gap-7 text-sm text-muted-foreground md:flex">{[['Workspace','/workspace'],['Model Registry','/registry'],['Evaluation','/evaluation'],['Architecture','/architecture']].map(([label,href]) => <Link key={href} href={href} className="transition hover:text-foreground">{label}</Link>)}</div>
        <div className="flex items-center gap-2"><Link href="/workspace" className="button-primary hidden sm:flex">Launch app <ArrowRight size={15}/></Link></div>
      </nav>
    </header>{children}
    <footer className="mx-auto flex max-w-7xl flex-col gap-3 border-t border-white/10 px-5 py-8 text-sm text-muted-foreground md:flex-row md:items-center md:justify-between"><span>© 2026 SatQuery AI · ISRO / SAC research interface</span><span className="flex gap-4"><Link href="/registry">Models</Link><Link href="/evaluation">Benchmarks</Link><Link href="/architecture">System trace</Link></span></footer>
  </div>
}

export function Glass({ children, className = '' }: { children: React.ReactNode; className?: string }) { return <section className={`glass ${className}`}>{children}</section> }
export function Pill({ children, tone = 'blue' }: { children: React.ReactNode; tone?: 'blue'|'green'|'orange'|'muted' }) { return <span className={`pill pill-${tone}`}>{children}</span> }

const steps = ['Interpret query','Validate inputs','Select tools','Run specialists','Integrate outputs','Compute confidence']
export function AgentStepper({ active = 3 }: { active?: number }) { return <div className="space-y-3">{steps.map((step, i) => <div key={step} className="flex items-center gap-3 text-sm"><span className={`step-icon ${i < active ? 'done' : i === active ? 'active' : ''}`}>{i < active ? <Check size={13}/> : i === active ? <Loader2 size={13} className="animate-spin"/> : <Circle size={10}/>}</span><span className={i <= active ? 'text-foreground' : 'text-muted-foreground'}>{step}</span>{i === active && <span className="ml-auto text-xs text-cyan">running</span>}</div>)}</div> }
export function Confidence({ value = 92 }: { value?: number }) { return <div className="space-y-2"><div className="flex justify-between text-xs"><span className="text-muted-foreground">Model confidence</span><strong className="text-cyan">{value}% · High</strong></div><div className="confidence-track"><div style={{ width: `${value}%` }} /></div></div> }
export function TraceViewer() { const [copied,setCopied] = useState(false); const trace = '{\n  "task": "change_detection",\n  "aoi": "Ahmedabad / Sabarmati corridor",\n  "models": ["S3-CD", "S5-Segment"],\n  "co_registration": 0.98,\n  "status": "complete",\n  "latency_ms": 4210\n}' ; return <div><div className="mb-3 flex items-center justify-between"><span className="eyebrow">execution.trace</span><button className="icon-btn" aria-label="Copy trace" onClick={() => { navigator.clipboard?.writeText(trace); setCopied(true) }}>{copied ? <Check size={15}/> : <Copy size={15}/>}</button></div><pre className="trace">{trace}</pre></div> }

export function Evidence({ pair = false }: { pair?: boolean }) { return <div className="evidence relative overflow-hidden"><div className="evidence-grid"/><div className="evidence-land"/><div className="absolute left-[23%] top-[30%] h-16 w-24 border-2 border-cyan shadow-[0_0_24px_rgba(90,200,250,.65)]"/><div className="absolute left-[52%] top-[49%] h-20 w-32 border-2 border-orange shadow-[0_0_24px_rgba(255,159,10,.45)]"/><span className="absolute left-[23%] top-[26%] text-[10px] text-cyan">NEW CONSTRUCTION</span>{pair && <span className="absolute right-3 top-3 rounded bg-black/60 px-2 py-1 text-[10px] text-white">Δ 2023 → 2024</span>}</div> }

export function Stat({ label, value, detail }: { label:string; value:string; detail:string }) { return <Glass className="p-5"><div className="eyebrow">{label}</div><div className="mt-2 text-3xl font-semibold tracking-tight">{value}</div><div className="mt-1 text-xs text-muted-foreground">{detail}</div></Glass> }

// ---------------------------------------------------------------------------
// Search History Entry
// ---------------------------------------------------------------------------
interface SearchEntry {
  id: number
  query: string
  task: string
  thumbUrl: string
  timestamp: number
  status: string
  confidence: number
  result: AgentResponse
  imageFile: File
}

// ---------------------------------------------------------------------------
// Bounding Box Overlay — renders the actual image + dynamic bbox from API
// ---------------------------------------------------------------------------
function GroundedImageViewer({ imageUrl, bbox, label }: {
  imageUrl: string | null
  bbox: number[] | null  // [x1, y1, x2, y2] in original pixel coords from model
  label?: string
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)

  const onLoad = () => {
    const img = imgRef.current
    if (img) setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight })
  }

  // Compute scaled bbox position relative to displayed image size
  const scaledBox = useMemo(() => {
    if (!bbox || bbox.length < 4 || !naturalSize || !containerRef.current || !imgRef.current) return null
    const displayedW = imgRef.current.clientWidth
    const displayedH = imgRef.current.clientHeight
    if (displayedW === 0 || displayedH === 0) return null
    const scaleX = displayedW / naturalSize.w
    const scaleY = displayedH / naturalSize.h
    // Account for object-contain centering
    const containerW = containerRef.current.clientWidth
    const containerH = containerRef.current.clientHeight
    const offsetX = (containerW - displayedW) / 2
    const offsetY = (containerH - displayedH) / 2
    return {
      left: bbox[0] * scaleX + offsetX,
      top: bbox[1] * scaleY + offsetY,
      width: (bbox[2] - bbox[0]) * scaleX,
      height: (bbox[3] - bbox[1]) * scaleY,
    }
  }, [bbox, naturalSize])

  if (!imageUrl) {
    return (
      <div className="flex h-[220px] items-center justify-center rounded-2xl border border-white/10 bg-white/[.03]">
        <p className="text-xs text-muted-foreground">Upload an image & run a query to see grounded results</p>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40" style={{ minHeight: 180 }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        src={imageUrl}
        alt="Satellite imagery"
        onLoad={onLoad}
        className="block w-full object-contain"
        style={{ maxHeight: 320 }}
        draggable={false}
      />
      {scaledBox && (
        <>
          <div
            className="absolute border-2 border-cyan shadow-[0_0_16px_rgba(90,200,250,.55)] transition-all duration-200"
            style={{
              left: scaledBox.left,
              top: scaledBox.top,
              width: scaledBox.width,
              height: scaledBox.height,
              pointerEvents: 'none',
            }}
          />
          {label && (
            <span
              className="absolute rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-cyan backdrop-blur-sm"
              style={{ left: scaledBox.left, top: Math.max(0, scaledBox.top - 18) }}
            >
              {label}
            </span>
          )}
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Workspace — Main Analysis Interface
// ---------------------------------------------------------------------------
export function Workspace() {
  const [mode, setMode] = useState('Single Image')
  const [query, setQuery] = useState('')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<AgentResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [backendUp, setBackendUp] = useState<boolean | null>(null)
  const [agentStep, setAgentStep] = useState(0)

  // File state
  const [singleFile, setSingleFile] = useState<File | null>(null)
  const [opticalFile, setOpticalFile] = useState<File | null>(null)
  const [sarFile, setSarFile] = useState<File | null>(null)
  const [t1File, setT1File] = useState<File | null>(null)
  const [t2File, setT2File] = useState<File | null>(null)

  // Search history
  const [searchHistory, setSearchHistory] = useState<SearchEntry[]>([])

  const fileInputRef = useRef<HTMLInputElement>(null)
  const fileInputRef2 = useRef<HTMLInputElement>(null)
  const activeUploadSlot = useRef<'single' | 'optical' | 'sar' | 't1' | 't2'>('single')

  // Create object URLs for image previews — memoized to avoid re-creating on every render
  const singlePreviewUrl = useMemo(() => singleFile ? URL.createObjectURL(singleFile) : null, [singleFile])
  const opticalPreviewUrl = useMemo(() => opticalFile ? URL.createObjectURL(opticalFile) : null, [opticalFile])
  const sarPreviewUrl = useMemo(() => sarFile ? URL.createObjectURL(sarFile) : null, [sarFile])
  const t1PreviewUrl = useMemo(() => t1File ? URL.createObjectURL(t1File) : null, [t1File])
  const t2PreviewUrl = useMemo(() => t2File ? URL.createObjectURL(t2File) : null, [t2File])

  // Cleanup object URLs on unmount / file change
  useEffect(() => { return () => { if (singlePreviewUrl) URL.revokeObjectURL(singlePreviewUrl) } }, [singlePreviewUrl])
  useEffect(() => { return () => { if (opticalPreviewUrl) URL.revokeObjectURL(opticalPreviewUrl) } }, [opticalPreviewUrl])
  useEffect(() => { return () => { if (sarPreviewUrl) URL.revokeObjectURL(sarPreviewUrl) } }, [sarPreviewUrl])
  useEffect(() => { return () => { if (t1PreviewUrl) URL.revokeObjectURL(t1PreviewUrl) } }, [t1PreviewUrl])
  useEffect(() => { return () => { if (t2PreviewUrl) URL.revokeObjectURL(t2PreviewUrl) } }, [t2PreviewUrl])

  // The primary image URL used for both left preview and right grounded overlay
  const primaryImageUrl = mode === 'Single Image' ? singlePreviewUrl
    : mode === 'Optical + SAR' ? opticalPreviewUrl
    : t1PreviewUrl

  // Health check
  useEffect(() => {
    checkHealth().then(() => setBackendUp(true)).catch(() => setBackendUp(false))
  }, [])

  const handleFileSelect = (slot: typeof activeUploadSlot.current) => {
    activeUploadSlot.current = slot
    if (slot === 'sar' || slot === 't2') fileInputRef2.current?.click()
    else fileInputRef.current?.click()
  }

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    const slot = activeUploadSlot.current
    if (slot === 'single') setSingleFile(f)
    else if (slot === 'optical') setOpticalFile(f)
    else if (slot === 'sar') setSarFile(f)
    else if (slot === 't1') setT1File(f)
    else if (slot === 't2') setT2File(f)
    e.target.value = ''
  }

  const getActiveFiles = (): { name: string; size: string }[] => {
    const fmt = (f: File) => ({ name: f.name, size: `${(f.size / 1024).toFixed(0)} KB` })
    if (mode === 'Optical + SAR') return [opticalFile, sarFile].filter(Boolean).map(f => fmt(f!))
    if (mode === 'Two-Date Pair') return [t1File, t2File].filter(Boolean).map(f => fmt(f!))
    return singleFile ? [fmt(singleFile)] : []
  }

  const canRun = (): boolean => {
    if (!query.trim()) return false
    if (mode === 'Single Image') return !!singleFile
    if (mode === 'Optical + SAR') return !!opticalFile && !!sarFile
    if (mode === 'Two-Date Pair') return !!t1File && !!t2File
    return false
  }

  const run = async () => {
    if (!canRun() || running) return
    setRunning(true)
    setResult(null)
    setError(null)
    setAgentStep(0)

    const stepInterval = setInterval(() => {
      setAgentStep(prev => Math.min(prev + 1, 5))
    }, 800)

    try {
      const apiMode = mode === 'Optical + SAR' ? 'optical_sar' as const
        : mode === 'Two-Date Pair' ? 'bitemporal' as const
        : 'single' as const

      const res = await queryAgent({
        query,
        mode: apiMode,
        image: singleFile ?? undefined,
        imageOptical: opticalFile ?? undefined,
        imageSar: sarFile ?? undefined,
        imageT1: t1File ?? undefined,
        imageT2: t2File ?? undefined,
      })
      setResult(res)

      // Add to search history
      const primaryFile = singleFile || opticalFile || t1File
      if (primaryFile) {
        const thumbUrl = URL.createObjectURL(primaryFile)
        setSearchHistory(prev => [{
          id: Date.now(),
          query,
          task: res.task,
          thumbUrl,
          timestamp: Date.now(),
          status: res.status,
          confidence: res.confidence,
          result: res,
          imageFile: primaryFile,
        }, ...prev].slice(0, 20)) // keep last 20
      }
    } catch (err) {
      if (err instanceof ApiError) setError(`API Error ${err.status}: ${err.detail}`)
      else setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      clearInterval(stepInterval)
      setAgentStep(5)
      setRunning(false)
    }
  }

  // Restore a search from history
  const restoreSearch = (entry: SearchEntry) => {
    setQuery(entry.query)
    setResult(entry.result)
    setSingleFile(entry.imageFile)
    setMode('Single Image')
    setError(null)
  }

  const formatTimeAgo = (ts: number) => {
    const sec = Math.floor((Date.now() - ts) / 1000)
    if (sec < 60) return 'just now'
    const min = Math.floor(sec / 60)
    if (min < 60) return `${min}m ago`
    const hr = Math.floor(min / 60)
    return `${hr}h ago`
  }

  const activeFiles = getActiveFiles()
  const done = !!result
  const confValue = result ? Math.round(result.confidence * 100) : 0

  return (
    <Shell>
      <main className="mx-auto max-w-[1500px] px-4 py-6 lg:px-8">
        <div className="mb-6 flex items-end justify-between">
          <div>
            <div className="eyebrow">SATQUERY / WORKSPACE</div>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight">Ask your imagery anything.</h1>
          </div>
          <Pill tone={backendUp === true ? 'green' : backendUp === false ? 'orange' : 'muted'}>
            <span className={backendUp === true ? 'status-dot' : ''} />
            {backendUp === true ? 'backend ready' : backendUp === false ? 'backend offline' : 'checking…'}
          </Pill>
        </div>

        {/* Hidden file inputs */}
        <input ref={fileInputRef} type="file" className="hidden" accept=".tif,.tiff,.png,.jpg,.jpeg,.geotiff" onChange={onFileChange} />
        <input ref={fileInputRef2} type="file" className="hidden" accept=".tif,.tiff,.png,.jpg,.jpeg,.geotiff" onChange={onFileChange} />

        <div className="grid gap-4 xl:grid-cols-[300px_minmax(400px,1fr)_360px]">
          {/* ═══════════ LEFT: Input Imagery Panel ═══════════ */}
          <Glass className="flex flex-col p-4" style={{ maxHeight: 'calc(100vh - 140px)' }}>
            <div className="mb-4 flex items-center justify-between">
              <h2 className="font-medium">Input imagery</h2>
              <Upload size={17} className="text-muted-foreground" />
            </div>
            <div className="segmented mb-4">
              {['Single Image', 'Optical + SAR', 'Two-Date Pair'].map(x => (
                <button key={x} className={mode === x ? 'selected' : ''} onClick={() => setMode(x)}>{x}</button>
              ))}
            </div>

            {/* ACTUAL IMAGE PREVIEW — the main visual element */}
            {mode === 'Single Image' && singlePreviewUrl && (
              <div className="mb-3 cursor-pointer overflow-hidden rounded-2xl border border-white/10 bg-black/40" onClick={() => handleFileSelect('single')}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={singlePreviewUrl} alt={singleFile?.name} className="block w-full object-contain" style={{ maxHeight: 200 }} draggable={false} />
              </div>
            )}
            {mode === 'Optical + SAR' && (opticalPreviewUrl || sarPreviewUrl) && (
              <div className="mb-3 grid grid-cols-2 gap-2">
                {opticalPreviewUrl && (
                  <div className="overflow-hidden rounded-xl border border-white/10 bg-black/40 cursor-pointer" onClick={() => handleFileSelect('optical')}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={opticalPreviewUrl} alt="Optical" className="block w-full object-contain" style={{ maxHeight: 120 }} draggable={false} />
                    <p className="px-2 py-1 text-[10px] text-muted-foreground">Optical</p>
                  </div>
                )}
                {sarPreviewUrl && (
                  <div className="overflow-hidden rounded-xl border border-white/10 bg-black/40 cursor-pointer" onClick={() => handleFileSelect('sar')}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={sarPreviewUrl} alt="SAR" className="block w-full object-contain" style={{ maxHeight: 120 }} draggable={false} />
                    <p className="px-2 py-1 text-[10px] text-muted-foreground">SAR</p>
                  </div>
                )}
              </div>
            )}
            {mode === 'Two-Date Pair' && (t1PreviewUrl || t2PreviewUrl) && (
              <div className="mb-3 grid grid-cols-2 gap-2">
                {t1PreviewUrl && (
                  <div className="overflow-hidden rounded-xl border border-white/10 bg-black/40 cursor-pointer" onClick={() => handleFileSelect('t1')}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={t1PreviewUrl} alt="T1" className="block w-full object-contain" style={{ maxHeight: 120 }} draggable={false} />
                    <p className="px-2 py-1 text-[10px] text-muted-foreground">T1 · Before</p>
                  </div>
                )}
                {t2PreviewUrl && (
                  <div className="overflow-hidden rounded-xl border border-white/10 bg-black/40 cursor-pointer" onClick={() => handleFileSelect('t2')}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={t2PreviewUrl} alt="T2" className="block w-full object-contain" style={{ maxHeight: 120 }} draggable={false} />
                    <p className="px-2 py-1 text-[10px] text-muted-foreground">T2 · After</p>
                  </div>
                )}
              </div>
            )}

            {/* Upload dropzone — shown when no image for current mode */}
            {mode === 'Single Image' && !singleFile && (
              <div className="upload-zone" onClick={() => handleFileSelect('single')}>
                <Upload size={22} className="mx-auto mb-2 text-cyan" />
                <p className="text-sm">Drop imagery or browse</p>
                <p className="mt-1 text-xs text-muted-foreground">GeoTIFF, TIFF, PNG, JPEG · up to 2 GB</p>
              </div>
            )}
            {mode === 'Optical + SAR' && (!opticalFile || !sarFile) && (
              <div className="space-y-2">
                {!opticalFile && (
                  <div className="upload-zone" onClick={() => handleFileSelect('optical')}>
                    <Upload size={18} className="mx-auto mb-1 text-cyan" />
                    <p className="text-xs">Optical (Sentinel-2)</p>
                  </div>
                )}
                {!sarFile && (
                  <div className="upload-zone" onClick={() => handleFileSelect('sar')}>
                    <Upload size={18} className="mx-auto mb-1 text-cyan" />
                    <p className="text-xs">SAR Radar (Sentinel-1)</p>
                  </div>
                )}
              </div>
            )}
            {mode === 'Two-Date Pair' && (!t1File || !t2File) && (
              <div className="space-y-2">
                {!t1File && (
                  <div className="upload-zone" onClick={() => handleFileSelect('t1')}>
                    <Upload size={18} className="mx-auto mb-1 text-cyan" />
                    <p className="text-xs">Time T1 — Before</p>
                  </div>
                )}
                {!t2File && (
                  <div className="upload-zone" onClick={() => handleFileSelect('t2')}>
                    <Upload size={18} className="mx-auto mb-1 text-cyan" />
                    <p className="text-xs">Time T2 — After</p>
                  </div>
                )}
              </div>
            )}

            {/* File metadata */}
            {activeFiles.length > 0 && (
              <div className="mt-3 space-y-1">
                {activeFiles.map((f, i) => (
                  <div key={i} className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[.04] px-3 py-2">
                    <div className="min-w-0">
                      <p className="truncate text-xs font-medium">{f.name}</p>
                      <p className="text-[10px] text-muted-foreground">{f.size}</p>
                    </div>
                    <Pill tone="green"><span className="status-dot" /> loaded</Pill>
                  </div>
                ))}
              </div>
            )}

            {/* ─── RECENT SEARCHES ─── */}
            <div className="mt-5 flex min-h-0 flex-1 flex-col">
              <div className="eyebrow mb-3">RECENT SEARCHES</div>
              <div className="flex-1 space-y-2 overflow-y-auto pr-1" style={{ maxHeight: 200 }}>
                {searchHistory.length === 0 ? (
                  <p className="py-4 text-center text-xs text-muted-foreground">No searches yet</p>
                ) : (
                  searchHistory.map(entry => (
                    <button
                      key={entry.id}
                      className="flex w-full items-start gap-2.5 rounded-xl border border-white/8 bg-white/[.03] p-2 text-left transition hover:bg-white/[.08]"
                      onClick={() => restoreSearch(entry)}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={entry.thumbUrl}
                        alt=""
                        className="h-9 w-9 flex-none rounded-lg object-cover"
                        draggable={false}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-xs font-medium">{entry.query}</p>
                        <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                          <span className="capitalize">{entry.task}</span>
                          <span>·</span>
                          <Clock size={9} />
                          <span>{formatTimeAgo(entry.timestamp)}</span>
                          {entry.status && (
                            <>
                              <span>·</span>
                              <span className={entry.status === 'verified' ? 'text-green-400' : 'text-orange-400'}>{entry.status}</span>
                            </>
                          )}
                        </div>
                      </div>
                    </button>
                  ))
                )}
              </div>
            </div>
          </Glass>

          {/* ═══════════ CENTER: Query & Response ═══════════ */}
          {/* flex column with min-h-0 so overflow works inside the flex child */}
          <Glass className="flex min-h-[640px] flex-col p-5">
            <div className="mb-4 flex flex-none items-center justify-between">
              <div>
                <div className="eyebrow">NATURAL LANGUAGE QUERY</div>
                <p className="mt-1 text-sm text-muted-foreground">No model selection required. The agent routes your task.</p>
              </div>
              <Pill>agentic mode</Pill>
            </div>

            {/* Scrollable result area — this is the part that grows and scrolls */}
            <div className="min-h-0 flex-1 overflow-y-auto space-y-5 pr-1">
              {/* User bubble */}
              {(query || result) && (
                <div className="flex justify-end">
                  <div className="user-bubble">{query || 'Ask about this imagery…'}</div>
                </div>
              )}

              {/* Running state */}
              {running && (
                <Glass className="mx-auto max-w-md p-5">
                  <div className="mb-4 flex items-center justify-between">
                    <span className="font-medium">Agent is working</span>
                    <span className="text-xs text-cyan">step {agentStep + 1} / 6</span>
                  </div>
                  <AgentStepper active={agentStep} />
                  <div className="shimmer mt-5 h-1 rounded-full" />
                </Glass>
              )}

              {/* Error */}
              {error && !running && (
                <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-5">
                  <div className="mb-2 flex items-center gap-2"><Pill tone="orange">error</Pill></div>
                  <p className="text-sm text-red-300">{error}</p>
                  <p className="mt-2 text-xs text-muted-foreground">Make sure the backend is running: python scripts/run_api_server.py</p>
                </div>
              )}

              {/* Success result */}
              {result && !running && (
                <div className="response-card">
                  <div className="mb-3 flex items-center gap-2">
                    <Pill tone={result.verification?.passed ? 'green' : 'orange'}>
                      {result.verification?.passed ? 'verified' : result.verification?.decision || result.status}
                    </Pill>
                    <span className="text-xs text-muted-foreground">
                      {result.task} · {result.metadata?.backbone || 'agent'}
                    </span>
                  </div>
                  <p className="text-lg leading-relaxed">{result.text}</p>
                  <div className="mt-5">
                    <Confidence value={confValue} />
                  </div>
                  {result.verification && (
                    <div className="mt-4 rounded-xl border border-white/10 bg-white/[.03] p-3">
                      <div className="eyebrow mb-2">VERIFICATION GATE</div>
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Decision</span>
                          <span className={result.verification.passed ? 'text-green-400' : 'text-orange-400'}>
                            {result.verification.decision}
                          </span>
                        </div>
                        {result.verification.reasons.length > 0 && (
                          <div className="mt-1">
                            {result.verification.reasons.map((r, i) => (
                              <p key={i} className="text-muted-foreground">• {r}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Empty state */}
              {!running && !result && !error && (
                <div className="empty-state">
                  <div className="orbital-icon"><Satellite size={28} /></div>
                  <p className="mt-4 text-sm text-muted-foreground">
                    Your grounded answer, evidence, and execution trace will appear here.
                  </p>
                </div>
              )}
            </div>

            {/* Query input bar — pinned to bottom, never pushed away */}
            <div className="mt-4 flex-none">
              <div className="mb-3 flex flex-wrap gap-2">
                {['Describe this scene', 'Locate the airstrip', 'What changed between dates?', 'Compare optical and SAR'].map(x => (
                  <button key={x} className="query-chip" onClick={() => setQuery(x)}>{x}</button>
                ))}
              </div>
              <div className="query-box">
                <input
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') run() }}
                  placeholder="Ask about this imagery…"
                  aria-label="Natural language query"
                />
                <button className="button-primary" onClick={run} disabled={running || !canRun()}>
                  {running ? 'Running…' : 'Ask'} <ArrowRight size={15} />
                </button>
              </div>
              <p className="mt-2 text-center text-[11px] text-muted-foreground">
                SatQuery can make mistakes. Validate critical findings against source imagery.
              </p>
            </div>
          </Glass>

          {/* ═══════════ RIGHT: Grounded Results Panel ═══════════ */}
          <Glass className="flex flex-col p-4" style={{ maxHeight: 'calc(100vh - 140px)' }}>
            <div className="mb-4 flex flex-none items-center justify-between">
              <h2 className="font-medium">Grounded results</h2>
              <Pill tone={done ? 'green' : 'muted'}>{done ? 'live' : 'awaiting query'}</Pill>
            </div>

            {/* Scrollable results content */}
            <div className="min-h-0 flex-1 overflow-y-auto pr-1">
              {/* ACTUAL IMAGE with dynamic bounding box overlay */}
              <GroundedImageViewer
                imageUrl={primaryImageUrl}
                bbox={result?.spatial_evidence?.bbox ?? null}
                label={result?.task === 'grounding' ? query : undefined}
              />

              <div className="mt-4 grid grid-cols-2 gap-2">
                <button className="control-btn"><Layers3 size={14} /> overlay</button>
                <button className="control-btn"><ScanSearch size={14} /> zoom 100%</button>
              </div>
              <div className="my-5 border-t border-white/10" />

              <div className="eyebrow mb-3">EVIDENCE SUMMARY</div>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Task</span>
                  <span>{result?.task || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Confidence</span>
                  <span className="text-cyan">{done ? `${confValue}%` : '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <span>{result?.status || '—'}</span>
                </div>
                {result?.spatial_evidence?.bbox && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Bounding box</span>
                    <span className="font-mono text-xs">[{result.spatial_evidence.bbox.map(v => v.toFixed(0)).join(', ')}]</span>
                  </div>
                )}
                {result?.spatial_evidence?.mask && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Change ratio</span>
                    <span>{((result.spatial_evidence.mask as any).changed_ratio * 100).toFixed(1)}%</span>
                  </div>
                )}
              </div>

              {/* EXECUTION TRACE — internally scrollable */}
              <div className="my-5 border-t border-white/10" />
              <details open>
                <summary className="mb-3 flex cursor-pointer list-none items-center justify-between text-sm font-medium">
                  Execution trace <ChevronDown size={15} />
                </summary>
                <div style={{ maxHeight: 240, overflowY: 'auto', overflowX: 'auto' }}>
                  {result?.trace ? (
                    <pre className="trace" style={{ whiteSpace: 'pre', wordBreak: 'keep-all' }}>{JSON.stringify(result.trace, null, 2)}</pre>
                  ) : (
                    <TraceViewer />
                  )}
                </div>
              </details>
            </div>

            {/* Download button — pinned at bottom */}
            <button className="button-secondary mt-4 w-full flex-none justify-center" onClick={() => {
              if (result) {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `satquery_report_${Date.now()}.json`
                a.click()
                URL.revokeObjectURL(url)
              } else {
                alert('No results to download yet.')
              }
            }}>
              <Download size={15} /> Download report
            </button>
          </Glass>
        </div>
      </main>
    </Shell>
  )
}

export function Home() {
  const rotatingWords = ['ANALYZE', 'COMPARE', 'DISCOVER']
  const [wordIndex, setWordIndex] = useState(0)
  const wordRef = useRef<HTMLSpanElement>(null)
  const underlineRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (!underlineRef.current || !wordRef.current) {
        setWordIndex((prev) => (prev + 1) % rotatingWords.length)
        return
      }

      const tl = gsap.timeline({
        onComplete: () => {
          setWordIndex((prev) => (prev + 1) % rotatingWords.length)
        },
      })

      // 1. Underline slides/draws out
      tl.to(underlineRef.current, {
        scaleX: 0,
        transformOrigin: 'right center',
        duration: 0.32,
        ease: 'power2.in',
      })
      // 2. Word slides up with fade out
      .to(
        wordRef.current,
        {
          y: -8,
          opacity: 0,
          duration: 0.28,
          ease: 'power2.in',
        },
        '<0.05'
      )
    }, 2800)

    return () => window.clearInterval(interval)
  }, [rotatingWords.length])

  useEffect(() => {
    if (!wordRef.current || !underlineRef.current) return
    const tl = gsap.timeline()
    tl.fromTo(
      wordRef.current,
      { y: 8, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.35, ease: 'power2.out' }
    ).fromTo(
      underlineRef.current,
      { scaleX: 0, transformOrigin: 'left center' },
      { scaleX: 1, duration: 0.4, ease: 'power2.out' },
      '-=0.15'
    )
  }, [wordIndex])

  return (
    <Shell>
      <main>
        <section id="satquery-hero" className="hero relative flex min-h-screen flex-col justify-between items-center overflow-hidden px-5 pt-12 pb-16 text-center lg:pt-20 bg-[#0A0A0A]">
          {/* Ambient space background stars (CSS-only, zero WebGL) */}
          <div
            className="pointer-events-none absolute inset-0 z-0 opacity-40"
            style={{
              backgroundImage:
                'radial-gradient(1.5px 1.5px at 25px 35px, #ffffff, rgba(0,0,0,0)), radial-gradient(1px 1px at 50px 80px, #93c5fd, rgba(0,0,0,0)), radial-gradient(1.5px 1.5px at 120px 45px, #ffffff, rgba(0,0,0,0)), radial-gradient(1px 1px at 180px 140px, #60a5fa, rgba(0,0,0,0))',
              backgroundSize: '240px 240px',
            }}
            aria-hidden="true"
          />
          {/* Realistic 3D Earth Experience */}
          <EarthExperience />
          <div id="satquery-hero-content" className="relative z-10 mx-auto max-w-5xl pt-4 lg:pt-8">
            <Pill><Satellite size={13}/> agentic vision-language analysis</Pill>
            <h1 className="mx-auto mt-6 max-w-3xl text-balance text-4xl font-semibold uppercase tracking-[-.045em] sm:text-5xl lg:text-6xl drop-shadow-[0_2px_16px_rgba(0,0,0,0.8)]">
              Ask your satellite imagery to{' '}
              <span className="relative inline-block whitespace-nowrap">
                <span ref={wordRef} className="inline-block text-[#f5f5f5]">
                  {rotatingWords[wordIndex]}
                </span>
                <span
                  ref={underlineRef}
                  className="absolute -bottom-1 left-0 h-[2.5px] w-full rounded-full bg-gradient-to-r from-[#5a9bff] via-[#7ee2a8] to-[#ffb454]"
                />
              </span>
            </h1>
            <p className="mx-auto mt-6 max-w-xl text-pretty text-base lg:text-lg leading-7 lg:leading-8 text-muted-foreground drop-shadow-[0_2px_8px_rgba(0,0,0,0.8)]">
              SatQuery AI turns complex remote-sensing workflows into grounded, explainable answers — with the right specialist model selected automatically.
            </p>
            <div className="mt-9 flex justify-center gap-3">
              <Link href="/workspace" className="button-primary">Try SatQuery AI <ArrowRight size={16}/></Link>
              <Link href="/architecture" className="button-secondary">See how it works</Link>
            </div>
          </div>
          {/* Spacer to preserve identical hero positioning */}
          <div className="relative z-10 h-12 w-full" />
        </section>
        <section className="mx-auto max-w-7xl px-5 py-20"><div className="max-w-xl"><div className="eyebrow">ONE INTERFACE / FIVE CAPABILITIES</div><h2 className="mt-3 text-4xl font-semibold tracking-tight">From pixels to decisions.</h2></div><div className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-3">{[['Single-image VQA','Describe scenes, answer questions, and caption Earth observation imagery.','ScanSearch'],['Text-guided grounding','Locate airstrips, vessels, roads, and facilities with evidence overlays.','MapPin'],['Bi-temporal change','Surface construction, vegetation loss, and flood extent across dates.','GitBranch'],['Optical–SAR fusion','Fuse complementary modalities for robust all-weather interpretation.','Layers3'],['Agentic orchestration','No manual model selection. SatQuery routes, binds, runs, and traces.','Cpu']].map(([title,copy,icon])=><Glass key={title} className="p-6"><div className="feature-icon">{icon==='ScanSearch'?<ScanSearch/>:icon==='MapPin'?<MapPin/>:icon==='GitBranch'?<GitBranch/>:icon==='Layers3'?<Layers3/>:<Cpu/>}</div><h3 className="mt-5 text-lg font-medium">{title}</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">{copy}</p></Glass>)}</div></section><section className="mx-auto max-w-7xl px-5 py-20"><Glass className="p-8 lg:p-12"><div className="eyebrow">HOW IT WORKS</div><h2 className="mt-3 text-4xl font-semibold tracking-tight">One question. A complete evidence chain.</h2><div className="mt-10 grid gap-6 md:grid-cols-5">{['Upload','Ask','Agent routes','Models run','Grounded answer'].map((x,i)=><div key={x} className="relative"><div className="step-number">{i+1}</div><h3 className="mt-4 font-medium">{x}</h3><p className="mt-2 text-sm text-muted-foreground">{['Bring Sentinel, Cartosat, or RISAT imagery.','Use natural language, not model syntax.','The controller selects tools and parameters.','Specialists execute with provenance.','Answer, confidence, evidence, and trace.'][i]}</p></div>)}</div></Glass></section></main></Shell>) }

export function Registry() {
  const fallbackModels = [
    ['S1', 'Single-image VLM', 'RSVQA · VRSBench', 'Implemented / Tested', '92%'],
    ['S2', 'Grounding specialist', 'RefCOCO · AirBus', 'Implemented / Tested', '0.81 IoU'],
    ['S3', 'Change detection', 'CDVQA · LEVIR-CD', 'Implemented / Tested', '87% OA'],
    ['S4', 'Optical–SAR fusion', 'BigEarthNet-MM', 'Partial', '84% mAP'],
    ['S5', 'Segmentation', 'LoveDA · SpaceNet', 'Not done', '—'],
  ]

  const [liveModels, setLiveModels] = useState<ModelInfo[]>([])
  const [fetched, setFetched] = useState(false)

  useEffect(() => {
    getModels()
      .then(res => { setLiveModels(res.models); setFetched(true) })
      .catch(() => setFetched(false))
  }, [])

  // Merge live model status into the static table
  const getStatus = (taskId: string): { status: string; tone: 'green' | 'orange' | 'muted' } => {
    const taskMap: Record<string, string> = { S1: 'vqa', S2: 'grounding', S3: 'change', S4: 'fusion', S5: 'segmentation' }
    const task = taskMap[taskId]
    const live = liveModels.find(m => m.task === task)
    if (!fetched) return { status: fallbackModels.find(f => f[0] === taskId)?.[3] || '—', tone: 'muted' }
    if (live?.loaded) return { status: 'Loaded ✓', tone: 'green' }
    if (live) return { status: 'Registered', tone: 'orange' }
    return { status: 'Not available', tone: 'muted' }
  }

  return (
    <Shell>
      <main className="mx-auto max-w-7xl px-5 py-20">
        <div className="eyebrow">SATQUERY / MODEL REGISTRY</div>
        <h1 className="mt-3 max-w-2xl text-5xl font-semibold tracking-tight">Specialists, routed by intent.</h1>
        <p className="mt-5 max-w-xl text-lg leading-8 text-muted-foreground">
          A transparent capability layer for remote-sensing workflows. Every answer exposes which model ran and why.
        </p>
        {fetched && (
          <div className="mt-4">
            <Pill tone="green"><span className="status-dot" /> live from backend</Pill>
          </div>
        )}
        <div className="mt-12 overflow-hidden rounded-3xl border border-white/10 bg-white/[.04]">
          <div className="hidden grid-cols-[80px_1fr_1fr_180px_100px] gap-4 border-b border-white/10 px-6 py-4 text-xs uppercase tracking-wider text-muted-foreground md:grid">
            <span>ID</span><span>Specialist</span><span>Benchmarks</span><span>Status</span><span>Metric</span>
          </div>
          {fallbackModels.map(m => {
            const st = getStatus(m[0])
            return (
              <div key={m[0]} className="grid gap-3 border-b border-white/10 px-6 py-5 last:border-0 md:grid-cols-[80px_1fr_1fr_180px_100px] md:items-center">
                <span className="font-mono text-cyan">{m[0]}</span>
                <span className="font-medium">{m[1]}</span>
                <span className="text-sm text-muted-foreground">{m[2]}</span>
                <span><Pill tone={st.tone}>{st.status}</Pill></span>
                <span className="text-sm">{m[4]}</span>
              </div>
            )
          })}
        </div>
      </main>
    </Shell>
  )
}

export function Evaluation() { return <Shell><main className="mx-auto max-w-7xl px-5 py-20"><div className="eyebrow">SATQUERY / EVALUATION</div><h1 className="mt-3 text-5xl font-semibold tracking-tight">Benchmarks you can reproduce.</h1><p className="mt-5 max-w-xl text-lg leading-8 text-muted-foreground">Internal evaluation snapshot across visual question answering, change detection, grounding, and multimodal fusion.</p><div className="mt-12 grid gap-4 md:grid-cols-4"><Stat label="RSVQA accuracy" value="91.8%" detail="+8.4 pts vs optical baseline"/><Stat label="Change OA" value="87.2%" detail="CDVQA / 2024 holdout"/><Stat label="Grounding IoU" value="0.81" detail="S2 specialist · 1,240 prompts"/><Stat label="Median latency" value="4.2s" detail="p95 8.7s · agentic route"/></div><div className="mt-4 grid gap-4 lg:grid-cols-2"><Glass className="p-6"><div className="flex items-center justify-between"><div><div className="eyebrow">FUSION ABLATION</div><h2 className="mt-2 text-xl font-medium">Optical-only vs optical + SAR</h2></div><Activity className="text-cyan"/></div><div className="chart mt-8"><div className="bar optical"/><div className="bar fusion"/><div className="chart-labels"><span>Optical</span><span>Fusion</span></div></div><div className="mt-4 flex gap-5 text-xs text-muted-foreground"><span><i className="legend optical"/> mAP 0.71</span><span><i className="legend fusion"/> mAP 0.84</span></div></Glass><Glass className="p-6"><div className="eyebrow">REPRODUCE THIS RUN</div><h2 className="mt-2 text-xl font-medium">Evaluation command</h2><pre className="trace mt-6">satquery eval \
  --benchmark CDVQA \
  --modalities optical,sar \
  --split holdout-2024</pre><button className="button-secondary mt-4">Copy command <Copy size={14}/></button></Glass></div></main></Shell> }

export function Architecture() { const nodes=['Interpret','Validate','Select','Bind params','Execute','Integrate','Confidence','Trace']; return <Shell><main className="mx-auto max-w-7xl px-5 py-20"><div className="eyebrow">SATQUERY / ARCHITECTURE</div><h1 className="mt-3 max-w-3xl text-5xl font-semibold tracking-tight">A controller that keeps the reasoning visible.</h1><p className="mt-5 max-w-2xl text-lg leading-8 text-muted-foreground">SatQuery decomposes intent, binds evidence, calls specialist models, and returns a grounded answer with a complete provenance trace.</p><Glass className="mt-12 overflow-hidden p-6 lg:p-12"><div className="architecture-grid">{nodes.map((x,i)=><div key={x} className="architecture-node"><div className="node-icon">{i+1}</div><span>{x}</span>{i<nodes.length-1&&<ArrowRight className="node-arrow" size={18}/>}</div>)}</div><div className="mt-12 grid gap-4 md:grid-cols-3"><Glass className="p-5"><Database className="text-cyan"/><h3 className="mt-4 font-medium">Evidence first</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">Every claim links back to a region, modality, or temporal comparison.</p></Glass><Glass className="p-5"><Cpu className="text-cyan"/><h3 className="mt-4 font-medium">Tool-native routing</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">The controller selects the smallest capable specialist set for each task.</p></Glass><Glass className="p-5"><GitBranch className="text-cyan"/><h3 className="mt-4 font-medium">Reproducible trace</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">Parameters, timings, and model versions remain inspectable after every run.</p></Glass></div></Glass></main></Shell> }

export function PageRouter({ page }: { page: 'home'|'workspace'|'registry'|'evaluation'|'architecture' }) { return page==='home'?<Home/>:page==='workspace'?<Workspace/>:page==='registry'?<Registry/>:page==='evaluation'?<Evaluation/>:<Architecture/> }

// TODO: connect to backend: POST /api/query { images, modality, query } -> { answer, confidence, evidence, trace }
void Download
void GitBranch
void Database
void Activity
void MapPin
void ScanSearch
void Upload
void Layers3
void Cpu
void ChevronDown
void ArrowRight
void ArrowRight
void Circle
void Loader2
void Copy
void Check
void Sun
void Moon
void Satellite
void usePathname
void PageRouter
void Workspace
void Home
void Registry
void Evaluation
void Architecture
void Shell
void Glass
void Pill
void AgentStepper
void Confidence
void TraceViewer
void Evidence
void Stat
