'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Moon, Sun, Satellite, ArrowRight, Check, Circle, Loader2, Copy, Download, Upload, Layers3, ScanSearch, GitBranch, Database, Cpu, Activity, MapPin, ChevronDown } from 'lucide-react'
import { useEffect, useState, useRef } from 'react'
import gsap from 'gsap'

import { EarthExperience } from './earth-experience'

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

export function Workspace() { const [mode,setMode]=useState('Single Image'); const [query,setQuery]=useState(''); const [running,setRunning]=useState(false); const [done,setDone]=useState(false); const [file,setFile]=useState('sentinel2_ahmedabad_2024.tif'); const run=()=>{setRunning(true); setDone(false); setTimeout(()=>{setRunning(false);setDone(true)},4200)}; return <Shell><main className="mx-auto max-w-[1500px] px-4 py-6 lg:px-8"><div className="mb-6 flex items-end justify-between"><div><div className="eyebrow">SATQUERY / WORKSPACE</div><h1 className="mt-2 text-3xl font-semibold tracking-tight">Ask your imagery anything.</h1></div><Pill tone="green"><span className="status-dot"/> backend ready</Pill></div><div className="grid gap-4 xl:grid-cols-[300px_minmax(400px,1fr)_360px]"><Glass className="p-4"><div className="mb-4 flex items-center justify-between"><h2 className="font-medium">Input imagery</h2><Upload size={17} className="text-muted-foreground"/></div><div className="segmented mb-4">{['Single Image','Optical + SAR','Two-Date Pair'].map(x=><button key={x} className={mode===x?'selected':''} onClick={()=>setMode(x)}>{x}</button>)}</div><div className="upload-zone" onClick={()=>setFile('cartosat_aoi_upload.tif')}><Upload size={22} className="mx-auto mb-2 text-cyan"/><p className="text-sm">Drop imagery or browse</p><p className="mt-1 text-xs text-muted-foreground">GeoTIFF, TIFF, PNG, JPEG · up to 2 GB</p></div><div className="mt-4 rounded-2xl border border-white/10 bg-white/[.04] p-3"><div className="flex items-center gap-2"><div className="thumb-satellite"/><div className="min-w-0"><p className="truncate text-sm">{file}</p><p className="text-xs text-muted-foreground">Sentinel-2 · 13 bands · 10 m</p></div></div><div className="mt-3 flex items-center justify-between"><span className="text-xs text-muted-foreground">CRS EPSG:32643</span><Pill tone="green"><span className="status-dot"/> validated</Pill></div></div><div className="mt-5"><div className="eyebrow mb-3">RECENT SESSIONS</div>{['Sabarmati corridor · change','Kutch salt pans · grounding','Pune urban edge · caption'].map((x,i)=><button key={x} className="session-row"><span className="thumb-satellite small"/><span>{x}</span><span className="text-xs text-muted-foreground">{i+2}m</span></button>)}</div></Glass><Glass className="flex min-h-[640px] flex-col p-5"><div className="mb-6 flex items-center justify-between"><div><div className="eyebrow">NATURAL LANGUAGE QUERY</div><p className="mt-1 text-sm text-muted-foreground">No model selection required. The agent routes your task.</p></div><Pill>agentic mode</Pill></div><div className="flex-1 space-y-5"><div className="flex justify-end"><div className="user-bubble">{query || 'What changed around the Sabarmati corridor between these two dates?'}</div></div>{running ? <Glass className="mx-auto max-w-md p-5"><div className="mb-4 flex items-center justify-between"><span className="font-medium">Agent is working</span><span className="text-xs text-cyan">step 4 / 6</span></div><AgentStepper active={3}/><div className="shimmer mt-5 h-1 rounded-full"/></Glass> : done ? <div className="response-card"><div className="mb-3 flex items-center gap-2"><Pill tone="green">complete</Pill><span className="text-xs text-muted-foreground">4.21s · 2 specialists</span></div><p className="text-lg leading-relaxed">The primary change is a <strong>new linear construction footprint</strong> along the eastern Sabarmati embankment. The detected region spans approximately 1.8 ha, with a confidence of 92%. Vegetation cover decreased by 14% within the AOI.</p><div className="mt-5"><Confidence/></div></div> : <div className="empty-state"><div className="orbital-icon"><Satellite size={28}/></div><p className="mt-4 text-sm text-muted-foreground">Your grounded answer, evidence, and execution trace will appear here.</p></div>}</div><div className="mt-6"><div className="mb-3 flex flex-wrap gap-2">{['Describe this scene','Locate the airstrip','What changed between dates?','Compare optical and SAR'].map(x=><button key={x} className="query-chip" onClick={()=>setQuery(x)}>{x}</button>)}</div><div className="query-box"><input value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>{if(e.key==='Enter')run()}} placeholder="Ask about this imagery…" aria-label="Natural language query"/><button className="button-primary" onClick={run}>{running?'Running…':'Ask'} <ArrowRight size={15}/></button></div><p className="mt-2 text-center text-[11px] text-muted-foreground">SatQuery can make mistakes. Validate critical findings against source imagery.</p></div></Glass><Glass className="p-4"><div className="mb-4 flex items-center justify-between"><h2 className="font-medium">Grounded results</h2><Pill tone={done?'green':'muted'}>{done?'live':'awaiting query'}</Pill></div><Evidence pair={mode==='Two-Date Pair'}/><div className="mt-4 grid grid-cols-2 gap-2"><button className="control-btn"><Layers3 size={14}/> overlay</button><button className="control-btn"><ScanSearch size={14}/> zoom 100%</button></div><div className="my-5 border-t border-white/10"/><div className="eyebrow mb-3">EVIDENCE SUMMARY</div><div className="space-y-3 text-sm"><div className="flex justify-between"><span className="text-muted-foreground">AOI</span><span>Sabarmati / Ahmedabad</span></div><div className="flex justify-between"><span className="text-muted-foreground">Change area</span><span>1.8 ha</span></div><div className="flex justify-between"><span className="text-muted-foreground">Co-registration</span><span className="text-cyan">98%</span></div></div><div className="my-5 border-t border-white/10"/><details open><summary className="mb-4 flex cursor-pointer list-none items-center justify-between text-sm font-medium">Execution trace <ChevronDown size={15}/></summary><TraceViewer/></details><button className="button-secondary mt-5 w-full justify-center" onClick={()=>alert('Report export queued') }><Download size={15}/> Download report</button></Glass></div></main></Shell> }

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

export function Registry() { const models=[['S1','Single-image VLM','RSVQA · VRSBench','Implemented / Tested','92%'],['S2','Grounding specialist','RefCOCO · AirBus','Implemented / Tested','0.81 IoU'],['S3','Change detection','CDVQA · LEVIR-CD','Implemented / Tested','87% OA'],['S4','Optical–SAR fusion','BigEarthNet-MM','Partial','84% mAP'],['S5','Segmentation','LoveDA · SpaceNet','Not done','—']]; return <Shell><main className="mx-auto max-w-7xl px-5 py-20"><div className="eyebrow">SATQUERY / MODEL REGISTRY</div><h1 className="mt-3 max-w-2xl text-5xl font-semibold tracking-tight">Specialists, routed by intent.</h1><p className="mt-5 max-w-xl text-lg leading-8 text-muted-foreground">A transparent capability layer for remote-sensing workflows. Every answer exposes which model ran and why.</p><div className="mt-12 overflow-hidden rounded-3xl border border-white/10 bg-white/[.04]"><div className="hidden grid-cols-[80px_1fr_1fr_180px_100px] gap-4 border-b border-white/10 px-6 py-4 text-xs uppercase tracking-wider text-muted-foreground md:grid"><span>ID</span><span>Specialist</span><span>Benchmarks</span><span>Status</span><span>Metric</span></div>{models.map(m=><div key={m[0]} className="grid gap-3 border-b border-white/10 px-6 py-5 last:border-0 md:grid-cols-[80px_1fr_1fr_180px_100px] md:items-center"><span className="font-mono text-cyan">{m[0]}</span><span className="font-medium">{m[1]}</span><span className="text-sm text-muted-foreground">{m[2]}</span><span><Pill tone={m[3].startsWith('Implemented')?'green':m[3]==='Partial'?'orange':'muted'}>{m[3]}</Pill></span><span className="text-sm">{m[4]}</span></div>)}</div></main></Shell> }

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
