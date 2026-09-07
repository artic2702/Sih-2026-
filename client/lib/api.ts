// ---------------------------------------------------------------------------
// SatQuery AI — TypeScript API Client
// Typed interface to the FastAPI ML backend at /api/v1/*
// ---------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : '/api/v1'

// ---------------------------------------------------------------------------
// Response Types (mirroring src/api/schemas.py)
// ---------------------------------------------------------------------------

export interface SpatialEvidence {
  type: 'bbox' | 'mask' | 'none'
  source: string
  bbox: number[] | null
  mask: {
    shape: number[]
    dtype: string
    nonzero_pixels: number
    changed_ratio: number
  } | null
}

export interface ResultMetadata {
  model: string
  backbone: string
  checkpoint: string
  dataset: string
  input_modalities: string[]
  parameters: Record<string, unknown>
}

export interface VerificationReport {
  passed: boolean
  decision: 'SUPPORTED' | 'FALLBACK_GEE' | 'UNVERIFIED'
  reasons: string[]
  checks: Record<string, unknown>
  gee_evidence: Record<string, unknown> | null
}

export interface AgentResponse {
  success: boolean
  task: string
  text: string
  confidence: number
  spatial_evidence: SpatialEvidence | null
  metadata: ResultMetadata | null
  verification: VerificationReport
  status: string
  errors: string[]
  trace: Record<string, unknown>
}

export interface RSModelResult {
  task: string
  text: string
  confidence: number
  spatial_evidence: SpatialEvidence
  metadata: ResultMetadata
  status: string
  inference_seconds: number
}

export interface HealthResponse {
  status: string
  version: string
  device: string
  models_available: string[]
  timestamp: string
}

export interface ModelInfo {
  task: string
  name: string
  backbone: string
  loaded: boolean
}

export interface ModelsListResponse {
  models: ModelInfo[]
}

// ---------------------------------------------------------------------------
// API Error
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      detail = await res.text().catch(() => detail)
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Health & Model Registry
// ---------------------------------------------------------------------------

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, {
    method: 'GET',
    signal: AbortSignal.timeout(5000),
  })
  return handleResponse<HealthResponse>(res)
}

export async function getModels(): Promise<ModelsListResponse> {
  const res = await fetch(`${API_BASE}/models`, {
    method: 'GET',
    signal: AbortSignal.timeout(8000),
  })
  return handleResponse<ModelsListResponse>(res)
}

// ---------------------------------------------------------------------------
// Agent Query (Multipart File Upload)
// ---------------------------------------------------------------------------

export interface AgentQueryParams {
  query: string
  mode: 'single' | 'optical_sar' | 'bitemporal'
  /** Single image file */
  image?: File
  /** Optical image for fusion */
  imageOptical?: File
  /** SAR image for fusion */
  imageSar?: File
  /** T1 (before) image for change detection */
  imageT1?: File
  /** T2 (after) image for change detection */
  imageT2?: File
}

export async function queryAgent(params: AgentQueryParams): Promise<AgentResponse> {
  const formData = new FormData()
  formData.append('query', params.query)

  if (params.mode === 'bitemporal') {
    if (!params.imageT1 || !params.imageT2) {
      throw new ApiError(400, 'Bi-temporal mode requires both T1 and T2 images')
    }
    formData.append('image_t1', params.imageT1)
    formData.append('image_t2', params.imageT2)
  } else if (params.mode === 'optical_sar') {
    if (!params.imageOptical || !params.imageSar) {
      throw new ApiError(400, 'Optical+SAR mode requires both optical and SAR images')
    }
    formData.append('image_optical', params.imageOptical)
    formData.append('image_sar', params.imageSar)
  } else {
    if (!params.image) {
      throw new ApiError(400, 'Single image mode requires an image file')
    }
    formData.append('image', params.image)
  }

  const res = await fetch(`${API_BASE}/agent/upload-query`, {
    method: 'POST',
    body: formData,
    // No Content-Type header — browser sets multipart boundary automatically
  })
  return handleResponse<AgentResponse>(res)
}

// ---------------------------------------------------------------------------
// Direct Specialist Endpoints (Multipart)
// ---------------------------------------------------------------------------

export async function queryVQA(image: File, query: string): Promise<RSModelResult> {
  const fd = new FormData()
  fd.append('image', image)
  fd.append('query', query)
  const res = await fetch(`${API_BASE}/models/vqa/upload`, { method: 'POST', body: fd })
  return handleResponse<RSModelResult>(res)
}

export async function queryCaptioning(image: File, query?: string): Promise<RSModelResult> {
  const fd = new FormData()
  fd.append('image', image)
  if (query) fd.append('query', query)
  const res = await fetch(`${API_BASE}/models/captioning/upload`, { method: 'POST', body: fd })
  return handleResponse<RSModelResult>(res)
}

export async function queryGrounding(image: File, query: string): Promise<RSModelResult> {
  const fd = new FormData()
  fd.append('image', image)
  fd.append('query', query)
  const res = await fetch(`${API_BASE}/models/grounding/upload`, { method: 'POST', body: fd })
  return handleResponse<RSModelResult>(res)
}

export async function queryChange(imageT1: File, imageT2: File, query?: string): Promise<RSModelResult> {
  const fd = new FormData()
  fd.append('image_t1', imageT1)
  fd.append('image_t2', imageT2)
  if (query) fd.append('query', query)
  const res = await fetch(`${API_BASE}/models/change/upload`, { method: 'POST', body: fd })
  return handleResponse<RSModelResult>(res)
}

export async function queryFusion(imageOptical: File, imageSar: File, query?: string): Promise<RSModelResult> {
  const fd = new FormData()
  fd.append('image_optical', imageOptical)
  fd.append('image_sar', imageSar)
  if (query) fd.append('query', query)
  const res = await fetch(`${API_BASE}/models/fusion/upload`, { method: 'POST', body: fd })
  return handleResponse<RSModelResult>(res)
}
