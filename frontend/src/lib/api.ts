import axios, { type AxiosProgressEvent } from 'axios'
import type { EvaluationJob, JobCreated, StartEvaluationInput } from '../types/api'

/**
 * The browser talks to exactly one backend: the gateway, which chains OCR ->
 * reasoning -> feedback server-side. A blank VITE_GATEWAY_URL means same-origin,
 * which the Vite dev proxy in vite.config.ts then forwards to :8080.
 */
const RAW_GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL?.trim()
export const GATEWAY_ORIGIN = RAW_GATEWAY_URL ? RAW_GATEWAY_URL.replace(/\/+$/, '') : ''

// 30s is generous for every call we make: starting a job returns as soon as the
// images finish uploading, and polls are instant. No request waits for the
// pipeline itself, which can take minutes.
const client = axios.create({
  baseURL: `${GATEWAY_ORIGIN}/api/v1`,
  timeout: 30_000,
})

export type ApiErrorKind = 'canceled' | 'timeout' | 'network' | 'gateway' | 'http'

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly stage: string | null
  readonly code: string | null
  readonly status: number | null
  readonly details: unknown

  constructor(init: {
    message: string
    kind: ApiErrorKind
    stage?: string | null
    code?: string | null
    status?: number | null
    details?: unknown
  }) {
    super(init.message)
    this.name = 'ApiError'
    this.kind = init.kind
    this.stage = init.stage ?? null
    this.code = init.code ?? null
    this.status = init.status ?? null
    this.details = init.details ?? null
  }
}

/** Normalise anything axios can throw into one shape the UI can render. */
export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err

  if (axios.isCancel(err)) {
    return new ApiError({ message: 'The evaluation was cancelled.', kind: 'canceled' })
  }

  if (axios.isAxiosError(err)) {
    if (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT') {
      return new ApiError({
        message: 'The gateway took too long to respond. Check it is still running.',
        kind: 'timeout',
      })
    }

    if (!err.response) {
      // With VITE_GATEWAY_URL unset we call our own origin and let the Vite dev
      // proxy forward /api to :8080. Naming the page's own URL here would point
      // the reader at the wrong port, so describe the real target instead.
      const where = GATEWAY_ORIGIN
        ? GATEWAY_ORIGIN
        : 'http://127.0.0.1:8080 (via the Vite dev proxy)'
      return new ApiError({
        message:
          `Couldn't reach the gateway at ${where}. Start the backend with ` +
          `scripts\\dev.ps1, or run "uvicorn app.main:app --port 8080" in services/gateway. ` +
          `First time on this machine? Each service also needs its own .env file — ` +
          `those are gitignored, so they do not arrive with a git pull.`,
        kind: 'network',
      })
    }

    const status = err.response.status
    const body = err.response.data as Record<string, unknown> | undefined

    // The gateway's own envelope, which names the module that failed.
    if (body && typeof body.message === 'string' && typeof body.error_code === 'string') {
      return new ApiError({
        message: body.message,
        kind: 'gateway',
        stage: typeof body.stage === 'string' ? body.stage : null,
        code: body.error_code,
        status,
        details: body.details,
      })
    }

    // A bare FastAPI response: `detail` is a string for HTTPException but an
    // array of {loc, msg, type} for a 422 — the shape most likely to appear
    // while contracts are still settling.
    const detail = body?.detail
    if (typeof detail === 'string') {
      return new ApiError({ message: detail, kind: 'http', status })
    }
    if (Array.isArray(detail)) {
      const first = detail[0] as { loc?: unknown[]; msg?: string } | undefined
      const where = Array.isArray(first?.loc) ? first.loc.join('.') : ''
      return new ApiError({
        message: where ? `${where}: ${first?.msg ?? 'invalid'}` : (first?.msg ?? 'Invalid request.'),
        kind: 'http',
        status,
        details: detail,
      })
    }

    return new ApiError({ message: `Request failed with status ${status}.`, kind: 'http', status })
  }

  return new ApiError({
    message: err instanceof Error ? err.message : 'Something went wrong.',
    kind: 'http',
  })
}

export interface StartEvaluationOptions {
  signal?: AbortSignal
  onUploadProgress?: (percent: number) => void
}

/** Start the pipeline. Returns as soon as the images have uploaded. */
export async function startEvaluation(
  input: StartEvaluationInput,
  options: StartEvaluationOptions = {},
): Promise<JobCreated> {
  const form = new FormData()
  input.answerImages.forEach((file) => form.append('answer_images', file))
  form.append('marking_scheme_image', input.schemeImage)
  if (input.questionText?.trim()) {
    form.append('question_text', input.questionText.trim())
  }

  try {
    const { data } = await client.post<JobCreated>('/jobs', form, {
      signal: options.signal,
      onUploadProgress: (event: AxiosProgressEvent) => {
        if (options.onUploadProgress && event.total) {
          options.onUploadProgress(Math.round((event.loaded / event.total) * 100))
        }
      },
    })
    return data
  } catch (err) {
    throw toApiError(err)
  }
}

/** Poll a job. Answers 200 even for a failed run — check `status`. */
export async function getEvaluation(jobId: string): Promise<EvaluationJob> {
  try {
    const { data } = await client.get<EvaluationJob>(`/jobs/${jobId}`)
    return data
  } catch (err) {
    throw toApiError(err)
  }
}

export async function cancelEvaluation(jobId: string): Promise<void> {
  try {
    await client.delete(`/jobs/${jobId}`)
  } catch {
    // Best-effort: the job may already have finished.
  }
}
