import type { ApiError } from '../lib/api'

/**
 * Naming the module that failed is exactly right for a three-owner project:
 * the error tells you who fixes it.
 */
const MODULE_BY_STAGE: Record<string, string> = {
  ocr: 'OCR Extraction (Module 01)',
  reasoning: 'Reasoning & Marking (Module 02)',
  feedback: 'Feedback Generation (Module 03)',
}

/** Failures we know will happen, translated out of backend language. */
function explain(error: ApiError): string | null {
  const message = error.message.toLowerCase()

  if (error.code === 'SERVICE_UNAVAILABLE') {
    return 'That service is not running. Start the whole stack with scripts\\dev.ps1.'
  }
  if (message.includes('openai_api_key')) {
    return 'That service has no OpenAI key configured. Copy its .env.example to .env and fill in OPENAI_API_KEY.'
  }
  if (error.code === 'MARKING_SCHEME_INVALID') {
    return 'The marking scheme could not be read from that image. Check the photo is sharp, upright, and shows the whole scheme including the marks.'
  }
  if (error.code === 'NO_STEPS_EXTRACTED') {
    return 'No working steps were readable in the answer sheet. Try a clearer, better-lit photo.'
  }
  if (error.code === 'UPSTREAM_TIMEOUT') {
    return 'That stage took longer than its configured timeout. It may still finish — try again in a moment.'
  }

  // A 422 means the service rejected the request shape, which almost always
  // means that service is running older code than the repo. Never blame a cold
  // start for this — the two need completely different fixes.
  const details = error.details as { upstream_status?: number } | null
  if (details?.upstream_status === 422) {
    return 'That service rejected the request format. It is probably running an older version of its code than the repo — restart it so it picks up the current schema.'
  }

  if (error.stage === 'feedback') {
    // The feedback model runs on a Hugging Face Space that sleeps when idle.
    return 'The feedback model runs on a Hugging Face Space that cold-starts when idle. Wait a minute and try again — the second attempt is usually much faster.'
  }
  return null
}

export default function ErrorPanel({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const where = error.stage ? MODULE_BY_STAGE[error.stage] : null
  const hint = explain(error)

  return (
    <div className="bg-red-50 border border-red-100 rounded-xl px-5 py-4 mb-4">
      <p className="text-sm font-semibold text-red-700 mb-1">
        {where ? `Failed during ${where}` : 'Evaluation failed'}
      </p>
      <p className="text-sm text-red-600 leading-relaxed">{error.message}</p>

      {hint && <p className="text-xs text-red-500/90 mt-2 leading-relaxed">{hint}</p>}

      {(error.code || error.status || error.details != null) && (
        <details className="mt-3">
          <summary className="text-xs text-red-400 cursor-pointer hover:text-red-500">
            Technical details
          </summary>
          <pre className="mt-2 text-xs text-red-500/80 bg-white/60 rounded p-2 overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(
              { code: error.code, status: error.status, stage: error.stage, details: error.details },
              null,
              2,
            )}
          </pre>
        </details>
      )}

      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 text-xs font-medium text-red-600 hover:text-red-700 underline"
        >
          Try again
        </button>
      )}
    </div>
  )
}
