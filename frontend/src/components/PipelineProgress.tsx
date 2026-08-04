import type { PipelineStage } from '../types/api'

/**
 * Keys match the gateway's `stage` values exactly. Typing the array against
 * PipelineStage means a rename on the server becomes a compile error here
 * rather than a silently dead progress bar.
 */
const PIPELINE_STEPS: { key: PipelineStage; label: string }[] = [
  { key: 'ocr', label: 'OCR Extraction' },
  { key: 'reasoning', label: 'Reasoning & Marking' },
  { key: 'feedback', label: 'Generating Feedback' },
]

interface Props {
  stage: PipelineStage | null
  stageMessage: string | null
  uploadPercent: number
  elapsedSeconds: number
  failedStage?: string | null
  onCancel?: () => void
}

function formatElapsed(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

export default function PipelineProgress({
  stage,
  stageMessage,
  uploadPercent,
  elapsedSeconds,
  failedStage = null,
  onCancel,
}: Props) {
  const activeIdx = PIPELINE_STEPS.findIndex((s) => s.key === stage)
  const isDone = stage === 'done'

  return (
    <div className="bg-white rounded-xl border border-gray-100 px-5 py-4 mb-4">
      <div className="flex items-center gap-2 mb-3">
        {PIPELINE_STEPS.map((step, i) => {
          const failed = failedStage === step.key
          const done = isDone || (activeIdx >= 0 && i < activeIdx)
          const active = step.key === stage

          return (
            <div key={step.key} className="flex items-center gap-2">
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center text-xs flex-shrink-0 font-semibold ${
                  failed
                    ? 'bg-red-100 text-red-600'
                    : done
                      ? 'bg-green-100 text-green-600'
                      : active
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-400'
                }`}
              >
                {failed ? '!' : done ? '✓' : i + 1}
              </div>
              <span
                className={`text-xs font-medium ${
                  failed
                    ? 'text-red-600'
                    : done
                      ? 'text-green-600'
                      : active
                        ? 'text-blue-600'
                        : 'text-gray-400'
                }`}
              >
                {step.label}
              </span>
              {i < PIPELINE_STEPS.length - 1 && <div className="w-5 h-px bg-gray-200 mx-1" />}
            </div>
          )
        })}
      </div>

      {/* A multi-minute run with a static spinner reads as a hang, so say what
          is happening and how long it has been happening for. */}
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs text-gray-500 leading-relaxed">
          {stage === 'queued' && uploadPercent > 0 && uploadPercent < 100
            ? `Uploading images… ${uploadPercent}%`
            : (stageMessage ?? 'Starting…')}
        </p>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-gray-400 tabular-nums">{formatElapsed(elapsedSeconds)}</span>
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="text-xs text-gray-400 hover:text-red-500 transition"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
