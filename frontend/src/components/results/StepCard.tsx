import type { StepFeedback, StepValidity } from '../../types/api'

/**
 * Three states, not two. A Record keyed by StepValidity means a new value on
 * the backend is a compile error rather than a silently mis-coloured card.
 */
const STYLES: Record<StepValidity, { wrap: string; badge: string; icon: string; label: string }> = {
  correct: {
    wrap: 'border-green-100 bg-green-50/50',
    badge: 'bg-green-100 text-green-600',
    icon: '✓',
    label: 'Correct',
  },
  partial: {
    wrap: 'border-amber-100 bg-amber-50/50',
    badge: 'bg-amber-100 text-amber-600',
    icon: '~',
    label: 'Partially correct',
  },
  incorrect: {
    wrap: 'border-red-100 bg-red-50/50',
    badge: 'bg-red-100 text-red-500',
    icon: '✗',
    label: 'Incorrect',
  },
}

function FeedbackPart({
  marker,
  markerClass,
  label,
  text,
}: {
  marker: string
  markerClass: string
  label: string
  text: string
}) {
  return (
    <div className="flex items-start gap-2">
      <span className={`mt-0.5 flex-shrink-0 text-xs font-bold ${markerClass}`}>{marker}</span>
      <p className="text-sm text-gray-600 leading-relaxed">
        <span className="text-gray-400 font-medium">{label} </span>
        {text}
      </p>
    </div>
  )
}

export default function StepCard({ step }: { step: StepFeedback }) {
  const style = STYLES[step.validity] ?? STYLES.incorrect

  return (
    <div className={`rounded-lg border p-4 ${style.wrap}`}>
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-2">
          <span
            className={`w-5 h-5 rounded flex items-center justify-center text-xs font-bold flex-shrink-0 ${style.badge}`}
          >
            {style.icon}
          </span>
          <span className="text-sm font-medium text-gray-700">Step {step.step_number}</span>
          <span className="text-xs text-gray-400">{style.label}</span>
        </div>
        <span className="text-xs text-gray-500 bg-white px-2 py-0.5 rounded border border-gray-100">
          {step.marks_awarded} marks
        </span>
      </div>

      {step.expression && (
        <p className="text-xs font-mono text-gray-500 bg-white/80 rounded px-2 py-1 mb-3 inline-block">
          {step.expression}
        </p>
      )}

      {/* The four-component feedback structure is Module 03's core output. */}
      <div className="space-y-1.5">
        {step.what_is_correct && (
          <FeedbackPart
            marker="✓"
            markerClass="text-green-500"
            label="What's correct:"
            text={step.what_is_correct}
          />
        )}
        {step.what_is_missing && (
          <FeedbackPart
            marker="✗"
            markerClass="text-red-400"
            label="What's missing:"
            text={step.what_is_missing}
          />
        )}
        {step.why_marks_reduced && (
          <FeedbackPart
            marker="!"
            markerClass="text-amber-500"
            label="Why marks were reduced:"
            text={step.why_marks_reduced}
          />
        )}
        {step.how_to_improve && (
          <FeedbackPart
            marker="→"
            markerClass="text-blue-500"
            label="How to improve:"
            text={step.how_to_improve}
          />
        )}

        {/* Fall back to the combined summary if the SLM's four-part blocks
            failed to parse and only the one-liner came through. */}
        {!step.what_is_correct && !step.how_to_improve && step.feedback && (
          <p className="text-sm text-gray-600 leading-relaxed">{step.feedback}</p>
        )}
      </div>
    </div>
  )
}
