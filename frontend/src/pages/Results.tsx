import { Link } from 'react-router-dom'
import StepCard from '../components/results/StepCard'
import { useEvaluationStore } from '../store/useEvaluationStore'

export default function Results() {
  const result = useEvaluationStore((state) => state.result)
  const reset = useEvaluationStore((state) => state.reset)

  if (!result) {
    return (
      <div className="text-center py-24">
        <p className="text-gray-400 text-sm mb-3">No results to display.</p>
        <Link to="/evaluate" className="text-sm text-blue-600 hover:underline font-medium">
          Start an evaluation →
        </Link>
      </div>
    )
  }

  // The gateway backfills a missing scheme total, but guard anyway — an
  // unguarded 0/0 renders as "NaN%".
  const percentage =
    result.total_marks > 0 ? Math.round((result.final_score / result.total_marks) * 100) : null
  const correctCount = result.step_feedback.filter((s) => s.validity === 'correct').length
  const totalSteps = result.step_feedback.length
  const reasoning = result.questions[0]?.reasoning ?? null

  // Steps OCR read but the feedback model produced nothing for — usually the
  // SLM's "=== STEP n ===" blocks failed to parse.
  const gradedNumbers = new Set(result.step_feedback.map((s) => s.step_number))
  const ungraded = (result.questions[0]?.student_steps ?? []).filter(
    (s) => !gradedNumbers.has(s.step_id),
  )

  return (
    <div className="max-w-3xl mx-auto py-4 space-y-4">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Evaluation Results</h2>
        {result.question_text && (
          <p className="text-sm text-gray-400 mt-1">{result.question_text}</p>
        )}
      </div>

      {/* Anything the pipeline had to infer or guess about. */}
      {result.warnings.length > 0 && (
        <div className="bg-amber-50/70 border border-amber-100 rounded-xl p-4">
          <p className="text-xs font-semibold text-amber-700 mb-2">Worth checking</p>
          <ul className="space-y-1">
            {result.warnings.map((warning, i) => (
              <li key={i} className="text-xs text-amber-700/80 leading-relaxed">
                • {warning}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Score row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2 bg-white rounded-xl border border-gray-100 shadow-sm p-5 flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center flex-shrink-0">
            <span className="text-lg font-bold text-blue-600">
              {percentage === null ? '—' : `${percentage}%`}
            </span>
          </div>
          <div>
            <p className="text-xs text-gray-400 font-medium mb-0.5">Final Score</p>
            <p className="text-2xl font-bold text-gray-900">
              {result.final_score}
              <span className="text-base font-normal text-gray-400"> / {result.total_marks}</span>
            </p>
          </div>
        </div>
        <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5 flex flex-col justify-center">
          <p className="text-xs text-gray-400 font-medium mb-0.5">Steps Correct</p>
          <p className="text-2xl font-bold text-gray-900">
            {correctCount}
            <span className="text-base font-normal text-gray-400"> / {totalSteps}</span>
          </p>
        </div>
      </div>

      {/* Module 02's output — the method it detected and how it marked. */}
      {reasoning && (
        <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
          <p className="text-sm font-semibold text-gray-800 mb-3">Method & Marking</p>
          <div className="space-y-2 text-sm text-gray-600 leading-relaxed">
            {reasoning.method_detection?.detected_method && (
              <p>
                <span className="text-gray-400">Detected method: </span>
                <span className="font-medium text-gray-700">
                  {reasoning.method_detection.detected_method}
                </span>
              </p>
            )}
            {reasoning.summary && <p>{reasoning.summary}</p>}
            {reasoning.method_feedback && (
              <p className="text-gray-500">{reasoning.method_feedback}</p>
            )}
            {reasoning.missing_steps_feedback && (
              <p className="text-amber-600">{reasoning.missing_steps_feedback}</p>
            )}
          </div>
        </div>
      )}

      {/* Step feedback */}
      <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
        <p className="text-sm font-semibold text-gray-800 mb-4">Step-by-Step Feedback</p>
        <div className="space-y-2">
          {result.step_feedback.map((step) => (
            <StepCard key={step.step_number} step={step} />
          ))}

          {ungraded.map((step) => (
            <div key={`ungraded-${step.step_id}`} className="rounded-lg border border-gray-100 bg-gray-50/60 p-4">
              <p className="text-sm font-medium text-gray-500 mb-1">Step {step.step_id}</p>
              <p className="text-xs font-mono text-gray-400 bg-white/80 rounded px-2 py-1 mb-2 inline-block">
                {step.content}
              </p>
              <p className="text-xs text-gray-400">
                This step was read from your answer sheet, but no feedback was generated for it.
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Overall feedback */}
      <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
        <p className="text-sm font-semibold text-gray-800 mb-2">Overall Feedback</p>
        <p className="text-sm text-gray-500 leading-relaxed">{result.overall_feedback}</p>
      </div>

      {/* Improvement suggestions */}
      {result.improvement_suggestions.length > 0 && (
        <div className="bg-amber-50 rounded-xl border border-amber-100 p-5">
          <p className="text-sm font-semibold text-gray-800 mb-3">Suggestions for Improvement</p>
          <ul className="space-y-1.5">
            {result.improvement_suggestions.map((suggestion, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                <span className="text-amber-500 mt-0.5 flex-shrink-0">→</span>
                {suggestion}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex gap-3 items-center pb-4">
        <Link
          to="/evaluate"
          onClick={reset}
          className="bg-blue-600 text-white px-5 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition"
        >
          Evaluate Another
        </Link>
        <Link
          to="/"
          className="bg-gray-100 text-gray-600 px-5 py-2 rounded-lg text-sm font-medium hover:bg-gray-200 transition"
        >
          Back to Home
        </Link>
        {result.timings_ms.total != null && (
          <span className="text-xs text-gray-300 ml-auto">
            Evaluated in {(result.timings_ms.total / 1000).toFixed(1)}s
          </span>
        )}
      </div>
    </div>
  )
}
