import { Link } from 'react-router-dom'
import { useEvaluationStore } from '../store/useEvaluationStore'

export default function Results() {
  const { feedbackResult, reset } = useEvaluationStore()

  if (!feedbackResult) {
    return (
      <div className="text-center py-24">
        <p className="text-gray-400 text-sm mb-3">No results to display.</p>
        <Link to="/evaluate" className="text-sm text-blue-600 hover:underline font-medium">
          Start an evaluation →
        </Link>
      </div>
    )
  }

  const percentage = Math.round((feedbackResult.final_score / feedbackResult.total_marks) * 100)
  const correctCount = feedbackResult.step_feedback?.filter((s: any) => s.is_correct).length ?? 0
  const totalSteps = feedbackResult.step_feedback?.length ?? 0

  return (
    <div className="max-w-3xl mx-auto py-4 space-y-4">
      <h2 className="text-xl font-bold text-gray-900">Evaluation Results</h2>

      {/* Score row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2 bg-white rounded-xl border border-gray-100 shadow-sm p-5 flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center flex-shrink-0">
            <span className="text-lg font-bold text-blue-600">{percentage}%</span>
          </div>
          <div>
            <p className="text-xs text-gray-400 font-medium mb-0.5">Final Score</p>
            <p className="text-2xl font-bold text-gray-900">
              {feedbackResult.final_score}
              <span className="text-base font-normal text-gray-400"> / {feedbackResult.total_marks}</span>
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

      {/* Step feedback */}
      <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
        <p className="text-sm font-semibold text-gray-800 mb-4">Step-by-Step Feedback</p>
        <div className="space-y-2">
          {feedbackResult.step_feedback?.map((step: any) => (
            <div
              key={step.step_number}
              className={`rounded-lg border p-4 ${
                step.is_correct
                  ? 'border-green-100 bg-green-50/50'
                  : 'border-red-100 bg-red-50/50'
              }`}
            >
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`w-5 h-5 rounded flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                      step.is_correct ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-500'
                    }`}
                  >
                    {step.is_correct ? '✓' : '✗'}
                  </span>
                  <span className="text-sm font-medium text-gray-700">Step {step.step_number}</span>
                </div>
                <span className="text-xs text-gray-500 bg-white px-2 py-0.5 rounded border border-gray-100">
                  {step.marks_awarded} marks
                </span>
              </div>
              {step.expression && (
                <p className="text-xs font-mono text-gray-500 bg-white/80 rounded px-2 py-1 mb-2 inline-block">
                  {step.expression}
                </p>
              )}
              <p className="text-sm text-gray-600 leading-relaxed">{step.feedback}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Overall feedback */}
      <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
        <p className="text-sm font-semibold text-gray-800 mb-2">Overall Feedback</p>
        <p className="text-sm text-gray-500 leading-relaxed">{feedbackResult.overall_feedback}</p>
      </div>

      {/* Improvement suggestions */}
      {feedbackResult.improvement_suggestions?.length > 0 && (
        <div className="bg-amber-50 rounded-xl border border-amber-100 p-5">
          <p className="text-sm font-semibold text-gray-800 mb-3">Suggestions for Improvement</p>
          <ul className="space-y-1.5">
            {feedbackResult.improvement_suggestions.map((s: string, i: number) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                <span className="text-amber-500 mt-0.5 flex-shrink-0">→</span>
                {s}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex gap-3 pb-4">
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
      </div>
    </div>
  )
}
