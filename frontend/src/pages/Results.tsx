import { Link } from 'react-router-dom'
import { useEvaluationStore } from '../store/useEvaluationStore'

export default function Results() {
  const { feedbackResult, reset } = useEvaluationStore()

  if (!feedbackResult) {
    return (
      <div className="text-center py-20">
        <p className="text-gray-500">No results yet.</p>
        <Link to="/evaluate" className="mt-4 inline-block text-blue-600 underline">
          Evaluate an answer
        </Link>
      </div>
    )
  }

  const percentage = Math.round((feedbackResult.final_score / feedbackResult.total_marks) * 100)

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Evaluation Results</h2>

      {/* Score Card */}
      <div className="bg-blue-600 text-white rounded-xl p-6 flex items-center justify-between">
        <div>
          <p className="text-sm opacity-75">Final Score</p>
          <p className="text-5xl font-bold mt-1">
            {feedbackResult.final_score}
            <span className="text-2xl font-normal opacity-75"> / {feedbackResult.total_marks}</span>
          </p>
        </div>
        <div className="text-right">
          <p className="text-4xl font-bold">{percentage}%</p>
        </div>
      </div>

      {/* Step Feedback */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <h3 className="font-semibold text-gray-800 mb-4">Step-by-Step Feedback</h3>
        <div className="space-y-3">
          {feedbackResult.step_feedback?.map((step: any) => (
            <div
              key={step.step_number}
              className={`p-4 rounded-lg border-l-4 ${
                step.is_correct
                  ? 'border-green-500 bg-green-50'
                  : 'border-red-400 bg-red-50'
              }`}
            >
              <div className="flex justify-between items-center mb-1">
                <span className="font-medium text-gray-700 text-sm">
                  Step {step.step_number}
                  <span className="ml-2">
                    {step.is_correct ? '✓' : '✗'}
                  </span>
                </span>
                <span className="text-xs text-gray-500 bg-white px-2 py-0.5 rounded-full border">
                  {step.marks_awarded} marks
                </span>
              </div>
              <p className="text-sm font-mono text-gray-600 mb-1">{step.expression}</p>
              <p className="text-sm text-gray-700">{step.feedback}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Overall Feedback */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <h3 className="font-semibold text-gray-800 mb-3">Overall Feedback</h3>
        <p className="text-gray-600 text-sm leading-relaxed">{feedbackResult.overall_feedback}</p>
      </div>

      {/* Improvement Suggestions */}
      {feedbackResult.improvement_suggestions?.length > 0 && (
        <div className="bg-yellow-50 rounded-xl border border-yellow-200 p-6">
          <h3 className="font-semibold text-gray-800 mb-3">Suggestions for Improvement</h3>
          <ul className="space-y-2">
            {feedbackResult.improvement_suggestions.map((s: string, i: number) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="text-yellow-500 mt-0.5">→</span>
                {s}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex gap-4">
        <Link
          to="/evaluate"
          onClick={reset}
          className="bg-blue-600 text-white px-6 py-2.5 rounded-lg hover:bg-blue-700 transition text-sm font-medium"
        >
          Evaluate Another
        </Link>
        <Link
          to="/"
          className="bg-gray-100 text-gray-700 px-6 py-2.5 rounded-lg hover:bg-gray-200 transition text-sm font-medium"
        >
          Back to Home
        </Link>
      </div>
    </div>
  )
}
