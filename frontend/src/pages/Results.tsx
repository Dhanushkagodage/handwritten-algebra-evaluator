import { Link } from 'react-router-dom'
import ResultsPanel from '../components/results/ResultsPanel'
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

  return (
    <div className="max-w-3xl mx-auto py-4">
      <ResultsPanel
        result={result}
        footer={
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
        }
      />
    </div>
  )
}
