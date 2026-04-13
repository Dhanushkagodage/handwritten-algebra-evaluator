import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="max-w-3xl mx-auto text-center py-20">
      <h1 className="text-4xl font-bold text-gray-800 mb-4">
        AI-Based Algebra Answer Evaluator
      </h1>
      <p className="text-gray-500 text-lg mb-10">
        Upload a handwritten A/L algebra answer sheet and get instant
        step-by-step feedback powered by AI.
      </p>
      <div className="flex justify-center gap-4">
        <Link
          to="/evaluate"
          className="bg-blue-600 text-white px-8 py-3 rounded-lg text-base font-medium hover:bg-blue-700 transition"
        >
          Start Evaluation
        </Link>
      </div>

      <div className="mt-16 grid grid-cols-3 gap-6 text-left">
        {[
          { title: 'OCR Extraction', desc: 'Extracts handwritten math steps from scanned answer sheets.' },
          { title: 'Step-by-Step Reasoning', desc: 'Validates each algebraic step and detects the solution method.' },
          { title: 'Explainable Feedback', desc: 'Generates clear, student-friendly feedback using a fine-tuned SLM.' },
        ].map((card) => (
          <div key={card.title} className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <h3 className="font-semibold text-gray-800 mb-2">{card.title}</h3>
            <p className="text-gray-500 text-sm">{card.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
