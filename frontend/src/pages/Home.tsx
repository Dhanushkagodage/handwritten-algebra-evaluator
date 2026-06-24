import { Link } from 'react-router-dom'

const features = [
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    title: 'OCR Extraction',
    desc: 'Extracts handwritten algebra steps from scanned answer sheets with high precision.',
  },
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    title: 'Step-by-Step Reasoning',
    desc: 'Validates each algebraic step and detects the solution method used.',
  },
  {
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-3 3-3-3z" />
      </svg>
    ),
    title: 'Explainable Feedback',
    desc: 'Generates clear, student-friendly feedback using a fine-tuned language model.',
  },
]

const steps = [
  { number: '01', label: 'Upload', desc: 'Answer sheet & marking scheme' },
  { number: '02', label: 'Analyse', desc: 'OCR + reasoning pipeline' },
  { number: '03', label: 'Feedback', desc: 'Stepwise explanations' },
]

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto">
      {/* Hero */}
      <div className="text-center pt-16 pb-12">
        <span className="inline-block text-xs font-semibold tracking-widest text-blue-600 uppercase mb-4">
          AI-Powered Grading
        </span>
        <h1 className="text-4xl font-bold text-gray-900 leading-tight mb-4">
          Automated Evaluation of<br />Handwritten Algebra Answers
        </h1>
        <p className="text-gray-400 text-base max-w-xl mx-auto mb-8 leading-relaxed">
          Upload a handwritten A/L algebra answer sheet and receive instant
          step-by-step feedback aligned with your marking scheme.
        </p>
        <Link
          to="/evaluate"
          className="inline-flex items-center gap-2 bg-blue-600 text-white px-6 py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 transition"
        >
          Start Evaluation
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
          </svg>
        </Link>
      </div>

      {/* How it works */}
      <div className="border-t border-gray-100 pt-12 pb-10">
        <p className="text-xs font-semibold tracking-widest text-gray-400 uppercase text-center mb-8">
          How it works
        </p>
        <div className="flex items-start justify-center">
          {steps.map((s, i) => (
            <div key={s.number} className="flex items-start">
              <div className="text-center w-36">
                <div className="w-8 h-8 bg-blue-50 text-blue-600 rounded-lg text-xs font-bold flex items-center justify-center mx-auto mb-2">
                  {s.number}
                </div>
                <p className="font-semibold text-gray-800 text-sm">{s.label}</p>
                <p className="text-gray-400 text-xs mt-0.5">{s.desc}</p>
              </div>
              {i < steps.length - 1 && (
                <div className="mt-4 w-12 h-px bg-gray-200 mx-1 flex-shrink-0" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Feature cards */}
      <div className="grid grid-cols-3 gap-4 pb-16">
        {features.map((f) => (
          <div key={f.title} className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
            <div className="w-8 h-8 bg-gray-50 text-gray-500 rounded-lg flex items-center justify-center mb-3">
              {f.icon}
            </div>
            <h3 className="font-semibold text-gray-800 text-sm mb-1">{f.title}</h3>
            <p className="text-gray-400 text-xs leading-relaxed">{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
