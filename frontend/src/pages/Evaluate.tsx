import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { processImage, analyzeReasoning, generateFeedback } from '../lib/api'
import { useEvaluationStore } from '../store/useEvaluationStore'

export default function Evaluate() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState('')
  const [error, setError] = useState<string | null>(null)
  const { setOcrResult, setReasoningResult, setFeedbackResult } = useEvaluationStore()
  const navigate = useNavigate()

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/*': [] },
    maxFiles: 1,
    onDrop: (files) => setFile(files[0]),
  })

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    try {
      setStep('Extracting handwritten steps (OCR)...')
      const ocr = await processImage(file)
      setOcrResult(ocr)

      setStep('Analyzing reasoning and marking...')
      const reasoning = await analyzeReasoning({
        question_text: ocr.question_text,
        student_steps: ocr.student_steps,
        marking_scheme: [],
        total_marks: 5,
      })
      setReasoningResult(reasoning)

      setStep('Generating step-by-step feedback...')
      const feedback = await generateFeedback({
        question_text: reasoning.question_text,
        student_steps: reasoning.step_analysis,
        detected_method: reasoning.detected_method,
        assigned_marks: reasoning.assigned_marks,
        total_marks: reasoning.total_marks,
        marking_scheme: reasoning.marking_scheme,
      })
      setFeedbackResult(feedback)

      navigate('/results')
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
      setStep('')
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Answer Sheet</h2>
      <p className="text-gray-500 text-sm mb-6">
        Upload a scanned image of the student's handwritten algebra answer.
      </p>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition ${
          isDragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 bg-white'
        }`}
      >
        <input {...getInputProps()} />
        {file ? (
          <div>
            <p className="text-green-600 font-medium">{file.name}</p>
            <p className="text-gray-400 text-sm mt-1">Click to replace</p>
          </div>
        ) : (
          <div>
            <p className="text-gray-400">Drag & drop an image here, or click to select</p>
            <p className="text-gray-300 text-sm mt-1">PNG, JPG supported</p>
          </div>
        )}
      </div>

      {file && (
        <div className="mt-4">
          <img
            src={URL.createObjectURL(file)}
            alt="Preview"
            className="rounded-lg max-h-48 object-contain border border-gray-200"
          />
        </div>
      )}

      {loading && (
        <div className="mt-4 text-sm text-blue-600 font-medium animate-pulse">{step}</div>
      )}

      {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}

      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        className="mt-6 w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
      >
        {loading ? 'Evaluating...' : 'Evaluate Answer'}
      </button>
    </div>
  )
}
