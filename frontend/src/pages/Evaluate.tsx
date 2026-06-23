import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { processImage, analyzeReasoning, generateFeedback } from '../lib/api'
import { useEvaluationStore } from '../store/useEvaluationStore'

export default function Evaluate() {
  const [answerFile, setAnswerFile] = useState<File | null>(null)
  const [schemeFile, setSchemeFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState('')
  const [error, setError] = useState<string | null>(null)
  const { setOcrResult, setReasoningResult, setFeedbackResult } = useEvaluationStore()
  const navigate = useNavigate()

  const {
    getRootProps: getAnswerRootProps,
    getInputProps: getAnswerInputProps,
    isDragActive: isAnswerDragActive,
  } = useDropzone({
    accept: { 'image/*': [] },
    maxFiles: 1,
    onDrop: (files) => setAnswerFile(files[0]),
  })

  const {
    getRootProps: getSchemeRootProps,
    getInputProps: getSchemeInputProps,
    isDragActive: isSchemeDragActive,
  } = useDropzone({
    accept: { 'image/*': [] },
    maxFiles: 1,
    onDrop: (files) => setSchemeFile(files[0]),
  })

  const handleSubmit = async () => {
    if (!answerFile || !schemeFile) return
    setLoading(true)
    setError(null)

    try {
      // OCR both images in parallel
      setStep('Running OCR on both images...')
      const [answerOcr, schemeOcr] = await Promise.all([
        processImage(answerFile),
        processImage(schemeFile),
      ])
      setOcrResult(answerOcr)

      // Map marking scheme OCR steps → marking scheme format for reasoning service.
      // EasyOCR reads the whole row as one string, e.g. "x^2 + x - 6 = 0  1 mark"
      // so we parse the mark value out of the end of the expression.
      const parseSchemeStep = (s: any) => {
        const expr: string = s.expression || ''
        // Match trailing "N mark" or "N marks" (case-insensitive)
        const m = expr.match(/^(.*?)\s+(\d+)\s+marks?$/i)
        if (m) {
          return { expression: m[1].trim(), marks: parseInt(m[2]) }
        }
        return { expression: expr, marks: 1 }
      }
      const markingScheme = schemeOcr.student_steps.map((s: any) => {
        const { expression, marks } = parseSchemeStep(s)
        return { step_no: s.step_number, description: expression, marks }
      })
      const totalMarks = markingScheme.reduce((sum: number, m: any) => sum + m.marks, 0) || 5

      setStep('Analyzing reasoning and marking...')
      const reasoning = await analyzeReasoning({
        question_text: answerOcr.question_text,
        student_steps: answerOcr.student_steps,
        marking_scheme: markingScheme,
        total_marks: totalMarks,
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
      <h2 className="text-2xl font-bold text-gray-800 mb-2">Evaluate Answer Sheet</h2>
      <p className="text-gray-500 text-sm mb-6">
        Upload both images — the system runs OCR on each and evaluates the student's answer.
      </p>

      {/* ── Answer sheet ── */}
      <p className="text-sm font-semibold text-gray-700 mb-2">
        Student Answer Sheet <span className="text-red-500">*</span>
      </p>
      <div
        {...getAnswerRootProps()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition ${
          isAnswerDragActive
            ? 'border-blue-500 bg-blue-50'
            : answerFile
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 bg-white'
        }`}
      >
        <input {...getAnswerInputProps()} />
        {answerFile ? (
          <div>
            <p className="text-blue-600 font-medium">{answerFile.name}</p>
            <p className="text-gray-400 text-sm mt-1">Click to replace</p>
          </div>
        ) : (
          <div>
            <p className="text-gray-400">Drag & drop answer sheet image, or click to select</p>
            <p className="text-gray-300 text-sm mt-1">PNG, JPG supported</p>
          </div>
        )}
      </div>

      {answerFile && (
        <div className="mt-3">
          <img
            src={URL.createObjectURL(answerFile)}
            alt="Answer sheet preview"
            className="rounded-lg max-h-40 object-contain border border-gray-200"
          />
        </div>
      )}

      {/* ── Marking scheme ── */}
      <p className="text-sm font-semibold text-gray-700 mt-6 mb-2">
        Marking Scheme <span className="text-red-500">*</span>
      </p>
      <div
        {...getSchemeRootProps()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition ${
          isSchemeDragActive
            ? 'border-green-500 bg-green-50'
            : schemeFile
            ? 'border-green-400 bg-green-50'
            : 'border-gray-300 hover:border-green-400 bg-white'
        }`}
      >
        <input {...getSchemeInputProps()} />
        {schemeFile ? (
          <div>
            <p className="text-green-600 font-medium">{schemeFile.name}</p>
            <p className="text-gray-400 text-sm mt-1">Click to replace</p>
          </div>
        ) : (
          <div>
            <p className="text-gray-400">Drag & drop marking scheme image, or click to select</p>
            <p className="text-gray-300 text-sm mt-1">PNG, JPG supported</p>
          </div>
        )}
      </div>

      {schemeFile && (
        <div className="mt-3">
          <img
            src={URL.createObjectURL(schemeFile)}
            alt="Marking scheme preview"
            className="rounded-lg max-h-40 object-contain border border-gray-200"
          />
        </div>
      )}

      {loading && (
        <div className="mt-5 text-sm text-blue-600 font-medium animate-pulse">{step}</div>
      )}

      {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}

      <button
        onClick={handleSubmit}
        disabled={!answerFile || !schemeFile || loading}
        className="mt-6 w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
      >
        {loading ? 'Evaluating...' : 'Evaluate Answer'}
      </button>
    </div>
  )
}
