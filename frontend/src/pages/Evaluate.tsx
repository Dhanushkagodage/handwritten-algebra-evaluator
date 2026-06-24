import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { processImage, analyzeReasoning, generateFeedback } from '../lib/api'
import { useEvaluationStore } from '../store/useEvaluationStore'

const UploadIcon = () => (
  <svg className="w-8 h-8 text-gray-300 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
  </svg>
)

const PIPELINE_STEPS = [
  { key: 'ocr', label: 'OCR Extraction' },
  { key: 'reasoning', label: 'Reasoning & Marking' },
  { key: 'feedback', label: 'Generating Feedback' },
]

export default function Evaluate() {
  const [answerFile, setAnswerFile] = useState<File | null>(null)
  const [schemeFile, setSchemeFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeStep, setActiveStep] = useState<string>('')
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
      setActiveStep('ocr')
      const [answerOcr, schemeOcr] = await Promise.all([
        processImage(answerFile),
        processImage(schemeFile),
      ])
      setOcrResult(answerOcr)

      const parseSchemeStep = (s: any) => {
        const expr: string = s.expression || ''
        const m = expr.match(/^(.*?)\s+(\d+)\s+marks?$/i)
        if (m) return { expression: m[1].trim(), marks: parseInt(m[2]) }
        return { expression: expr, marks: 1 }
      }
      const markingScheme = schemeOcr.student_steps.map((s: any) => {
        const { expression, marks } = parseSchemeStep(s)
        return { step_no: s.step_number, description: expression, marks }
      })
      const totalMarks = markingScheme.reduce((sum: number, m: any) => sum + m.marks, 0) || 5

      setActiveStep('reasoning')
      const reasoning = await analyzeReasoning({
        question_text: answerOcr.question_text,
        student_steps: answerOcr.student_steps,
        marking_scheme: markingScheme,
        total_marks: totalMarks,
      })
      setReasoningResult(reasoning)

      setActiveStep('feedback')
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
      setActiveStep('')
    }
  }

  const activeIdx = PIPELINE_STEPS.findIndex((s) => s.key === activeStep)

  return (
    <div className="max-w-3xl mx-auto py-4">
      <div className="mb-8">
        <h2 className="text-xl font-bold text-gray-900 mb-1">Evaluate Answer Sheet</h2>
        <p className="text-gray-400 text-sm">Upload both images to begin the evaluation pipeline.</p>
      </div>

      {/* Upload grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Answer sheet */}
        <div>
          <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">
            Student Answer Sheet <span className="text-red-400 normal-case font-normal">*</span>
          </p>
          <div
            {...getAnswerRootProps()}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all min-h-[160px] flex flex-col items-center justify-center ${
              isAnswerDragActive
                ? 'border-blue-400 bg-blue-50'
                : answerFile
                ? 'border-blue-300 bg-blue-50/40'
                : 'border-gray-200 hover:border-gray-300 bg-white hover:bg-gray-50/60'
            }`}
          >
            <input {...getAnswerInputProps()} />
            {answerFile ? (
              <>
                <img
                  src={URL.createObjectURL(answerFile)}
                  alt="Answer sheet preview"
                  className="rounded-lg max-h-28 object-contain mb-2"
                />
                <p className="text-xs text-blue-600 font-medium truncate max-w-full px-2">{answerFile.name}</p>
                <p className="text-xs text-gray-400 mt-0.5">Click to replace</p>
              </>
            ) : (
              <>
                <UploadIcon />
                <p className="text-xs text-gray-400 mt-2">Drag & drop or click to select</p>
                <p className="text-xs text-gray-300 mt-0.5">PNG, JPG</p>
              </>
            )}
          </div>
        </div>

        {/* Marking scheme */}
        <div>
          <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">
            Marking Scheme <span className="text-red-400 normal-case font-normal">*</span>
          </p>
          <div
            {...getSchemeRootProps()}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all min-h-[160px] flex flex-col items-center justify-center ${
              isSchemeDragActive
                ? 'border-emerald-400 bg-emerald-50'
                : schemeFile
                ? 'border-emerald-300 bg-emerald-50/40'
                : 'border-gray-200 hover:border-gray-300 bg-white hover:bg-gray-50/60'
            }`}
          >
            <input {...getSchemeInputProps()} />
            {schemeFile ? (
              <>
                <img
                  src={URL.createObjectURL(schemeFile)}
                  alt="Marking scheme preview"
                  className="rounded-lg max-h-28 object-contain mb-2"
                />
                <p className="text-xs text-emerald-600 font-medium truncate max-w-full px-2">{schemeFile.name}</p>
                <p className="text-xs text-gray-400 mt-0.5">Click to replace</p>
              </>
            ) : (
              <>
                <UploadIcon />
                <p className="text-xs text-gray-400 mt-2">Drag & drop or click to select</p>
                <p className="text-xs text-gray-300 mt-0.5">PNG, JPG</p>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Pipeline progress */}
      {loading && (
        <div className="bg-white rounded-xl border border-gray-100 px-5 py-4 mb-4">
          <div className="flex items-center gap-2">
            {PIPELINE_STEPS.map((s, i) => {
              const done = i < activeIdx
              const active = s.key === activeStep
              return (
                <div key={s.key} className="flex items-center gap-2">
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-xs flex-shrink-0 font-semibold ${
                      done
                        ? 'bg-green-100 text-green-600'
                        : active
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-400'
                    }`}
                  >
                    {done ? '✓' : i + 1}
                  </div>
                  <span
                    className={`text-xs font-medium ${
                      done ? 'text-green-600' : active ? 'text-blue-600' : 'text-gray-400'
                    }`}
                  >
                    {s.label}
                  </span>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <div className="w-5 h-px bg-gray-200 mx-1" />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-100 rounded-lg px-4 py-3 mb-4">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={!answerFile || !schemeFile || loading}
        className="w-full bg-blue-600 text-white py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
      >
        {loading ? 'Evaluating…' : 'Evaluate Answer Sheet'}
      </button>
    </div>
  )
}
