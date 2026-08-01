import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import ErrorPanel from '../components/ErrorPanel'
import PipelineProgress from '../components/PipelineProgress'
import { useEvaluation } from '../hooks/useEvaluation'

const UploadIcon = () => (
  <svg className="w-8 h-8 text-gray-300 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
  </svg>
)

/** Object URLs leak unless they are revoked, and re-created on every render. */
function usePreviewUrl(file: File | null): string | null {
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url)
    }
  }, [url])
  return url
}

export default function Evaluate() {
  const [answerFile, setAnswerFile] = useState<File | null>(null)
  const [schemeFile, setSchemeFile] = useState<File | null>(null)
  const [questionText, setQuestionText] = useState('')
  const navigate = useNavigate()

  const onSuccess = useCallback(() => navigate('/results'), [navigate])
  const evaluation = useEvaluation(onSuccess)

  const answerPreview = usePreviewUrl(answerFile)
  const schemePreview = usePreviewUrl(schemeFile)

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

  // One call. The gateway extracts the answer AND the marking scheme (via the
  // purpose-built /extract-marking-scheme endpoint), runs the reasoning agents,
  // and generates feedback — all server-side.
  const handleSubmit = () => {
    if (!answerFile || !schemeFile) return
    evaluation.submit({
      answerImages: [answerFile],
      schemeImage: schemeFile,
      questionText,
    })
  }

  return (
    <div className="max-w-3xl mx-auto py-4">
      <div className="mb-8">
        <h2 className="text-xl font-bold text-gray-900 mb-1">Evaluate Answer Sheet</h2>
        <p className="text-gray-400 text-sm">Upload both images to begin the evaluation pipeline.</p>
      </div>

      {/* Upload grid */}
      <div className="grid grid-cols-2 gap-4 mb-4">
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
            {answerFile && answerPreview ? (
              <>
                <img
                  src={answerPreview}
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
            {schemeFile && schemePreview ? (
              <>
                <img
                  src={schemePreview}
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

      {/* Both OCR endpoints use this to anchor extraction, so it materially
          improves accuracy when the question isn't legible on the sheet. */}
      <div className="mb-6">
        <label htmlFor="question-text" className="block text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">
          Question Text <span className="text-gray-300 normal-case font-normal">(optional, improves accuracy)</span>
        </label>
        <input
          id="question-text"
          type="text"
          value={questionText}
          onChange={(e) => setQuestionText(e.target.value)}
          placeholder="e.g. Solve x² − 5x + 6 = 0"
          disabled={evaluation.loading}
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm placeholder:text-gray-300 focus:outline-none focus:border-blue-300 disabled:bg-gray-50"
        />
      </div>

      {evaluation.loading && (
        <PipelineProgress
          stage={evaluation.stage}
          stageMessage={evaluation.stageMessage}
          uploadPercent={evaluation.uploadPercent}
          elapsedSeconds={evaluation.elapsedSeconds}
          onCancel={evaluation.cancel}
        />
      )}

      {evaluation.error && !evaluation.loading && (
        <ErrorPanel error={evaluation.error} onRetry={evaluation.reset} />
      )}

      <button
        onClick={handleSubmit}
        disabled={!answerFile || !schemeFile || evaluation.loading}
        className="w-full bg-blue-600 text-white py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
      >
        {evaluation.loading ? 'Evaluating…' : 'Evaluate Answer Sheet'}
      </button>

      {!evaluation.loading && (
        <p className="text-xs text-gray-300 text-center mt-3 leading-relaxed">
          A full evaluation takes about a minute. The first run of the day can take longer
          while the feedback model wakes up.
        </p>
      )}
    </div>
  )
}
