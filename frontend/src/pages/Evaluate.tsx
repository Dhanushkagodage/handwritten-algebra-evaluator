import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useDropzone } from 'react-dropzone'
import ErrorPanel from '../components/ErrorPanel'
import PipelineProgress from '../components/PipelineProgress'
import ResultsPanel from '../components/results/ResultsPanel'
import { useEvaluation } from '../hooks/useEvaluation'
import { useEvaluationStore } from '../store/useEvaluationStore'

const UploadIcon = () => (
  <svg className="w-8 h-8 text-gray-300 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
  </svg>
)

const CloseIcon = ({ className = 'w-3.5 h-3.5' }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
)

/** Tailwind can't build class names at runtime, so accents are full literals. */
const ACCENTS = {
  blue: {
    dragging: 'border-blue-400 bg-blue-50',
    filled: 'border-blue-300 bg-blue-50/40',
    filename: 'text-blue-600',
  },
  emerald: {
    dragging: 'border-emerald-400 bg-emerald-50',
    filled: 'border-emerald-300 bg-emerald-50/40',
    filename: 'text-emerald-600',
  },
} as const

/**
 * Object URLs leak unless they are revoked, and re-created on every render.
 *
 * The URL is created inside the effect rather than in a useMemo so that create
 * and revoke are the same statement. A memoised URL revoked by a cleanup can
 * never be recreated — the memo won't re-run, since `file` hasn't changed — and
 * StrictMode does exactly that on every mount (setup → cleanup → setup), so any
 * remount of this component would leave a live <img> pointing at a dead blob.
 */
function usePreviewUrl(file: File | null): string | null {
  const [entry, setEntry] = useState<{ file: File; url: string } | null>(null)

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    // Allocating the blob URL is external-resource sync, the carve-out the rule
    // documents: it has to pair with the revoke in this cleanup, and the render
    // pass cannot produce it without leaking one URL per render.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setEntry({ file, url })
    return () => URL.revokeObjectURL(url)
  }, [file])

  // An entry for any other file is stale — either the effect for a just-selected
  // file hasn't run yet, or the file was cleared and the cleanup already revoked
  // this URL. Both read as "no preview", without a second setState round-trip.
  return entry?.file === file ? entry.url : null
}

interface UploadBoxProps {
  label: string
  accent: keyof typeof ACCENTS
  file: File | null
  onFile: (file: File | null) => void
  /** Shorter box for the narrow left column of the side-by-side results view. */
  compact?: boolean
}

function UploadBox({ label, accent, file, onFile, compact = false }: UploadBoxProps) {
  const preview = usePreviewUrl(file)
  const tone = ACCENTS[accent]
  const [zoomed, setZoomed] = useState(false)

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/*': [] },
    maxFiles: 1,
    onDrop: (files) => onFile(files[0]),
  })

  // Removing the file while zoomed would leave an overlay with a revoked src.
  useEffect(() => {
    if (!file) setZoomed(false)
  }, [file])

  useEffect(() => {
    if (!zoomed) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setZoomed(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoomed])

  return (
    <div>
      <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">
        {label} <span className="text-red-400 normal-case font-normal">*</span>
      </p>
      <div
        {...getRootProps()}
        className={`relative ${compact ? 'h-[230px]' : 'h-[400px]'} border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all flex flex-col items-center justify-center ${
          isDragActive
            ? tone.dragging
            : file
              ? tone.filled
              : 'border-gray-200 hover:border-gray-300 bg-white hover:bg-gray-50/60'
        }`}
      >
        <input {...getInputProps()} />
        {file && preview ? (
          <>
            {/* The image is capped by max-height/max-width with auto dimensions
                rather than stretched with object-contain, so its element box is
                exactly the picture — no letterbox gap for the remove button to
                float in. The wrapper then shrink-wraps it in both axes. */}
            <div className="flex-1 min-h-0 w-full flex items-center justify-center mb-2">
              <div className="relative min-w-0 max-w-full">
                {/* Without stopPropagation this would hit the dropzone root and
                    open the file picker instead of the lightbox. */}
                <img
                  src={preview}
                  alt={`${label} preview`}
                  onClick={(e) => {
                    e.stopPropagation()
                    setZoomed(true)
                  }}
                  className={`block w-auto h-auto ${compact ? 'max-h-[140px]' : 'max-h-[310px]'} max-w-full rounded-lg cursor-zoom-in`}
                />
                {/* Clicks bubble into the dropzone root, which would re-open the
                    file picker instead of clearing the selection. */}
                <button
                  type="button"
                  aria-label={`Remove ${label}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    onFile(null)
                  }}
                  className="absolute top-1 right-1 w-6 h-6 rounded-full bg-white/90 border border-gray-200 text-gray-400 flex items-center justify-center shadow-sm hover:bg-red-50 hover:border-red-200 hover:text-red-500 transition"
                >
                  <CloseIcon />
                </button>
              </div>
            </div>
            <p className={`shrink-0 text-xs font-medium truncate max-w-full px-2 ${tone.filename}`}>{file.name}</p>
            <p className="shrink-0 text-xs text-gray-400 mt-0.5">
              Click the image to enlarge · click here to replace
            </p>
          </>
        ) : (
          <>
            <UploadIcon />
            <p className="text-xs text-gray-400 mt-2">Drag & drop or click to select</p>
            <p className="text-xs text-gray-300 mt-0.5">PNG, JPG</p>
          </>
        )}
      </div>

      {/* Portalled to <body>: rendered inside the dropzone, every click in the
          overlay would bubble back into it and re-open the file picker. */}
      {zoomed &&
        preview &&
        createPortal(
          <div
            role="dialog"
            aria-modal="true"
            aria-label={`${label} preview`}
            onClick={() => setZoomed(false)}
            className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8 cursor-zoom-out"
          >
            <img
              src={preview}
              alt={`${label} enlarged preview`}
              onClick={(e) => e.stopPropagation()}
              className="max-h-full max-w-full rounded-lg object-contain shadow-2xl cursor-default"
            />
            <button
              type="button"
              aria-label="Close preview"
              onClick={() => setZoomed(false)}
              className="absolute top-5 right-5 w-9 h-9 rounded-full bg-white/90 text-gray-600 flex items-center justify-center hover:bg-white transition"
            >
              <CloseIcon className="w-5 h-5" />
            </button>
          </div>,
          document.body,
        )}
    </div>
  )
}

export default function Evaluate() {
  const [answerFile, setAnswerFile] = useState<File | null>(null)
  const [schemeFile, setSchemeFile] = useState<File | null>(null)
  const resultRef = useRef<HTMLDivElement | null>(null)

  // The result lives in the store, so the uploads stay mounted alongside it
  // instead of the page being replaced by /results.
  const result = useEvaluationStore((state) => state.result)
  const clearResult = useEvaluationStore((state) => state.reset)

  // The hook already writes the result to the store; nothing else to do here,
  // but the callback must be stable or the hook's effect re-fires every render.
  const onSuccess = useCallback(() => {
    // Stacked layout (narrow screens) puts the report below the fold.
    if (window.innerWidth < 1024) {
      window.requestAnimationFrame(() =>
        resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      )
    }
  }, [])
  const evaluation = useEvaluation(onSuccess)

  const split = result !== null

  // One call. The gateway extracts the answer AND the marking scheme (via the
  // purpose-built /extract-marking-scheme endpoint), runs the reasoning agents,
  // and generates feedback — all server-side.
  const handleSubmit = () => {
    if (!answerFile || !schemeFile) return
    evaluation.submit({
      answerImages: [answerFile],
      schemeImage: schemeFile,
    })
  }

  const uploadPanel = (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 mb-1">Evaluate Answer Sheet</h2>
        <p className="text-gray-400 text-sm">
          {split
            ? 'Swap either image and evaluate again — the report updates on the right.'
            : 'Upload both images to begin the evaluation pipeline.'}
        </p>
      </div>

      {/* Side by side while there is room; stacked once the results panel
          takes the other half of the row. */}
      <div className={`grid gap-4 mb-6 ${split ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <UploadBox
          label="Student Answer Sheet"
          accent="blue"
          file={answerFile}
          onFile={setAnswerFile}
          compact={split}
        />
        <UploadBox
          label="Marking Scheme"
          accent="emerald"
          file={schemeFile}
          onFile={setSchemeFile}
          compact={split}
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
        {evaluation.loading ? 'Evaluating…' : split ? 'Evaluate Again' : 'Evaluate Answer Sheet'}
      </button>

      {!evaluation.loading && (
        <p className="text-xs text-gray-300 text-center mt-3 leading-relaxed">
          A full evaluation takes about a minute. The first run of the day can take longer
          while the feedback model wakes up.
        </p>
      )}
    </div>
  )

  // Both layouts render the same element tree — only the class names differ, and
  // the results column is appended. Branching into two separate `return`s instead
  // would change the shape of the tree at the moment a result arrives, so React
  // would reconcile the wrapper against the panel it wraps, unmount both upload
  // boxes and mount fresh ones. That silently discards their state (the selected
  // file survives only because it lives up here) for no visual gain.
  return (
    <div
      className={
        split
          ? 'max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[minmax(0,400px)_minmax(0,1fr)] gap-8 items-start'
          : 'max-w-3xl mx-auto'
      }
    >
      {/* Pinned while the report scrolls past it. `top` clears the sticky navbar
          (h-14) plus the main element's py-8; the max-height/overflow pair keeps
          the panel usable on short viewports, where it would otherwise have its
          bottom — including the evaluate button — cut off with no way to reach it. */}
      <div
        className={
          split ? 'lg:sticky lg:top-[4.5rem] lg:max-h-[calc(100vh-5.5rem)] lg:overflow-y-auto lg:pr-1' : ''
        }
      >
        {uploadPanel}
      </div>
      {split && (
        <div ref={resultRef} className="min-w-0 scroll-mt-20">
          <ResultsPanel
            result={result}
            footer={
              <div className="flex gap-3 items-center pb-4">
                <button
                  type="button"
                  onClick={clearResult}
                  className="bg-gray-100 text-gray-600 px-5 py-2 rounded-lg text-sm font-medium hover:bg-gray-200 transition"
                >
                  Clear Results
                </button>
                {result.timings_ms.total != null && (
                  <span className="text-xs text-gray-300 ml-auto">
                    Evaluated in {(result.timings_ms.total / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
            }
          />
        </div>
      )}
    </div>
  )
}
