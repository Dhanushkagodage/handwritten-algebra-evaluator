import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  ApiError,
  cancelEvaluation,
  getEvaluation,
  startEvaluation,
  toApiError,
} from '../lib/api'
import { useEvaluationStore } from '../store/useEvaluationStore'
import type { EvaluationResult, PipelineStage, StartEvaluationInput } from '../types/api'

const POLL_INTERVAL_MS = 1500
const TERMINAL: string[] = ['succeeded', 'failed', 'cancelled']

export interface UseEvaluation {
  submit: (input: StartEvaluationInput) => void
  cancel: () => void
  reset: () => void
  loading: boolean
  /** Server-reported stage — drives the progress tracker. */
  stage: PipelineStage | null
  stageMessage: string | null
  uploadPercent: number
  /** Seconds since submit, so a multi-minute run doesn't look like a hang. */
  elapsedSeconds: number
  warnings: string[]
  error: ApiError | null
}

/**
 * Start an evaluation and poll it to completion.
 *
 * The gateway reports which of the three modules is currently running, so the
 * progress tracker reflects real server state instead of optimistic guesses.
 */
export function useEvaluation(onSuccess: (result: EvaluationResult) => void): UseEvaluation {
  const [jobId, setJobId] = useState<string | null>(null)
  const [uploadPercent, setUploadPercent] = useState(0)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [jobError, setJobError] = useState<ApiError | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const startedAtRef = useRef<number | null>(null)
  const setResult = useEvaluationStore((state) => state.setResult)

  const start = useMutation({
    mutationFn: (input: StartEvaluationInput) => {
      abortRef.current = new AbortController()
      return startEvaluation(input, {
        signal: abortRef.current.signal,
        onUploadProgress: setUploadPercent,
      })
    },
    // Never silently re-upload and re-run a pipeline that takes minutes.
    retry: false,
    onMutate: () => {
      setJobError(null)
      setUploadPercent(0)
      setElapsedSeconds(0)
      startedAtRef.current = Date.now()
    },
    onSuccess: (job) => setJobId(job.job_id),
    onError: (err) => setJobError(toApiError(err)),
  })

  const job = useQuery({
    queryKey: ['evaluation', jobId],
    queryFn: () => getEvaluation(jobId as string),
    enabled: jobId !== null,
    // TanStack Query v5 hands this callback the Query, not the data.
    refetchInterval: (query) =>
      TERMINAL.includes(query.state.data?.status ?? '') ? false : POLL_INTERVAL_MS,
    refetchOnWindowFocus: false,
    staleTime: 0,
    // Survive a momentary blip mid-demo rather than losing a running job.
    retry: 3,
  })

  const status = job.data?.status
  const isRunning = start.isPending || (jobId !== null && !TERMINAL.includes(status ?? ''))

  // Tick the elapsed clock while anything is in flight.
  useEffect(() => {
    if (!isRunning || startedAtRef.current === null) return
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - (startedAtRef.current as number)) / 1000))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [isRunning])

  useEffect(() => {
    if (!job.data) return

    if (job.data.status === 'succeeded' && job.data.result) {
      setResult(job.data.result)
      onSuccess(job.data.result)
      setJobId(null)
      return
    }

    if (job.data.status === 'failed' && job.data.error) {
      const { error } = job.data
      setJobError(
        new ApiError({
          message: error.message,
          kind: 'gateway',
          stage: error.stage,
          code: error.error_code,
          status: error.status_code,
          details: error.details,
        }),
      )
      setJobId(null)
      return
    }

    if (job.data.status === 'cancelled') {
      setJobId(null)
    }
  }, [job.data, onSuccess, setResult])

  const cancel = () => {
    abortRef.current?.abort()
    if (jobId) void cancelEvaluation(jobId)
    setJobId(null)
    start.reset()
  }

  const reset = () => {
    setJobError(null)
    setJobId(null)
    setUploadPercent(0)
    setElapsedSeconds(0)
    start.reset()
  }

  const stage: PipelineStage | null = job.data?.stage ?? (start.isPending ? 'queued' : null)

  return {
    submit: (input) => start.mutate(input),
    cancel,
    reset,
    loading: isRunning,
    stage,
    stageMessage: job.data?.stage_message ?? null,
    uploadPercent,
    elapsedSeconds,
    warnings: job.data?.warnings ?? [],
    error: jobError ?? (job.error ? toApiError(job.error) : null),
  }
}
