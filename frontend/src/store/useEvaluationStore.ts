import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import type { EvaluationResult } from '../types/api'

interface EvaluationState {
  /** The gateway returns OCR, reasoning and feedback output in one payload. */
  result: EvaluationResult | null
  submittedAt: string | null
  setResult: (result: EvaluationResult) => void
  reset: () => void
}

/**
 * Persisted to sessionStorage, not localStorage: a result belongs to one
 * sitting, and localStorage would hand a stale answer sheet to the next run.
 * A version bump discards anything written by an older shape rather than
 * letting the results page crash on it.
 */
export const useEvaluationStore = create<EvaluationState>()(
  persist(
    (set) => ({
      result: null,
      submittedAt: null,
      setResult: (result) => set({ result, submittedAt: new Date().toISOString() }),
      reset: () => set({ result: null, submittedAt: null }),
    }),
    {
      name: 'algebra-eval-v1',
      version: 1,
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({ result: state.result, submittedAt: state.submittedAt }),
      migrate: () => ({ result: null, submittedAt: null }) as Partial<EvaluationState>,
    },
  ),
)
