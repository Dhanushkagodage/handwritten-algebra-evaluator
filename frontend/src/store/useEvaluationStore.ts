import { create } from 'zustand'

interface EvaluationStore {
  ocrResult: any | null
  reasoningResult: any | null
  feedbackResult: any | null
  setOcrResult: (data: any) => void
  setReasoningResult: (data: any) => void
  setFeedbackResult: (data: any) => void
  reset: () => void
}

export const useEvaluationStore = create<EvaluationStore>((set) => ({
  ocrResult: null,
  reasoningResult: null,
  feedbackResult: null,
  setOcrResult: (data) => set({ ocrResult: data }),
  setReasoningResult: (data) => set({ reasoningResult: data }),
  setFeedbackResult: (data) => set({ feedbackResult: data }),
  reset: () => set({ ocrResult: null, reasoningResult: null, feedbackResult: null }),
}))
