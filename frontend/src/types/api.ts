/**
 * TypeScript mirrors of the backend contracts.
 *
 * Each block cites the Python file it mirrors, so the next time a contract
 * drifts there is one place to check. Note that `enum` is not available here:
 * tsconfig.app.json sets `erasableSyntaxOnly`, which makes TS enums a compile
 * error — hence the string-literal unions.
 */

// ── Module 01 — services/ocr-service/src/openai_vision_ocr.py ───────────────

export interface OcrStudentStep {
  step_id: number
  content: string
}

// ── The marking scheme — identical in all three services ────────────────────

export interface SchemeStep {
  step_no: number
  description: string
  expected_expression: string
  marks: number
}

export interface MarkingScheme {
  /** Marks AVAILABLE. Note EvaluationOutput.total_marks means marks EARNED. */
  total_marks: number
  steps: SchemeStep[]
}

// ── Module 02 — services/reasoning-service/app/schemas/output_schema.py ─────

/**
 * Reasoning's four-value step status.
 *
 * Deliberately NOT unified with StepValidity below: pretending these two sets
 * are the same is the entire class of bug the gateway exists to fix. The
 * gateway translates between them; the frontend only ever sees StepValidity.
 */
export type StepStatus = 'correct' | 'incorrect' | 'partially_correct' | 'unclear'

export interface StepAnalysis {
  step_id: number
  /** A BOOLEAN here. StepFeedback.validity is a STRING with the same name. */
  validity: boolean | null
  status: StepStatus | null
  method: string | null
  matched_scheme_step: number | null
  match_score: number | null
  marks_awarded: number
  max_marks: number
  confidence: number | null
}

export interface MethodDetection {
  detected_method: string
  method_is_valid: boolean | null
  alternative_methods_possible: boolean | null
  alternative_methods: string[]
  confidence: number | null
}

export interface EvaluationOutput {
  steps_analysis: StepAnalysis[]
  /** Marks EARNED. */
  total_marks: number
  /** Marks AVAILABLE. */
  max_marks: number
  percentage: number
  summary: string
  method_feedback: string
  missing_steps_feedback: string | null
  method_detection: MethodDetection | null
}

// ── Module 03 — services/feedback-service/app/models/schemas.py ─────────────

export type StepValidity = 'correct' | 'partial' | 'incorrect'

export interface StepFeedback {
  step_number: number
  expression: string
  validity: StepValidity
  marks_awarded: number
  /** The four-component feedback structure — Module 03's core contribution. */
  what_is_correct: string
  what_is_missing: string | null
  why_marks_reduced: string | null
  how_to_improve: string
  /** A combined one-line summary of the four parts above. */
  feedback: string
}

export interface FeedbackResponse {
  final_score: number
  total_marks: number
  step_feedback: StepFeedback[]
  overall_feedback: string
  improvement_suggestions: string[]
}

// ── Gateway — services/gateway/app/schemas/gateway.py ───────────────────────

/**
 * The middle three values match PIPELINE_STEPS in pages/Evaluate.tsx, so the
 * progress tracker is driven straight off the server's reported stage. Typing
 * that array against this union makes a gateway rename a compile error rather
 * than a silently dead progress bar.
 */
export type PipelineStage = 'queued' | 'ocr' | 'reasoning' | 'feedback' | 'done'
export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled'
export type StageStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped'

export interface StageRecord {
  key: PipelineStage
  status: StageStatus
  started_at: string | null
  finished_at: string | null
  duration_ms: number | null
  message: string | null
}

export interface ErrorEnvelope {
  error: boolean
  error_code: string
  message: string
  /** Mirrors `message` — the FastAPI convention the old client relied on. */
  detail: string
  stage: string | null
  status_code: number
  details: Record<string, unknown>
}

export interface QuestionResult {
  question_id: string
  question_text: string
  marking_scheme: MarkingScheme
  student_steps: OcrStudentStep[]
  final_answer: string | null
  reasoning: EvaluationOutput
  feedback: FeedbackResponse
}

export interface EvaluationResult {
  // Flattened primary question — mirrors FeedbackResponse field for field.
  final_score: number
  total_marks: number
  step_feedback: StepFeedback[]
  overall_feedback: string
  improvement_suggestions: string[]

  question_id: string
  question_text: string
  question_count: number
  questions: QuestionResult[]
  /** Anything the gateway had to infer or guess about, in plain English. */
  warnings: string[]
  timings_ms: Record<string, number>
}

export interface JobCreated {
  job_id: string
  status: JobStatus
  stage: PipelineStage
  poll_url: string
  poll_after_ms: number
}

export interface EvaluationJob {
  job_id: string
  status: JobStatus
  stage: PipelineStage
  stage_message: string | null
  stages: StageRecord[]
  progress: number
  created_at: string
  updated_at: string
  elapsed_ms: number
  poll_after_ms: number
  warnings: string[]
  result: EvaluationResult | null
  error: ErrorEnvelope | null
}

export interface StartEvaluationInput {
  answerImages: File[]
  schemeImage: File
  questionText?: string
}
