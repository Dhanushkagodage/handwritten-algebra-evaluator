import axios from 'axios'

const OCR_URL = 'http://localhost:8001/api/v1'
const REASONING_URL = 'http://localhost:8002/api/v1'
const FEEDBACK_URL = 'http://localhost:8003/api/v1'

export const ocrApi = axios.create({ baseURL: OCR_URL })
export const reasoningApi = axios.create({ baseURL: REASONING_URL })
export const feedbackApi = axios.create({ baseURL: FEEDBACK_URL })

export const processImage = async (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await ocrApi.post('/ocr', formData)
  return data
}

export const analyzeReasoning = async (payload: object) => {
  const { data } = await reasoningApi.post('/analyze', payload)
  return data
}

export const generateFeedback = async (payload: object) => {
  const { data } = await feedbackApi.post('/feedback', payload)
  return data
}
