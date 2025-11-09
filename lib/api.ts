/**
 * API Service Module for MemoryMark Backend Integration
 *
 * Provides typed API functions to interact with the Flask backend.
 * Handles error handling, timeouts, and model ID mapping.
 *
 * Documentation:
 * - Next.js Environment Variables: https://nextjs.org/docs/14/app/building-your-application/configuring/environment-variables
 * - Fetch Timeouts: https://dmitripavlutin.com/timeout-fetch-request/
 * - Error Handling: https://jessewarden.com/2025/02/error-handling-for-fetch-in-typescript.html
 */

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001'

// Timeout constants (in milliseconds)
const DEFAULT_TIMEOUT = 10000 // 10 seconds for health/models
const ANALYSIS_TIMEOUT = 90000 // 90 seconds for GPU analysis

/**
 * Model ID mapping: Frontend ID → Backend ID
 * Frontend uses descriptive IDs, backend uses short codes
 */
const MODEL_ID_MAP: Record<string, string> = {
  'bert-base': 'bert',
  'gpt-2': 'gpt2',
  'resnet-50': 'resnet'
}

/**
 * Backend API response types (matches Flask responses)
 */
interface BackendHealthResponse {
  status: 'healthy' | 'unhealthy'
  gpu_available: boolean
  gpu_name: string | null
  gpu_memory_total_gb: number
  device: string
  timestamp: string
  error?: string
}

interface BackendModel {
  id: string
  name: string
  description: string
  type: string
  huggingface_id: string
}

interface BackendModelsResponse {
  models: BackendModel[]
}

interface BatchResult {
  batch_size: number
  memory_mb: number
  memory_gb: number
  success: boolean
  error: string | null
}

interface BackendAnalysisData {
  model_name: string
  device: string
  gpu_total_gb: number
  optimal_batch_size: number
  optimal_memory_gb: number
  current_batch_size: number
  current_memory_gb: number
  waste_gb: number
  waste_percent: number
  speedup: number
  cost_savings_per_run: number
  cost_savings_annual: number
  results: BatchResult[]
}

interface BackendAnalysisResponse {
  status: 'success' | 'error'
  data?: BackendAnalysisData
  error?: string
}

/**
 * Custom error class for API errors
 */
export class APIError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: unknown
  ) {
    super(message)
    this.name = 'APIError'
  }
}

/**
 * Fetch with timeout using AbortSignal.timeout()
 * Modern approach that automatically handles timeout errors
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT
): Promise<Response> {
  try {
    const response = await fetch(url, {
      ...options,
      signal: AbortSignal.timeout(timeoutMs)
    })
    return response
  } catch (error) {
    // Check if it's a timeout error
    if (error instanceof Error && error.name === 'TimeoutError') {
      throw new APIError(
        `Request timeout after ${timeoutMs / 1000} seconds`,
        408,
        { originalError: error }
      )
    }
    // Check if it's an abort error (network failure)
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIError(
        'Request was aborted',
        0,
        { originalError: error }
      )
    }
    // Re-throw other errors
    throw error
  }
}

/**
 * Handle HTTP response errors
 * Fetch doesn't reject on HTTP error status (404, 500, etc), so we need to check manually
 */
async function handleResponse<T>(response: Response): Promise<T> {
  // Check if response is OK (status 200-299)
  if (!response.ok) {
    // Try to parse error message from response body
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`
    try {
      const errorData = await response.json()
      if (errorData.error) {
        errorMessage = errorData.error
      }
    } catch {
      // If JSON parsing fails, use default error message
    }

    throw new APIError(errorMessage, response.status)
  }

  // Parse JSON response
  try {
    const data = await response.json()
    return data as T
  } catch (error) {
    throw new APIError(
      'Failed to parse response JSON',
      response.status,
      { originalError: error }
    )
  }
}

/**
 * GET /health - Check backend health and GPU status
 *
 * @returns Health status with GPU information
 * @throws APIError if request fails
 */
export async function getHealth(): Promise<BackendHealthResponse> {
  const url = `${API_BASE_URL}/health`

  try {
    const response = await fetchWithTimeout(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    }, DEFAULT_TIMEOUT)

    return await handleResponse<BackendHealthResponse>(response)
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    throw new APIError(
      'Failed to fetch health status',
      0,
      { originalError: error }
    )
  }
}

/**
 * GET /models - Get list of available models
 *
 * @returns List of models with metadata
 * @throws APIError if request fails
 */
export async function getModels(): Promise<BackendModelsResponse> {
  const url = `${API_BASE_URL}/models`

  try {
    const response = await fetchWithTimeout(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    }, DEFAULT_TIMEOUT)

    return await handleResponse<BackendModelsResponse>(response)
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    throw new APIError(
      'Failed to fetch models list',
      0,
      { originalError: error }
    )
  }
}

/**
 * POST /analyze - Run memory analysis on a model
 *
 * Maps frontend model IDs to backend IDs:
 * - bert-base → bert
 * - gpt-2 → gpt2
 * - resnet-50 → resnet
 *
 * @param modelId Frontend model ID (bert-base, gpt-2, resnet-50)
 * @returns Analysis results with memory and cost metrics
 * @throws APIError if request fails or model ID is invalid
 */
export async function analyzeModel(modelId: string): Promise<BackendAnalysisData> {
  // Map frontend ID to backend ID
  const backendModelId = MODEL_ID_MAP[modelId]

  if (!backendModelId) {
    throw new APIError(
      `Invalid model ID: ${modelId}. Must be one of: ${Object.keys(MODEL_ID_MAP).join(', ')}`,
      400
    )
  }

  const url = `${API_BASE_URL}/analyze`

  try {
    const response = await fetchWithTimeout(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_name: backendModelId
      })
    }, ANALYSIS_TIMEOUT)

    const data = await handleResponse<BackendAnalysisResponse>(response)

    // Check if response indicates error
    if (data.status === 'error') {
      throw new APIError(
        data.error || 'Analysis failed',
        500
      )
    }

    // Ensure data field exists
    if (!data.data) {
      throw new APIError(
        'Invalid response: missing data field',
        500
      )
    }

    return data.data
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    throw new APIError(
      'Failed to run model analysis',
      0,
      { originalError: error }
    )
  }
}

/**
 * Check if API is reachable
 * Useful for connection testing before running analysis
 *
 * @returns true if API is reachable and healthy
 */
export async function checkAPIConnection(): Promise<boolean> {
  try {
    const health = await getHealth()
    return health.status === 'healthy'
  } catch {
    return false
  }
}

/**
 * Get API base URL (for debugging/logging)
 */
export function getAPIBaseURL(): string {
  return API_BASE_URL
}

/**
 * Map backend response to frontend AnalysisResult interface
 * Converts snake_case fields to camelCase
 *
 * @param backendData Backend analysis response from Flask
 * @returns Frontend-compatible AnalysisResult
 */
export function mapBackendToFrontend(backendData: BackendAnalysisData) {
  return {
    optimalBatchSize: backendData.optimal_batch_size,
    currentBatchSize: backendData.current_batch_size,
    optimalMemoryUsage: backendData.optimal_memory_gb,
    currentMemoryUsage: backendData.current_memory_gb,
    wasteGb: backendData.waste_gb,
    gpuMemoryTotal: backendData.gpu_total_gb,
    wastePercentage: backendData.waste_percent,
    speedup: backendData.speedup,
    costPerRun: backendData.cost_savings_per_run,
    annualSavings: backendData.cost_savings_annual,
    results: backendData.results.map(r => ({
      batchSize: r.batch_size,
      memoryMb: r.memory_mb,
      memoryGb: r.memory_gb,
      success: r.success,
      error: r.error
    }))
  }
}
