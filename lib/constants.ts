export interface Model {
  id: string
  name: string
  category: string
  params: string
}

export interface BatchTestResult {
  batchSize: number
  memoryMb: number
  memoryGb: number
  success: boolean
  error: string | null
}

/**
 * Analysis Result Interface
 * Maps backend response from Flask /analyze endpoint
 * Backend fields → Frontend fields (camelCase conversion)
 */
export interface AnalysisResult {
  // Batch size metrics
  optimalBatchSize: number          // optimal_batch_size - recommended batch size
  currentBatchSize: number           // current_batch_size - default batch size (16)

  // Memory metrics (in GB)
  optimalMemoryUsage: number         // optimal_memory_gb - memory at optimal batch
  currentMemoryUsage: number         // current_memory_gb - memory at current batch
  wasteGb: number                    // waste_gb - absolute waste in GB
  gpuMemoryTotal: number             // gpu_total_gb - total GPU memory available

  // Percentage and performance metrics
  wastePercentage: number            // waste_percent - percentage of GPU wasted
  speedup: number                    // speedup - training speedup factor

  // Cost savings metrics
  costPerRun: number                 // cost_savings_per_run - savings per training run
  annualSavings: number              // cost_savings_annual - yearly savings

  // Raw test results array
  results: BatchTestResult[]         // results - all batch size test results
}

export const MODELS: Model[] = [
  {
    id: "bert-base",
    name: "BERT Base",
    category: "NLP",
    params: "110M params",
  },
  {
    id: "gpt-2",
    name: "GPT-2",
    category: "LLM",
    params: "117M params",
  },
  {
    id: "resnet-50",
    name: "ResNet-50",
    category: "Vision",
    params: "25M params",
  },
]
