export interface Model {
  id: string
  name: string
  category: string
  params: string
}

export interface AnalysisResult {
  currentMemoryUsage: number
  optimalMemoryUsage: number
  wastePercentage: number
  speedup: number
  costPerRun: number
  annualSavings: number
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
