/**
 * Terminal Simulator - Generates realistic terminal logs for MemoryMark analysis
 *
 * Simulates the actual backend output when running:
 * $ python memorymark.py <model>
 */

export type LogType = 'command' | 'header' | 'info' | 'success' | 'error' | 'divider' | 'result'

export interface TerminalLog {
  text: string
  type: LogType
  delay: number  // milliseconds to wait before showing this log
}

/**
 * Model-specific configuration based on actual GPU test results
 * Data from Lambda Labs A10 24GB testing
 */
const MODEL_CONFIGS = {
  'bert-base': {
    huggingface_id: 'google-bert/bert-base-uncased',
    optimal_batch: 288,
    batch_start: 8,
    batch_increment: 8,
    memory_per_batch: 0.075,  // GB per batch size unit
    base_memory: 0.3,
  },
  'gpt-2': {
    huggingface_id: 'openai-community/gpt2',
    optimal_batch: 152,
    batch_start: 8,
    batch_increment: 8,
    memory_per_batch: 0.14,
    base_memory: 0.5,
  },
  'resnet-50': {
    huggingface_id: 'microsoft/resnet-50',
    optimal_batch: 264,
    batch_start: 8,
    batch_increment: 8,
    memory_per_batch: 0.08,
    base_memory: 0.3,
  }
}

/**
 * Generate realistic terminal logs for a model analysis
 */
export function generateTerminalLogs(modelId: string, modelName: string): TerminalLog[] {
  const config = MODEL_CONFIGS[modelId as keyof typeof MODEL_CONFIGS]
  if (!config) {
    throw new Error(`Unknown model: ${modelId}`)
  }

  const logs: TerminalLog[] = []

  // Command prompt
  logs.push({
    text: `$ python memorymark.py ${modelId.replace('-base', '').replace('-2', '2').replace('-50', '')}`,
    type: 'command',
    delay: 300
  })

  logs.push({ text: '', type: 'info', delay: 100 })

  // Header
  logs.push({
    text: '🔧 MemoryMark - GPU Memory Optimizer',
    type: 'header',
    delay: 200
  })

  logs.push({
    text: '━'.repeat(60),
    type: 'divider',
    delay: 50
  })

  logs.push({ text: '', type: 'info', delay: 50 })

  // GPU and model info
  logs.push({
    text: 'Device: cuda (NVIDIA A10 - 24GB)',
    type: 'info',
    delay: 150
  })

  logs.push({
    text: `Model: ${config.huggingface_id}`,
    type: 'info',
    delay: 150
  })

  logs.push({
    text: '━'.repeat(60),
    type: 'divider',
    delay: 50
  })

  logs.push({ text: '', type: 'info', delay: 100 })

  // Loading model
  logs.push({
    text: `Loading ${modelId}...`,
    type: 'info',
    delay: 800
  })

  logs.push({
    text: '✓ Model loaded successfully',
    type: 'success',
    delay: 1200
  })

  logs.push({ text: '', type: 'info', delay: 100 })

  logs.push({
    text: 'Starting batch size analysis...',
    type: 'info',
    delay: 400
  })

  logs.push({ text: '', type: 'info', delay: 200 })

  // Generate batch size tests
  let currentBatch = config.batch_start
  const sampleInterval = 4  // Sample every 4th batch for reasonable log length

  while (currentBatch <= config.optimal_batch) {
    const memoryGb = (config.base_memory + currentBatch * config.memory_per_batch).toFixed(2)

    logs.push({
      text: `[Testing] batch_size=${currentBatch.toString().padEnd(3)} → ${memoryGb.padStart(5)} GB ✓`,
      type: 'success',
      delay: currentBatch % (sampleInterval * config.batch_increment) === 0 ? 600 : 300
    })

    currentBatch += config.batch_increment
  }

  // OOM error (next batch size fails)
  const oomBatch = currentBatch
  logs.push({
    text: `[Testing] batch_size=${oomBatch.toString().padEnd(3)} → OOM ✗`,
    type: 'error',
    delay: 700
  })

  logs.push({ text: '', type: 'info', delay: 200 })

  // Final results
  logs.push({
    text: '━'.repeat(60),
    type: 'divider',
    delay: 300
  })

  logs.push({
    text: 'Analysis Complete!',
    type: 'header',
    delay: 500
  })

  const optimalMemory = (config.base_memory + config.optimal_batch * config.memory_per_batch).toFixed(1)
  const speedup = (config.optimal_batch / 16).toFixed(1)
  const efficiency = ((parseFloat(optimalMemory) / 24) * 100).toFixed(1)

  logs.push({
    text: `Optimal batch size: ${config.optimal_batch} (${speedup}x faster)`,
    type: 'result',
    delay: 200
  })

  logs.push({
    text: `GPU efficiency: ${efficiency}%`,
    type: 'result',
    delay: 200
  })

  logs.push({
    text: '━'.repeat(60),
    type: 'divider',
    delay: 100
  })

  logs.push({ text: '', type: 'info', delay: 100 })

  return logs
}

/**
 * Get total estimated duration for all logs
 */
export function getTotalDuration(logs: TerminalLog[]): number {
  return logs.reduce((total, log) => total + log.delay, 0)
}
