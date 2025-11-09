"use client"

import { useEffect, useRef, useState } from "react"
import { Terminal } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { generateTerminalLogs, type TerminalLog } from "@/lib/terminal-simulator"

interface TerminalViewProps {
  modelName: string
  modelId: string
  onComplete?: () => void
}

export default function TerminalView({ modelName, modelId, onComplete }: TerminalViewProps) {
  const [logs, setLogs] = useState<TerminalLog[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const allLogs = useRef<TerminalLog[]>([])

  // Generate all logs on mount
  useEffect(() => {
    allLogs.current = generateTerminalLogs(modelId, modelName)
    setCurrentIndex(0)
    setLogs([])
  }, [modelId, modelName])

  // Simulate typing effect - add logs progressively
  useEffect(() => {
    if (currentIndex >= allLogs.current.length) {
      // Analysis complete
      if (onComplete) {
        setTimeout(() => onComplete(), 1000)
      }
      return
    }

    const currentLog = allLogs.current[currentIndex]
    const timer = setTimeout(() => {
      setLogs(prev => [...prev, currentLog])
      setCurrentIndex(prev => prev + 1)
    }, currentLog.delay)

    return () => clearTimeout(timer)
  }, [currentIndex, onComplete])

  // Auto-scroll to bottom when new logs appear
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [logs])

  return (
    <div className="space-y-4 animate-in fade-in duration-500">
      {/* Terminal Header */}
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/10 rounded-lg blur opacity-75"></div>
        <div className="relative bg-card/80 backdrop-blur-md border border-accent/30 rounded-lg p-4">
          <div className="flex items-center gap-3 mb-2">
            <Terminal className="w-5 h-5 text-accent" />
            <span className="text-sm font-mono text-accent">ubuntu@lambda-gpu</span>
            <span className="text-sm text-muted-foreground">~/delta-infinity/backend</span>
          </div>
          <div className="text-xs text-muted-foreground">
            <span className="text-accent">Model:</span> {modelName} • <span className="text-accent">GPU:</span> NVIDIA A10 (24GB)
          </div>
        </div>
      </div>

      {/* Terminal Output */}
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-b from-accent/10 to-accent/5 rounded-lg blur opacity-50"></div>
        <div className="relative bg-[#0a0a0a] border border-accent/20 rounded-lg overflow-hidden">
          {/* Terminal Title Bar */}
          <div className="bg-card/50 border-b border-accent/20 px-4 py-2 flex items-center gap-2">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
            </div>
            <span className="text-xs font-mono text-muted-foreground ml-2">memorymark.py</span>
          </div>

          {/* Terminal Content */}
          <ScrollArea ref={scrollAreaRef} className="h-[400px]">
            <div className="p-6 font-mono text-sm space-y-1">
              {logs.map((log, index) => (
                <div
                  key={index}
                  className={`${log.type === 'command' ? 'text-accent' :
                             log.type === 'header' ? 'text-accent font-bold' :
                             log.type === 'success' ? 'text-green-400' :
                             log.type === 'error' ? 'text-red-400' :
                             log.type === 'divider' ? 'text-accent/40' :
                             'text-muted-foreground'
                            } whitespace-pre-wrap`}
                >
                  {log.text}
                </div>
              ))}

              {/* Blinking cursor */}
              {currentIndex < allLogs.current.length && (
                <span className="inline-block w-2 h-4 bg-accent ml-1 animate-pulse"></span>
              )}
            </div>
          </ScrollArea>
        </div>
      </div>

      {/* Progress Indicator */}
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/10 rounded-lg blur opacity-50"></div>
        <div className="relative bg-card/60 backdrop-blur-md border border-accent/20 rounded-lg p-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">
              {currentIndex < allLogs.current.length ? (
                <>Analyzing GPU Memory...</>
              ) : (
                <>Analysis Complete</>
              )}
            </span>
            <span className="text-accent font-mono">
              {currentIndex}/{allLogs.current.length} steps
            </span>
          </div>
          <div className="mt-2 h-1 bg-card rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent to-accent/60 transition-all duration-300"
              style={{ width: `${(currentIndex / allLogs.current.length) * 100}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  )
}
