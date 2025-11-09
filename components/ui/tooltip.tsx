"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface TooltipProps {
  children: React.ReactNode
  content: string
  className?: string
}

export function Tooltip({ children, content, className }: TooltipProps) {
  const [isVisible, setIsVisible] = React.useState(false)

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        className="cursor-help"
      >
        {children}
      </div>
      {isVisible && (
        <div
          className={cn(
            "absolute z-50 px-3 py-2 text-sm text-foreground bg-card/95 border border-accent/30 rounded-lg shadow-lg backdrop-blur-md",
            "bottom-full left-1/2 -translate-x-1/2 mb-2 w-64",
            "animate-in fade-in duration-200",
            className
          )}
        >
          {content}
          {/* Tooltip arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
            <div className="border-4 border-transparent border-t-accent/30"></div>
          </div>
        </div>
      )}
    </div>
  )
}
