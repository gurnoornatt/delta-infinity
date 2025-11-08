"use client"

interface MemoryGaugeProps {
  label: string
  value: number
  maxValue: number
  color: "green" | "red" | "blue" | "purple"
  unit: string
}

export default function MemoryGauge({ label, value, maxValue, color, unit }: MemoryGaugeProps) {
  const percentage = (value / maxValue) * 100
  const circumference = 2 * Math.PI * 45

  const getColorClasses = () => {
    switch (color) {
      case "green":
        return "text-accent"
      case "red":
        return "text-red-400"
      case "blue":
        return "text-blue-400"
      case "purple":
        return "text-purple-400"
    }
  }

  const getGradientColor = () => {
    switch (color) {
      case "green":
        return "#00ff88"
      case "red":
        return "#f87171"
      case "blue":
        return "#60a5fa"
      case "purple":
        return "#c084fc"
    }
  }

  const dashOffset = circumference - (percentage / 100) * circumference

  return (
    <div className="flex flex-col items-center justify-center p-4">
      <div className="relative w-32 h-32">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
          {/* Background circle */}
          <circle cx="60" cy="60" r="45" fill="none" stroke="rgba(255, 255, 255, 0.1)" strokeWidth="8" />
          {/* Progress circle */}
          <circle
            cx="60"
            cy="60"
            r="45"
            fill="none"
            stroke={getGradientColor()}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            className="transition-all duration-700"
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <p className={`text-2xl ${getColorClasses()}`}>{percentage.toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground">
            {value.toFixed(1)}
            {unit}
          </p>
        </div>
      </div>
      <p className="text-sm text-muted-foreground text-center mt-3">{label}</p>
    </div>
  )
}
