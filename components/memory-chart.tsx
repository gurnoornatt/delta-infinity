"use client"

interface MemoryChartProps {
  title: string
  data: { time: string; value: number }[]
  color: "green" | "red" | "orange"
}

export default function MemoryChart({ title, data, color }: MemoryChartProps) {
  const maxValue = Math.max(...data.map((d) => d.value))
  const minValue = Math.min(...data.map((d) => d.value))
  const range = maxValue - minValue || 1

  const getColorClass = () => {
    switch (color) {
      case "green":
        return "stroke-accent"
      case "red":
        return "stroke-red-400"
      case "orange":
        return "stroke-orange-400"
    }
  }

  const points = data
    .map((d, i) => {
      const x = (i / (data.length - 1)) * 280
      const y = 100 - ((d.value - minValue) / range) * 80
      return `${x},${y}`
    })
    .join(" ")

  return (
    <div className="relative group">
      <div className="absolute inset-0 bg-gradient-to-b from-card/20 to-card/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-500"></div>
      <div className="relative bg-card border border-border/50 rounded-lg p-6 backdrop-blur-md">
        <h3 className="text-sm text-foreground mb-4">{title}</h3>
        <div className="relative h-32">
          <svg viewBox="0 0 300 120" className="w-full h-full" preserveAspectRatio="none">
            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map((y) => (
              <line key={y} x1="0" y1={y} x2="300" y2={y} stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" />
            ))}

            {/* Data line */}
            <polyline
              points={points}
              fill="none"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={getColorClass()}
            />

            {/* Gradient fill */}
            <defs>
              <linearGradient id={`grad-${title}`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop
                  offset="0%"
                  stopColor={color === "green" ? "#00ff88" : color === "red" ? "#f87171" : "#fb923c"}
                  stopOpacity="0.2"
                />
                <stop
                  offset="100%"
                  stopColor={color === "green" ? "#00ff88" : color === "red" ? "#f87171" : "#fb923c"}
                  stopOpacity="0"
                />
              </linearGradient>
            </defs>
            <polygon points={`0,100 ${points} 300,100`} fill={`url(#grad-${title})`} />
          </svg>
        </div>
        <div className="flex justify-between mt-3 text-xs text-muted-foreground">
          <span>{data[0]?.time}</span>
          <span>{data[data.length - 1]?.time}</span>
        </div>
      </div>
    </div>
  )
}
