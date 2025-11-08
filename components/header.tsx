"use client"

export default function Header() {
  return (
    <header className="sticky top-0 z-20 border-b border-border/40 bg-background/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center gap-3">
          <div className="text-2xl text-accent">■</div>
          <div>
            <h1 className="text-2xl tracking-tight">Delta-Infinity</h1>
            <p className="text-sm text-muted-foreground">GPU Memory Waste Detector</p>
          </div>
        </div>
      </div>
    </header>
  )
}
