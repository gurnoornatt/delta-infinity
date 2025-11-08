"use client"

export default function GPUInfo() {
  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium">GPU Configuration</label>
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-1000"></div>
        <div className="relative bg-card border border-border/50 rounded-lg p-6 backdrop-blur-md">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">GPU Type</p>
              <p className="text-lg text-accent">NVIDIA A10</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Memory</p>
              <p className="text-lg text-accent">24 GB</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
