"use client"

export default function LoadingState() {
  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/10 rounded-lg blur opacity-75"></div>
        <div className="relative bg-card border border-border/50 rounded-lg p-8 backdrop-blur-md">
          <div className="space-y-6">
            {/* Animated progress bar */}
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Analysis Progress</p>
              <div className="w-full bg-background/50 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-accent via-accent/70 to-accent rounded-full animate-pulse"
                  style={{
                    animation: "loading 2s ease-in-out infinite",
                    backgroundSize: "200% 100%",
                  }}
                ></div>
              </div>
            </div>

            {/* Loading text */}
            <div className="flex items-center gap-2 justify-center">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0s" }}></div>
                <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
              </div>
              <p className="text-accent font-medium">Analyzing GPU Memory...</p>
            </div>

            {/* Substeps */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-4 border-t border-border/30">
              {["Reading Kernel", "Profiling Memory", "Optimizing Config"].map((step, i) => (
                <div key={step} className="text-center text-sm text-muted-foreground">
                  <div className="flex items-center justify-center mb-2">
                    <div className="w-4 h-4 rounded-full border border-accent/50 flex items-center justify-center">
                      {i === 0 && <div className="w-2 h-2 bg-accent rounded-full animate-pulse"></div>}
                    </div>
                  </div>
                  {step}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes loading {
          0% {
            background-position: 200% 0;
            opacity: 0.5;
          }
          50% {
            opacity: 1;
          }
          100% {
            background-position: -200% 0;
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  )
}
