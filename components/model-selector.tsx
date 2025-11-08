"use client"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { Model } from "@/lib/constants"

interface ModelSelectorProps {
  models: Model[]
  selected: Model
  onSelect: (model: Model) => void
}

export default function ModelSelector({ models, selected, onSelect }: ModelSelectorProps) {
  return (
    <div className="space-y-3">
      <label className="block text-sm">Model Selection</label>
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-accent/10 rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-1000"></div>
        <div className="relative bg-card border border-border/50 rounded-lg p-4 backdrop-blur-md z-20">
          <Select
            value={selected.id}
            onValueChange={(id) => {
              const model = models.find((m) => m.id === id)
              if (model) onSelect(model)
            }}
          >
            <SelectTrigger className="border-border/30 bg-background/50 hover:bg-background/80 transition">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="border-border/50 bg-card z-50">
              {models.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  <div className="flex flex-col">
                    <span>{model.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {model.category} • {model.params}
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  )
}
