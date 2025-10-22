'use client'

import { Sparkles } from 'lucide-react'

export function PlaygroundBanner() {
  return (
    <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-white">
      <div className="container mx-auto px-4 py-1">
        <div className="flex items-center justify-center gap-2 text-sm">
          <Sparkles className="h-4 w-4" />
          <span className="font-medium">You're in Playground Mode!</span>
          <span className="text-white/90">
            Your work won't be saved. Sign up to save your work.
          </span>
        </div>
      </div>
    </div>
  )
}

