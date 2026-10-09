'use client'

import { Button } from '@/components/ui/button'
import { toast } from '@/hooks/use-toast'
import { Download } from 'lucide-react'
import { downloadUrl } from '@/lib/download'

/** Run a download and report failure as a toast instead of an unhandled rejection. */
export async function runDownload(label: string, job: () => Promise<void>) {
  try {
    await job()
  } catch (error) {
    console.error('Download failed:', error)
    toast({
      title: 'Download failed',
      description: error instanceof Error ? error.message : `Could not download ${label}`,
      variant: 'destructive'
    })
  }
}

/** A small download button pinned to one image (grid tiles and single views). */
export function ImageDownloadButton({ url, label }: { url: string; label: string }) {
  return (
    <Button
      variant="outline"
      size="icon"
      className="h-8 w-8 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm shadow"
      title={`Download ${label}`}
      aria-label={`Download ${label}`}
      onClick={() => runDownload(label, () => downloadUrl(url))}
    >
      <Download size={14} />
    </Button>
  )
}
