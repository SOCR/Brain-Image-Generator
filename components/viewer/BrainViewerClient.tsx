'use client'

import { useRef, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { Button } from '@/components/ui/button'
import { Brain, FolderOpen, Layers, Upload, X } from 'lucide-react'
import NiiVueViewer from '@/components/projects/NiiVueViewer'
import { cn } from '@/lib/utils'

/**
 * Standalone Brain Viewer, after the SOCR Brain Viewer (socr.umich.edu/HTML5/BrainViewer):
 * drop or pick a volume (optionally plus a label map) and view it in 3D and in slices.
 *
 * Everything stays in the browser: files are handed to NiiVue as blob: URLs and are never
 * uploaded -- which matters, because people will drop real patient scans here.
 */

const EXTENSIONS = ['.nii', '.nii.gz', '.mgh', '.mgz', '.nrrd']
const ACCEPT = EXTENSIONS.join(',')
// NiiVue's public demo template (MNI152 T1), served with CORS from GitHub Pages.
const EXAMPLE = { url: 'https://niivue.github.io/niivue-demo-images/mni152.nii.gz', name: 'MNI152 template' }

const isVolume = (f: File) => EXTENSIONS.some(ext => f.name.toLowerCase().endsWith(ext))
// File order in a drop or multi-select is up to the OS/browser (often alphabetical, or the
// grabbed file first), so with two files the label map is recognised by its NAME when possible:
// BraTS names its mask *_seg.nii.gz, which would otherwise sort before *_t1.nii.gz.
const LABEL_NAME = /seg|label|mask|aseg|aparc/i

export default function BrainViewerClient() {
  const [volume, setVolume] = useState<File | null>(null)
  const [labelMap, setLabelMap] = useState<File | null>(null)
  const [exampleUrl, setExampleUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const volumeInput = useRef<HTMLInputElement>(null)
  const labelInput = useRef<HTMLInputElement>(null)

  // One file = the volume. Two files = volume + label map (the label map picked by name, else
  // the second file). Anything that is not a supported volume is ignored, with a note saying so.
  const openFiles = (list: FileList | null) => {
    const all = Array.from(list ?? [])
    const files = all.filter(isVolume)
    setNotice(all.length > files.length
      ? `Skipped ${all.length - files.length} unsupported file(s). Supported: ${EXTENSIONS.join(', ')}`
      : null)
    if (files.length === 0) return
    const label = files.length > 1
      ? (files.find(f => LABEL_NAME.test(f.name) && files.some(g => g !== f && !LABEL_NAME.test(g.name))) ?? files[1])
      : null
    setExampleUrl(null)
    setVolume(files.find(f => f !== label) ?? files[0])
    setLabelMap(label)
  }

  const hasSource = !!volume || !!exampleUrl

  return (
    <div
      className="h-screen flex flex-col bg-gray-50 dark:bg-gray-950"
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
      // relatedTarget is where the drag went; null (left the window) or outside the page = done.
      onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setIsDragging(false) }}
      onDrop={(e) => { e.preventDefault(); setIsDragging(false); openFiles(e.dataTransfer.files) }}
    >
      <header className="h-14 flex-shrink-0 px-6 flex items-center justify-between border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2">
            <Image src="/socr.png" alt="SOCR Logo" width={28} height={28} />
            <span className="font-bold text-gray-900 dark:text-gray-100">SOCR BrainGen</span>
          </Link>
          <span className="text-gray-300 dark:text-gray-700">/</span>
          <span className="font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-1.5">
            <Brain className="h-4 w-4" /> Brain Viewer
          </span>
        </div>
        <Link href="/">
          <Button variant="ghost" size="sm">Back to the generator</Button>
        </Link>
      </header>

      <input ref={volumeInput} type="file" accept={ACCEPT} multiple className="hidden"
        onChange={(e) => { openFiles(e.target.files); e.target.value = '' }} />
      <input ref={labelInput} type="file" accept={ACCEPT} className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f && isVolume(f)) { setLabelMap(f); setNotice(null) }
          else if (f) setNotice(`Unsupported file. Supported: ${EXTENSIONS.join(', ')}`)
          e.target.value = ''
        }} />

      {notice && (
        <div className="px-6 py-2 text-sm text-amber-800 bg-amber-50 dark:text-amber-200 dark:bg-amber-950 border-b border-amber-200 dark:border-amber-900">
          {notice}
        </div>
      )}

      {!hasSource ? (
        // Landing state: the SOCR viewer's "Drop files here or Select files"
        <div className="flex-1 flex items-center justify-center p-8">
          <div className={cn(
            "w-full max-w-2xl rounded-2xl border-2 border-dashed p-12 text-center transition-colors",
            isDragging
              ? "border-indigo-400 bg-indigo-50 dark:bg-indigo-950/40"
              : "border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
          )}>
            <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
            <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Drop files here or{' '}
              <button className="text-indigo-600 dark:text-indigo-400 underline" onClick={() => volumeInput.current?.click()}>
                select files
              </button>
            </h1>
            <p className="text-gray-500 dark:text-gray-400 mb-6">
              View a brain volume in 3D and in axial, coronal and sagittal slices.
              Drop two files to show the second as a label map (e.g. a tumor segmentation).
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Supported volumes: {EXTENSIONS.join(' / ')}
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
              Your files stay on your computer; nothing is uploaded.
            </p>
            <Button variant="outline" onClick={() => { setNotice(null); setVolume(null); setLabelMap(null); setExampleUrl(EXAMPLE.url) }}>
              <Brain className="h-4 w-4 mr-2" /> Try an example ({EXAMPLE.name})
            </Button>
          </div>
        </div>
      ) : (
        <>
          {/* Loaded state: what is shown, and controls to change it */}
          <div className="flex-shrink-0 flex flex-wrap items-center gap-3 px-4 py-2 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 text-sm">
            <span className="flex items-center gap-1.5 text-gray-700 dark:text-gray-300">
              <FolderOpen className="h-4 w-4" />
              <span className="font-medium">Volume:</span> {volume?.name ?? EXAMPLE.name}
            </span>
            <span className="flex items-center gap-1.5 text-gray-700 dark:text-gray-300">
              <Layers className="h-4 w-4" />
              <span className="font-medium">Label map:</span> {labelMap?.name ?? 'none'}
              {labelMap && (
                <button aria-label="Remove label map" className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200" onClick={() => setLabelMap(null)}>
                  <X className="h-4 w-4" />
                </button>
              )}
            </span>
            <div className="ml-auto flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={() => labelInput.current?.click()}>
                {labelMap ? 'Replace label map' : 'Add label map'}
              </Button>
              <Button variant="outline" size="sm" onClick={() => volumeInput.current?.click()}>
                Open other files
              </Button>
              <Button variant="ghost" size="sm" onClick={() => { setVolume(null); setLabelMap(null); setExampleUrl(null); setNotice(null) }}>
                Close
              </Button>
            </div>
          </div>
          <div className={cn("flex-1 min-h-0", isDragging && "ring-4 ring-inset ring-indigo-400")}>
            <NiiVueViewer niftiUrl={exampleUrl} file={volume} overlayFile={labelMap} />
          </div>
        </>
      )}
    </div>
  )
}
