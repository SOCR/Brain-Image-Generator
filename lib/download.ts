/**
 * Browser-side downloads for generated images.
 *
 * WHY fetch-then-save instead of <a href={url} download>: the images live on Supabase
 * Storage, a different origin, and browsers ignore the `download` attribute on cross-origin
 * links -- the file would open in a tab instead of saving. Fetching it into a Blob first
 * makes the link same-origin (a blob: URL), so the name we choose is honoured.
 *
 * Nothing here is gated on sign-in: playground users download exactly like everyone else.
 */

import type { SupabaseClient } from '@supabase/supabase-js'

export interface DownloadableSet {
  name: string
  createdAt: string
  images: Record<string, string | null>
  model?: string
  params?: unknown
  dimensionType?: '2D' | '3D'
}

/** The file name a URL should be saved under: its last path segment (e.g. flair.png, mri3D.nii.gz). */
export function fileNameFromUrl(url: string, fallback = 'image'): string {
  try {
    const last = new URL(url).pathname.split('/').pop()
    return last ? decodeURIComponent(last) : fallback
  } catch {
    return fallback
  }
}

/** Make a set name safe to use as a file name on Windows, macOS and Linux. */
export function safeFileName(name: string): string {
  return name.replace(/[<>:"/\\|?*\x00-\x1f]+/g, '_').replace(/\s+/g, '_').slice(0, 100) || 'brain_images'
}

export function saveBlob(blob: Blob, filename: string) {
  const href = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = href
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  // Revoking immediately can cancel the download in some browsers; give it a moment.
  setTimeout(() => URL.revokeObjectURL(href), 10_000)
}

async function fetchBlob(url: string): Promise<Blob> {
  const res = await fetch(url, { cache: 'no-store' })
  if (!res.ok) throw new Error(`Could not fetch ${fileNameFromUrl(url)} (HTTP ${res.status})`)
  return res.blob()
}

/** Download one generated file under its own name. */
export async function downloadUrl(url: string, filename = fileNameFromUrl(url)) {
  saveBlob(await fetchBlob(url), filename)
}

/**
 * Download every file in a set as one .zip, plus params.json recording how it was made.
 *
 * The same URL can sit under several image-type keys (the 3D path files the one .nii.gz under
 * the first free slot), so files are de-duplicated by URL before zipping.
 */
export async function downloadSetZip(set: DownloadableSet) {
  // Loaded on demand so JSZip is not in the bundle of every page that shows images.
  const { default: JSZip } = await import('jszip')
  const zip = new JSZip()

  const seen = new Set<string>()
  // Storage URLs are deliberately NOT recorded: they contain the account's user and project
  // UUIDs, and this zip is exactly what people share as supplementary data.
  const files: { type: string; file: string }[] = []
  for (const [type, url] of Object.entries(set.images)) {
    if (!url || seen.has(url)) continue
    seen.add(url)
    let file = fileNameFromUrl(url, `${type}`)
    if (files.some(f => f.file === file)) file = `${type}_${file}`
    zip.file(file, await fetchBlob(url))
    files.push({ type, file })
  }
  if (files.length === 0) throw new Error('This set has no files to download')

  zip.file('params.json', JSON.stringify({
    app: 'SOCR Brain Image Generator',
    set: set.name,
    created_at: set.createdAt,
    downloaded_at: new Date().toISOString(),
    model: set.model ?? null,
    // From the files themselves unless the caller knows better: a 3D set is a NIfTI volume.
    dimension: set.dimensionType ?? (files.some(f => f.file.includes('.nii')) ? '3D' : '2D'),
    parameters: set.params ?? null,
    files,
  }, null, 2))

  saveBlob(await zip.generateAsync({ type: 'blob' }), `${safeFileName(set.name)}.zip`)
}

/** `parameters_used` comes back from Supabase as a JSON string or an object; normalise it. */
export function parseParams(raw: unknown): unknown {
  if (typeof raw !== 'string') return raw ?? undefined
  try { return JSON.parse(raw) } catch { return raw }
}

/**
 * Keep a hand-flipped volume as its own Library item, beside the original. The original file and
 * row are left alone: they are the model's unmodified output, and storage allows no overwrite.
 * Being a normal row, the copy works with the Download menu, the zip and the Library unchanged.
 * Returns whether a new item was added (false: these flips were already saved).
 */
export async function saveFlippedVolume(
  supabase: SupabaseClient,
  sourceUrl: string,
  fileName: string,
  flips: string,
  bytes: Uint8Array,
) {
  const sourcePath = sourceUrl.split('?')[0].split('/object/public/volumes/')[1]
  if (!sourcePath) throw new Error('This volume is not a stored result')
  const { data: source, error } = await supabase
    .from('generated_images').select('*').eq('file_path', sourceUrl).limit(1).maybeSingle()
  if (error) throw error
  if (!source) throw new Error('This volume is not in your Library')

  const path = `${sourcePath.slice(0, sourcePath.lastIndexOf('/'))}/${fileName}`
  const volumes = supabase.storage.from('volumes')
  const { error: uploadError } = await volumes.upload(path, bytes, { contentType: 'application/octet-stream' })
  // The same flips saved again produce the same file, so "already exists" is not a failure.
  if (uploadError && !/exist|duplicate/i.test(uploadError.message)) throw uploadError
  const fileUrl = volumes.getPublicUrl(path).data.publicUrl

  const { data: existing } = await supabase
    .from('generated_images').select('id').eq('file_path', fileUrl).limit(1).maybeSingle()
  if (existing) return false

  const { error: insertError } = await supabase.from('generated_images').insert({
    name: `${source.name}_flipped-${flips}`,
    project_id: source.project_id,
    model_id: source.model_id,
    user_id: source.user_id,
    file_path: fileUrl,
    // A set name, not a URL: this reaches the shareable params.json (see downloadSetZip).
    parameters_used: { ...(parseParams(source.parameters_used) as object), flipped: flips, flipped_from: source.name },
  })
  if (insertError) throw insertError
  return true
}
