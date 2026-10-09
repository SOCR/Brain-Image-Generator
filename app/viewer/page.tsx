import type { Metadata } from 'next'
import BrainViewerClient from '@/components/viewer/BrainViewerClient'

export const metadata: Metadata = {
  title: 'Brain Viewer | SOCR BrainGen',
  description: 'View MRI volumes (NIfTI, MGH/MGZ, NRRD) in 3D and in axial, coronal and sagittal slices, in your browser.',
}

// Public on purpose (like the playground): no sign-in needed, and nothing is uploaded.
export default function ViewerPage() {
  return <BrainViewerClient />
}
