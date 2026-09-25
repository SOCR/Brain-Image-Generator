'use client'

import { useState, useEffect } from 'react'
import { toast } from '@/hooks/use-toast'
import ParameterControlPanel from '@/components/projects/ParameterControlPanel'
import ImageViewerPanel, { ImageSet } from '@/components/projects/ImageViewerPanel'

interface GenerationPageClientProps {
  projectId: string
  userId: string
  isPlayground?: boolean
}

// Define image types
type ImageType = 't1' | 't2' | 'flair' | 'seg' | 't1ce'
const IMAGE_TYPES: ImageType[] = ['t1', 't2', 'flair', 'seg', 't1ce']

export default function GenerationPageClient({ projectId, userId, isPlayground = false }: GenerationPageClientProps) {
  // States for model selection
  const [dimensionType, setDimensionType] = useState<'2D' | '3D'>('2D')
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedImageIds, setGeneratedImageIds] = useState<string[]>([])

  // State for image sets
  const [imageSets, setImageSets] = useState<ImageSet[]>([])
  const [selectedSetIndex, setSelectedSetIndex] = useState<number>(0)
  const [selectedImageType, setSelectedImageType] = useState<ImageType>('flair')
  const [numImages, setNumImages] = useState<number>(1)

  // Model parameters
  const [tumour, setTumour] = useState<string>('With Tumor')
  const [sliceOrientation, setSliceOrientation] = useState<string>('Axial')
  const [sliceLocation, setSliceLocation] = useState<string>('Middle')
  const [resolution, setResolution] = useState<string>('64')

  // Conditional-diffusion parameters (braingen_CondDiffuser_BraTS_v1).
  // All parameter state lives here in the parent; ParameterControlPanel is fully
  // controlled and holds none of its own. Defaults MUST be non-empty valid strings:
  // nothing validates params before the POST (the Generate button only checks that a
  // model is selected), so an empty string would go straight to the backend.
  const [lobe, setLobe] = useState<string>('Frontal')
  const [tumourSize, setTumourSize] = useState<string>('Moderate')

  // Which (Lobe, Slice Location) pairs the DEPLOYED backend can actually generate.
  //
  // This is not cosmetic. The conditioning bank fills only 11 of the 18 combinations, and the
  // gaps are anatomy rather than a bug: the cerebellum does not appear in superior slices, and
  // the insula is small enough that it can fail the lobe-area gate at some levels. If the panel
  // offered all 18, picking a missing one would raise in the backend, image_generation.py's
  // blanket `except` would swallow it into an empty list, and FastAPI would return HTTP 200 --
  // a green "Success" toast over a blank viewer. So we ask the backend what exists and grey out
  // the rest.
  //
  // `null` means "not asked yet". That is deliberately distinct from "asked, and the answer was
  // nothing": while it is null the panel leaves every option ENABLED, so a slow or unreachable
  // backend degrades to today's behaviour instead of a control panel where nothing is
  // clickable. The route never rejects -- it returns ready:false with empty lists -- so this
  // settles to a real answer either way.
  const [conddiffCells, setConddiffCells] = useState<
    { ready: boolean; pairs: { lobe: string; slice_location: string }[] } | null
  >(null)

  // Fetched once on mount rather than on every model switch: the answer changes only when
  // someone re-exports the bank and redeploys, and the panel needs it the instant the user
  // picks the model.
  useEffect(() => {
    let cancelled = false
    fetch('/api/conddiff-cells')
      .then(r => r.json())
      .then(d => { if (!cancelled) setConddiffCells({ ready: !!d.ready, pairs: d.pairs || [] }) })
      .catch(() => { if (!cancelled) setConddiffCells({ ready: false, pairs: [] }) })
    // React 18 StrictMode double-invokes effects in development; the flag stops the second
    // response from overwriting state after this component has already unmounted.
    return () => { cancelled = true }
  }, [])

  // Reset selected model when dimension type changes
  useEffect(() => {
    setSelectedModel('')
  }, [dimensionType])

  // The conditional diffusion model runs a real DDIM sampling loop on the CPU (~950 GFLOP per
  // step, 50 steps by default) and backend/image_generation.py loops `for i in range(n_images)`
  // SEQUENTIALLY inside one synchronous request. Asking for 5 multiplies an already
  // minutes-long request by five. Clamp the state itself (not just the input's max) so that a
  // count left over from another model cannot survive the switch, and so the success toast
  // continues to report the truth.
  useEffect(() => {
    if (selectedModel === 'braingen_CondDiffuser_BraTS_v1 (2D)') {
      setNumImages(1)
    }
  }, [selectedModel])

  // Parse image info from generated images
  useEffect(() => {
    const fetchImageDetails = async () => {
      if (generatedImageIds.length === 0) return

      try {
        // For playground mode, generatedImageIds contains direct URLs
        // For authenticated mode, generatedImageIds contains database IDs
        if (isPlayground) {
          // Direct URL processing for playground mode
          const sets: Record<string, ImageSet> = {}
          
          for (const imageUrlsOrUrl of generatedImageIds) {
            // Could be an array of URLs or a single URL
            const urls = Array.isArray(imageUrlsOrUrl) ? imageUrlsOrUrl : [imageUrlsOrUrl]
            const setId = `playground_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
            
            sets[setId] = {
              id: setId,
              name: `Playground Set ${Object.keys(sets).length + 1}`,
              createdAt: new Date().toISOString(),
              images: {
                't1': null,
                't2': null,
                'flair': null,
                'seg': null,
                't1ce': null
              }
            }

            // Map URLs to image types based on filename
            for (const url of urls) {
              if (typeof url === 'string') {
                // Extract image type from URL
                if (url.includes('t1.png')) {
                  sets[setId].images['t1'] = url
                } else if (url.includes('t2.png')) {
                  sets[setId].images['t2'] = url
                } else if (url.includes('flair.png')) {
                  sets[setId].images['flair'] = url
                } else if (url.includes('seg.png')) {
                  sets[setId].images['seg'] = url
                } else if (url.includes('t1ce.png')) {
                  sets[setId].images['t1ce'] = url
                } else if (url.includes('.nii.gz')) {
                  // 3D volume - add to t1 for now
                  sets[setId].images['t1'] = url
                }
              }
            }
          }

          const setsArray = Object.values(sets)
          setImageSets(setsArray)

          // Select first image type that exists in the first set
          if (setsArray.length > 0) {
            setSelectedSetIndex(0)
            const firstSet = setsArray[0]
            const firstAvailableType = IMAGE_TYPES.find(type => firstSet.images[type]) || 'flair'
            setSelectedImageType(firstAvailableType)
          }
        } else {
          // Database fetch for authenticated users
          const response = await fetch('/api/images/batch', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ids: generatedImageIds }),
          })

          if (response.ok) {
            const data = await response.json()
            console.log('API Response Data:', data)
            const images = data.images || []

            // Process images into sets
            const sets: Record<string, ImageSet> = {}

            for (const image of images) {
              console.log('Image:', image)

              // Extract set name from image name (after the dash)
              const nameParts = image.name.split(' - ')
              const imageType = nameParts[0].toLowerCase() as ImageType
              const setName = nameParts[1] || ''

              if (!sets[setName]) {
                sets[setName] = {
                  id: setName,
                  name: setName,
                  createdAt: image.created_at,
                  images: {
                    't1': null,
                    't2': null,
                    'flair': null,
                    'seg': null,
                    't1ce': null
                  }
                }
              }

              // Add image URL to the set
              if (IMAGE_TYPES.includes(imageType)) {
                sets[setName].images[imageType] = image.file_path
              } else if (image.file_path && image.file_path.includes('.nii.gz')) {
                // If it's a nifti file, add it to one of the existing types
                for (const type of IMAGE_TYPES) {
                  if (!sets[setName].images[type]) {
                    sets[setName].images[type] = image.file_path
                    break
                  }
                }
              }
            }

            // Convert to array and sort by creation date (newest first)
            const setsArray = Object.values(sets).sort((a, b) =>
              new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
            )

            setImageSets(setsArray)

            // Select first image type that exists in the first set
            if (setsArray.length > 0) {
              setSelectedSetIndex(0)
              const firstSet = setsArray[0]
              const firstAvailableType = IMAGE_TYPES.find(type => firstSet.images[type]) || 'flair'
              setSelectedImageType(firstAvailableType)
            }
          } else {
            console.error('Failed to fetch images')
          }
        }
      } catch (error) {
        console.error('Error fetching image details:', error)
      }
    }

    fetchImageDetails()
  }, [generatedImageIds, isPlayground])

  // Prepare parameters based on selected model
  const getModelParams = () => {
    if (!selectedModel) return {}

    if (selectedModel === 'braingen_GAN_seg_TCGA_v1 (2D)') {
      return { tumour }
    }

    if (selectedModel === 'braingen_CondDiffuser_BraTS_v1 (2D)') {
      // Keys are snake_case (the file's existing convention: sliceLocation ->
      // slice_location) and must match the backend's params lookups exactly.
      //
      // `sliceLocation` is shared with the cGAN models, whose Sagittal/Coronal
      // orientations can leave it at 'Left' / 'Anterior' etc. This model only
      // understands the axial triple, so coerce rather than send a value the backend
      // has never heard of. ParameterControlPanel applies the identical coercion to the
      // value it DISPLAYS, so the control and the request always agree.
      const level = ['Inferior', 'Middle', 'Superior'].includes(sliceLocation)
        ? sliceLocation
        : 'Middle'

      return {
        tumour,
        lobe,
        slice_location: level,
        tumour_size: tumourSize
      }
    }

    if (selectedModel.includes('braingen_cGAN_Multicontrast') ||
        selectedModel === 'braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)') {
      return {
        tumour,
        slice_orientation: sliceOrientation,
        slice_location: sliceLocation
      }
    }

    if (selectedModel === 'braingen_gan3d_BraTS_64_v1 (3D)') {
      return { resolution }
    }

    return {}
  }

  // Generate image function
  const generateImage = async () => {
    if (!selectedModel) {
      toast({
        title: "Error",
        description: "Please select a model",
        variant: "destructive"
      })
      return
    }

    setIsGenerating(true)

    try {
      const requestBody = {
        user_id: userId,
        project_id: projectId,
        model_name: selectedModel,
        n_images: numImages,
        params: getModelParams(),
        is_playground: isPlayground
      }

      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        // /api/generate puts the backend's reason in `error` (e.g. the GPU quota message).
        const body = await response.json().catch(() => null)
        throw new Error(body?.error || 'Error generating image')
      }

      const data = await response.json()

      // The backend answers HTTP 200 with an EMPTY inner list whenever an inference call
      // raised: backend/image_generation.py catches every exception, prints it to the
      // container log, and substitutes `image_path = []`. For the conditional diffusion
      // model that is the normal failure shape -- a missing checkpoint, a missing
      // conditioning bank, or a (lobe, slice location) combination the shipped bank cannot
      // serve all land here. Without this check the user gets a green "Success" toast over
      // an empty viewer with no indication that anything went wrong.
      const returned: unknown[] = Array.isArray(data.image_ids) ? data.image_ids : []
      // `entry.some(Boolean)` rather than `entry.length > 0`: supabase_storage.add_to_database
      // swallows its own errors and returns None, so a failed upload can come back as [null],
      // which is just as blank to the viewer as [].
      const produced = returned.filter(entry =>
        Array.isArray(entry) ? entry.some(Boolean) : Boolean(entry)
      )
      if (produced.length === 0) {
        throw new Error('The backend returned no images. Check the backend logs for the cause.')
      }

      setGeneratedImageIds(data.image_ids)
      // Report what actually came back, not what was asked for.
      toast({
        title: "Success",
        description: `Generated ${produced.length} image set${produced.length > 1 ? 's' : ''}`,
      })
    } catch (error) {
      console.error('Error:', error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to generate image",
        variant: "destructive"
      })
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="flex-1 flex">
      <ParameterControlPanel
        dimensionType={dimensionType}
        setDimensionType={setDimensionType}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
        numImages={numImages}
        setNumImages={setNumImages}
        tumour={tumour}
        setTumour={setTumour}
        sliceOrientation={sliceOrientation}
        setSliceOrientation={setSliceOrientation}
        sliceLocation={sliceLocation}
        setSliceLocation={setSliceLocation}
        resolution={resolution}
        setResolution={setResolution}
        // Conditional-diffusion parameters -- required props, so these must be threaded
        // here in the same change as the interface addition in ParameterControlPanel.
        lobe={lobe}
        setLobe={setLobe}
        tumourSize={tumourSize}
        setTumourSize={setTumourSize}
        // Optional: `null` until the capability fetch settles, which the panel reads as
        // "leave everything enabled" so a slow backend does not lock the controls.
        conddiffCells={conddiffCells}
        isGenerating={isGenerating}
        onGenerate={generateImage}
      />
      <ImageViewerPanel
        dimensionType={dimensionType}
        imageSets={imageSets}
        isGenerating={isGenerating}
        numImages={numImages}
        selectedSetIndex={selectedSetIndex}
        setSelectedSetIndex={setSelectedSetIndex}
        selectedImageType={selectedImageType}
        setSelectedImageType={setSelectedImageType}
      />
    </div>
  )
}
