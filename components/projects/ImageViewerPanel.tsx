'use client'

import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { toast } from '@/hooks/use-toast'
import {
  Loader2,
  ImageIcon,
  Brain,
  Info,
  Calendar,
  Tag,
  Eye
} from 'lucide-react'
import NiiVueViewer from '@/components/projects/NiiVueViewer'

// Define image types
type ImageType = 't1' | 't2' | 'flair' | 'seg' | 't1ce'
const IMAGE_TYPES: ImageType[] = ['t1', 't2', 'flair', 'seg', 't1ce']

// Readable display names for image types
const IMAGE_TYPE_NAMES: Record<ImageType, string> = {
  't1': 'T1',
  't2': 'T2',
  'flair': 'FLAIR',
  'seg': 'Segment',
  't1ce': 'T1 CE'
}

// Interface for image sets
export interface ImageSet {
  id: string
  images: Record<ImageType, string | null>
  name: string
  createdAt: string
}

interface ImageViewerPanelProps {
  dimensionType: '2D' | '3D'
  imageSets: ImageSet[]
  isGenerating: boolean
  numImages: number
  selectedSetIndex: number
  setSelectedSetIndex: (index: number) => void
  selectedImageType: ImageType
  setSelectedImageType: (type: ImageType) => void
}

export default function ImageViewerPanel({
  dimensionType,
  imageSets,
  isGenerating,
  numImages,
  selectedSetIndex,
  setSelectedSetIndex,
  selectedImageType,
  setSelectedImageType,
}: ImageViewerPanelProps) {
  // Render content based on selected model type (2D/3D)
  const renderContent = () => {
    if (isGenerating) {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-gray-400" />
            <p className="text-gray-500">Generating {numImages} image set{numImages > 1 ? 's' : ''}...</p>
            <p className="text-gray-400 text-sm mt-2">This may take a moment</p>
          </div>
        </div>
      )
    }

    if (imageSets.length === 0) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 bg-white">
          <div className="relative h-16 w-16 mb-6">
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-[var(--color-1)] via-[var(--color-3)] to-[var(--color-5)] opacity-20"></div>
            <ImageIcon className="relative h-16 w-16 text-gray-300" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No Images Generated Yet</h3>
          <p className="text-gray-500 max-w-md">
            Configure your model parameters on the left, then click "Generate" to create brain images.
          </p>
        </div>
      )
    }

    if (dimensionType === '2D') {
      return render2DViewer()
    } else {
      return render3DViewer()
    }
  }

  // Render 2D image viewer with contrast selection
  const render2DViewer = () => {
    const currentSet = imageSets[selectedSetIndex]

    if (!currentSet) {
      return <div className="flex-1 flex items-center justify-center text-gray-500">No image set selected</div>
    }

    // Get available image types for this set
    const availableTypes = IMAGE_TYPES.filter(type => currentSet.images[type])

    return (
      <>
        {/* Fixed Top bar with set selection and info */}
        <div className="flex-shrink-0 flex items-center justify-between bg-white border-b border-gray-200 px-4 py-2">
          <div className="flex items-center gap-2">
            <Label className="text-xs font-medium text-gray-600">Set:</Label>
            <Select
              value={String(selectedSetIndex)}
              onValueChange={(value) => setSelectedSetIndex(Number(value))}
            >
              <SelectTrigger className="h-8 min-w-[200px] text-sm bg-white">
                <SelectValue placeholder="Select set" />
              </SelectTrigger>
              <SelectContent>
                {imageSets.map((set, index) => (
                  <SelectItem key={set.id} value={String(index)} className="text-sm">
                    {set.name} ({new Date(set.createdAt).toLocaleTimeString()})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-gray-500 hover:text-gray-700">
                  <Info size={16} />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-80">
                <div className="space-y-3">
                  <h4 className="font-medium flex items-center gap-2">
                    <Tag size={16} className="text-gray-500" />
                    Image Set Information
                  </h4>
                  <Separator />
                  <div className="grid grid-cols-[20px_1fr] gap-x-2 gap-y-2 items-start">
                    <Tag size={14} className="text-gray-500 mt-0.5" />
                    <div className="text-sm">
                      <span className="font-medium">Name:</span> {currentSet.name}
                    </div>

                    <Calendar size={14} className="text-gray-500 mt-0.5" />
                    <div className="text-sm">
                      <span className="font-medium">Created:</span> {new Date(currentSet.createdAt).toLocaleString()}
                    </div>

                    <Eye size={14} className="text-gray-500 mt-0.5" />
                    <div className="text-sm">
                      <span className="font-medium">Available contrasts:</span>{' '}
                      {availableTypes.map(type => IMAGE_TYPE_NAMES[type]).join(', ')}
                    </div>
                  </div>
                </div>
              </PopoverContent>
            </Popover>
          </div>

          <div className="text-xs text-gray-500 font-medium">
            {selectedSetIndex + 1} / {imageSets.length}
          </div>
        </div>

        {/* Scrollable Image content area */}
        <div className="flex-1 overflow-y-auto bg-white">
          <Tabs
            value={selectedImageType}
            onValueChange={(v) => setSelectedImageType(v as ImageType)}
            className="h-full flex flex-col"
          >
            {availableTypes.map(type => (
              <TabsContent
                key={type}
                value={type}
                className="flex-1 m-0 relative"
              >
                {/* Full-size image container */}
                <div className="absolute inset-0 flex items-center justify-center bg-gray-50 p-8">
                  {currentSet.images[type] ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={currentSet.images[type] || ''}
                      alt={`${IMAGE_TYPE_NAMES[type]} image`}
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <div className="text-gray-400">
                      No {IMAGE_TYPE_NAMES[type]} image available
                    </div>
                  )}
                </div>

                {/* Tab switcher - bottom right corner */}
                <div className="absolute bottom-6 right-6 z-10">
                  <TabsList className="p-1 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-lg shadow-lg">
                    {availableTypes.map(type => (
                      <TabsTrigger
                        key={type}
                        value={type}
                        disabled={!currentSet.images[type]}
                        className="px-4 py-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-[var(--color-1)] data-[state=active]:via-[var(--color-3)] data-[state=active]:to-[var(--color-5)] data-[state=active]:text-white rounded-md min-w-[70px] transition-all"
                      >
                        {IMAGE_TYPE_NAMES[type]}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </>
    )
  }

  // Render 3D viewer
  const render3DViewer = () => {
    const currentSet = imageSets[selectedSetIndex]

    if (!currentSet) {
      return <div className="flex-1 flex items-center justify-center text-gray-500">No 3D volume selected</div>
    }

    // Check all image types - any of them might contain our nifti file
    const niftiUrl = Object.values(currentSet.images).find(url =>
      url && url.includes('.nii.gz')
    )

    if (!niftiUrl) {
      return (
        <>
          <div className="flex-shrink-0 flex items-center justify-between bg-white border-b border-gray-200 px-4 py-2">
            <div className="flex items-center gap-2">
              <Label className="text-xs font-medium text-gray-600">Set:</Label>
              <Select
                value={String(selectedSetIndex)}
                onValueChange={(value) => setSelectedSetIndex(Number(value))}
              >
                <SelectTrigger className="h-8 min-w-[200px] text-sm bg-white">
                  <SelectValue placeholder="Select set" />
                </SelectTrigger>
                <SelectContent>
                  {imageSets.map((set, index) => (
                    <SelectItem key={set.id} value={String(index)} className="text-sm">
                      {set.name} ({new Date(set.createdAt).toLocaleTimeString()})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center bg-white">
            <div className="text-center">
              <Brain className="h-16 w-16 mx-auto text-gray-300 mb-4" />
              <p className="text-gray-600 font-medium">No 3D volume found</p>
              <p className="text-gray-400 text-sm mt-2">
                Could not find a NIfTI file (.nii.gz) in this image set
              </p>
            </div>
          </div>
        </>
      )
    }

    return (
      <>
        {/* Fixed Top bar with set selection and info */}
        <div className="flex-shrink-0 flex items-center justify-between bg-white border-b border-gray-200 px-4 py-2">
          <div className="flex items-center gap-2">
            <Label className="text-xs font-medium text-gray-600">Set:</Label>
            <Select
              value={String(selectedSetIndex)}
              onValueChange={(value) => setSelectedSetIndex(Number(value))}
            >
              <SelectTrigger className="h-8 min-w-[200px] text-sm bg-white">
                <SelectValue placeholder="Select set" />
              </SelectTrigger>
              <SelectContent>
                {imageSets.map((set, index) => (
                  <SelectItem key={set.id} value={String(index)} className="text-sm">
                    {set.name} ({new Date(set.createdAt).toLocaleTimeString()})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-gray-500 hover:text-gray-700">
                  <Info size={16} />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-80">
                <div className="space-y-3">
                  <h4 className="font-medium flex items-center gap-2">
                    <Tag size={16} className="text-gray-500" />
                    Volume Information
                  </h4>
                  <Separator />
                  <div className="grid grid-cols-[20px_1fr] gap-x-2 gap-y-2 items-start">
                    <Tag size={14} className="text-gray-500 mt-0.5" />
                    <div className="text-sm">
                      <span className="font-medium">Name:</span> {currentSet.name}
                    </div>

                    <Calendar size={14} className="text-gray-500 mt-0.5" />
                    <div className="text-sm">
                      <span className="font-medium">Created:</span> {new Date(currentSet.createdAt).toLocaleString()}
                    </div>
                  </div>
                </div>
              </PopoverContent>
            </Popover>
          </div>

          <div className="text-xs text-gray-500 font-medium">
            {selectedSetIndex + 1} / {imageSets.length}
          </div>
        </div>

        {/* 3D Viewer Area */}
        <div className="flex-1 w-full">
          <NiiVueViewer
            niftiUrl={niftiUrl}
            onError={(error) => {
              console.error("NiiVue error:", error)
              toast({
                title: "Error",
                description: "Failed to load 3D volume",
                variant: "destructive"
              })
            }}
          />
        </div>
      </>
    )
  }

  return (
    <div className="flex-1 flex flex-col bg-gray-50">
      {renderContent()}
    </div>
  )
}
