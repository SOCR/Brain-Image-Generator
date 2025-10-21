'use client'

import { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

// Define image types
export type ImageType = 't1' | 't2' | 'flair' | 'seg' | 't1ce'

// Readable display names for image types
export const IMAGE_TYPE_NAMES: Record<ImageType, string> = {
  't1': 'T1',
  't2': 'T2',
  'flair': 'FLAIR',
  'seg': 'Segment',
  't1ce': 'T1 CE'
}

interface ImageViewerProps {
  images: Record<ImageType, string | null>
  className?: string
  showTabs?: boolean
  defaultType?: ImageType
}

export default function ImageViewer({
  images,
  className = '',
  showTabs = true,
  defaultType = 'flair'
}: ImageViewerProps) {
  // Get available image types
  const availableTypes = (Object.keys(images) as ImageType[]).filter(
    type => images[type]
  )

  const [selectedImageType, setSelectedImageType] = useState<ImageType>(defaultType)

  // Set first available type when images change
  useEffect(() => {
    if (availableTypes.length > 0 && !images[selectedImageType]) {
      setSelectedImageType(availableTypes[0])
    }
  }, [images, availableTypes, selectedImageType])

  if (availableTypes.length === 0) {
    return (
      <div className={`flex items-center justify-center ${className}`}>
        <div className="text-gray-400">No images available</div>
      </div>
    )
  }

  return (
    <Tabs
      value={selectedImageType}
      onValueChange={(v) => setSelectedImageType(v as ImageType)}
      className={`h-full flex flex-col ${className}`}
    >
      {availableTypes.map(type => (
        <TabsContent
          key={type}
          value={type}
          className="flex-1 m-0 relative"
        >
          {/* Full-size image container */}
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50 p-8">
            {images[type] ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={images[type] || ''}
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
          {showTabs && (
            <div className="absolute bottom-6 right-6 z-10">
              <TabsList className="p-1 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-lg shadow-lg">
                {availableTypes.map(type => (
                  <TabsTrigger
                    key={type}
                    value={type}
                    disabled={!images[type]}
                    className="px-4 py-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-[var(--color-1)] data-[state=active]:via-[var(--color-3)] data-[state=active]:to-[var(--color-5)] data-[state=active]:text-white rounded-md min-w-[70px] transition-all"
                  >
                    {IMAGE_TYPE_NAMES[type]}
                  </TabsTrigger>
                ))}
              </TabsList>
            </div>
          )}
        </TabsContent>
      ))}
    </Tabs>
  )
}
