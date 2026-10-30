'use client'

import { useEffect, useRef, useState } from 'react'
import { Niivue, SHOW_RENDER, MULTIPLANAR_TYPE } from '@niivue/niivue'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import {
  Box,
  Layers,
  SplitSquareVertical,
  Grid3X3,
  Palette,
  Maximize,
  Settings,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { flipVoxels } from '@/lib/flipVolume'

/**
 * Volume viewer modelled on the SOCR Brain Viewer (socr.umich.edu/HTML5/BrainViewer, built on
 * XTK/Slice:Drop): a 3D render beside axial / coronal / sagittal slices, slice-position
 * sliders, Window/Level, 3D opacity and an optional label-map overlay. Built on
 * NiiVue (WebGL2, maintained) rather than XTK (unmaintained), which this app already uses.
 *
 * A volume can come from a URL (generated 3D results on Supabase) or a local File (the
 * /viewer page). Local files are read in the browser and never uploaded.
 */
interface NiiVueViewerProps {
  niftiUrl?: string | null
  file?: File | null
  // Optional label map / overlay drawn over the volume (e.g. a tumour segmentation).
  overlayUrl?: string | null
  overlayFile?: File | null
  onError?: (error: Error) => void
}

// View types supported by NiiVue
type ViewType = 'multiplanar' | 'axial' | 'coronal' | 'sagittal' | 'render'

// Available color maps in NiiVue
const COLOR_MAPS = [
  'gray', 'red', 'green', 'blue', 'winter', 'summer',
  'autumn', 'spring', 'cool', 'hot', 'bone', 'copper',
  'pink', 'hsv', 'jet', 'parula', 'viridis', 'inferno',
  'magma', 'plasma', 'niivue'
]

// Slice sliders, in voxel-index order: vox[0] moves the sagittal plane, vox[1] the coronal,
// vox[2] the axial (for the usual RAS-ordered NIfTI).
const SLICE_AXES = [
  { axis: 2, label: 'Axial' },
  { axis: 1, label: 'Coronal' },
  { axis: 0, label: 'Sagittal' },
] as const

// Anatomical directions a user can flip on a generated volume. `ras` indexes NiiVue's permRAS,
// which says which voxel axis runs right-left (0), anterior-posterior (1), superior-inferior (2).
const FLIP_AXES = [
  { ras: 0, code: 'LR', label: 'Left / right' },
  { ras: 1, code: 'AP', label: 'Front / back' },
  { ras: 2, code: 'SI', label: 'Top / bottom' },
] as const
const NO_FLIPS = [false, false, false]

/** NiiVue loads by URL; a local File becomes a blob: URL, and `name` carries the extension it sniffs. */
function toSource(url?: string | null, file?: File | null) {
  if (file) return { url: URL.createObjectURL(file), name: file.name, isBlob: true }
  if (url) return { url, name: url.split('?')[0].split('/').pop() || 'volume.nii.gz', isBlob: false }
  return null
}

export default function NiiVueViewer({ niftiUrl = null, file = null, overlayUrl = null, overlayFile = null, onError }: NiiVueViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const niivueRef = useRef<Niivue | null>(null)
  // Callers pass inline arrow functions; keeping onError in a ref stops every parent re-render
  // from re-downloading the volume.
  const onErrorRef = useRef(onError)
  onErrorRef.current = onError
  // Loads run one at a time. nv.loadVolumes clears the list and then appends when its download
  // finishes, so two overlapping loads (React dev double-invokes effects; a user can drop files
  // in quick succession) would leave BOTH volumes loaded. Chaining makes the newest load start
  // only after the previous one has finished, and stale queued loads are skipped.
  const loadChainRef = useRef<Promise<void>>(Promise.resolve())
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [viewType, setViewType] = useState<ViewType>('multiplanar')
  const [bgOpacity, setBgOpacity] = useState(1.0)
  const [colorMap, setColorMap] = useState('gray')
  const [isClipPlaneEnabled, setIsClipPlaneEnabled] = useState(false)
  const [isRadiological, setIsRadiological] = useState(false)
  const [isOrientationCube, setIsOrientationCube] = useState(true)
  const [isColorbar, setIsColorbar] = useState(true)
  const [showCrosshair, setShowCrosshair] = useState(true)
  const [contrastMin, setContrastMin] = useState(0)
  const [contrastMax, setContrastMax] = useState(1)
  const [volumeRanges, setVolumeRanges] = useState<[number, number]>([0, 1])
  const [crosshairLocation, setCrosshairLocation] = useState("")
  // Volume size in voxels and the crosshair's voxel position (drives the slice sliders).
  const [dims, setDims] = useState<[number, number, number] | null>(null)
  const [vox, setVox] = useState<[number, number, number]>([0, 0, 0])
  const [hasOverlay, setHasOverlay] = useState(false)
  const [overlayOpacity, setOverlayOpacity] = useState(0.6)
  const [showOverlay, setShowOverlay] = useState(true)
  // Which FLIP_AXES are currently applied to the loaded voxels (reset by every load).
  const [flips, setFlips] = useState<boolean[]>(NO_FLIPS)
  // Generated volumes saved before the atlas fit have an exact identity affine.
  const [isUnfitted, setIsUnfitted] = useState(false)
  // The toolbar stays usable while a volume loads, but the settings effects below skip while
  // nv.volumes is empty. The load reads the LATEST settings from here after it finishes, so a
  // view/colour/opacity/clip change made mid-load is not lost.
  const settingsRef = useRef({ viewType, colorMap, bgOpacity, isClipPlaneEnabled, overlayOpacity, showOverlay })
  settingsRef.current = { viewType, colorMap, bgOpacity, isClipPlaneEnabled, overlayOpacity, showOverlay }

  // Initialize NiiVue
  useEffect(() => {
    if (!canvasRef.current) return

    // Only initialize once
    if (!niivueRef.current) {
      const nv = new Niivue({
        backColor: [0, 0, 0, 1],
        show3Dcrosshair: true,
        isColorbar: true,
        isOrientCube: true,
        meshThicknessOn2D: 0.5,
        meshXRay: 0.0,
        // SOCR-style multiplanar: the three slices AND the 3D render, in a 2x2 grid.
        multiplanarShowRender: SHOW_RENDER.ALWAYS,
        multiplanarLayout: MULTIPLANAR_TYPE.GRID,
        // Dropping a file on the canvas would load it behind React's back; the /viewer page
        // handles drag-and-drop itself.
        dragAndDropEnabled: false,
      })

      // Set up custom location change handler
      nv.onLocationChange = (data: any) => {
        if (data && typeof data.string === 'string') {
          setCrosshairLocation(data.string);
        }
        if (data?.vox) {
          setVox([Math.round(data.vox[0]), Math.round(data.vox[1]), Math.round(data.vox[2])])
        }
      };

      nv.attachToCanvas(canvasRef.current)
      niivueRef.current = nv
    }

    // Set listener for window resize
    const handleResize = () => {
      if (niivueRef.current && canvasRef.current) {
        niivueRef.current.drawScene()
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      // NiiVue registers its own window 'resize' listener that can never be removed, so the
      // instance outlives this component. Drop its volumes so at least the voxel data is freed.
      const nv = niivueRef.current
      if (nv) {
        nv.mediaUrlMap.clear()
        nv.volumes = []
      }
    }
  }, [])

  // Load the volume (and overlay) whenever a source changes
  useEffect(() => {
    let cancelled = false
    const base = toSource(niftiUrl, file)
    const overlay = toSource(overlayUrl, overlayFile)

    const loadVolume = async () => {
      const nv = niivueRef.current
      if (cancelled || !nv || !base) {
        if (!cancelled) setIsLoading(false)
        if (base?.isBlob) URL.revokeObjectURL(base.url)
        if (overlay?.isBlob) URL.revokeObjectURL(overlay.url)
        return
      }

      setIsLoading(true)
      setError(null)
      setFlips(NO_FLIPS)

      try {
        // loadVolumes empties nv.volumes, but NiiVue 0.52 also keeps every loaded image in
        // mediaUrlMap and never clears it there -- without this, each file opened on /viewer
        // would stay in memory until the page reloads.
        nv.mediaUrlMap.clear()
        const s0 = settingsRef.current
        await nv.loadVolumes([
          { url: base.url, name: base.name, colormap: s0.colorMap, opacity: s0.bgOpacity },
          ...(overlay ? [{ url: overlay.url, name: overlay.name, colormap: 'red', opacity: s0.showOverlay ? s0.overlayOpacity : 0 }] : []),
        ])
        if (cancelled) return

        const volume = nv.volumes[0]
        if (volume) {
          // Window/Level starts at the robust range NiiVue computed for this volume.
          const cal_min = volume.cal_min ?? 0
          const cal_max = volume.cal_max ?? 1
          const lo = volume.global_min ?? cal_min
          const hi = volume.global_max ?? cal_max
          setVolumeRanges([lo, hi > lo ? hi : lo + 1])
          setContrastMin(cal_min)
          setContrastMax(cal_max)
          if (volume.dims) setDims([volume.dims[1], volume.dims[2], volume.dims[3]])
          setIsUnfitted(!!volume.hdr?.affine.every((row, i) => row.every((v, j) => v === (i === j ? 1 : 0))))
        }
        setHasOverlay(nv.volumes.length > 1)
        // Apply whatever the toolbar says NOW (it may have changed during the download).
        const s = settingsRef.current
        if (volume) volume.colormap = s.colorMap
        nv.setOpacity(0, s.bgOpacity)
        if (nv.volumes.length > 1) nv.setOpacity(1, s.showOverlay ? s.overlayOpacity : 0)
        nv.setClipPlane(s.isClipPlaneEnabled ? [0, 0, 90] : [2, 0, 90])
        applyViewType(nv, s.viewType)
        nv.createOnLocationChange() // seed the slice sliders with the starting crosshair
        setIsLoading(false)
      } catch (err) {
        if (cancelled) return
        console.error('Error loading volume:', err)
        setError('Failed to load the volume. Supported: .nii, .nii.gz, .mgh, .mgz, .nrrd')
        setDims(null) // no stale slice sliders for a volume that is no longer shown
        setHasOverlay(false)
        setIsLoading(false)
        if (onErrorRef.current && err instanceof Error) {
          onErrorRef.current(err)
        }
      } finally {
        // NiiVue has read the bytes by now; release local-file blob URLs.
        if (base.isBlob) URL.revokeObjectURL(base.url)
        if (overlay?.isBlob) URL.revokeObjectURL(overlay.url)
      }
    }

    // .catch keeps the chain alive: a rejected link would silently stop every later load.
    loadChainRef.current = loadChainRef.current.then(loadVolume).catch(err => console.error('Volume load failed:', err))
    return () => { cancelled = true }
    // Display settings are applied by their own effects below; only a new source reloads.
  }, [niftiUrl, file, overlayUrl, overlayFile])

  const applyViewType = (nv: Niivue, type: ViewType) => {
    switch (type) {
      case 'multiplanar':
        nv.setSliceType(nv.sliceTypeMultiplanar)
        break
      case 'axial':
        nv.setSliceType(nv.sliceTypeAxial)
        break
      case 'coronal':
        nv.setSliceType(nv.sliceTypeCoronal)
        break
      case 'sagittal':
        nv.setSliceType(nv.sliceTypeSagittal)
        break
      case 'render':
        nv.setSliceType(nv.sliceTypeRender)
        nv.setRenderAzimuthElevation(110, 15)
        break
    }
  }

  // Update view type. (Opacity is NOT a dependency: moving the opacity slider must not snap
  // the 3D camera back to its default angle.)
  useEffect(() => {
    if (!niivueRef.current || niivueRef.current.volumes.length === 0) return
    applyViewType(niivueRef.current, viewType)
  }, [viewType])

  // Update settings effects for various properties
  useEffect(() => {
    if (!niivueRef.current || niivueRef.current.volumes.length === 0) return
    niivueRef.current.setOpacity(0, bgOpacity)
  }, [bgOpacity])

  useEffect(() => {
    if (!niivueRef.current || niivueRef.current.volumes.length < 2) return
    niivueRef.current.setOpacity(1, showOverlay ? overlayOpacity : 0)
  }, [overlayOpacity, showOverlay, hasOverlay])

  useEffect(() => {
    if (!niivueRef.current || niivueRef.current.volumes.length === 0) return
    niivueRef.current.volumes[0].colormap = colorMap;
    niivueRef.current.updateGLVolume();
  }, [colorMap])

  useEffect(() => {
    if (!niivueRef.current || niivueRef.current.volumes.length === 0) return
    if (isClipPlaneEnabled) {
      niivueRef.current.setClipPlane([0, 0, 90])
    } else {
      niivueRef.current.setClipPlane([2, 0, 90])
    }
    niivueRef.current.updateGLVolume()
  }, [isClipPlaneEnabled])

  useEffect(() => {
    if (!niivueRef.current) return
    niivueRef.current.setRadiologicalConvention(isRadiological);
  }, [isRadiological])

  useEffect(() => {
    if (!niivueRef.current) return
    niivueRef.current.opts.isOrientCube = isOrientationCube;
    niivueRef.current.drawScene();
  }, [isOrientationCube])

  useEffect(() => {
    if (!niivueRef.current) return
    niivueRef.current.opts.isColorbar = isColorbar;
    niivueRef.current.drawScene();
  }, [isColorbar])

  useEffect(() => {
    if (!niivueRef.current) return
    niivueRef.current.opts.show3Dcrosshair = showCrosshair;
    niivueRef.current.drawScene();
  }, [showCrosshair])

  // Window/Level = the displayed intensity range cal_min..cal_max.
  // ponytail: no separate SOCR-style "Threshold" -- NiiVue 0.52's transparent-below-min
  // (colormapType) only affects overlay layers, so on the main volume it was a no-op.
  useEffect(() => {
    if (!niivueRef.current || niivueRef.current.volumes.length === 0) return
    const volume = niivueRef.current.volumes[0];
    volume.cal_min = contrastMin;
    volume.cal_max = contrastMax;
    niivueRef.current.updateGLVolume();
  }, [contrastMin, contrastMax])

  // Move one slice plane to an absolute voxel index; NiiVue then reports the new location
  // through onLocationChange, which keeps all three sliders and the footer in sync.
  const setSlice = (axis: number, value: number) => {
    const nv = niivueRef.current
    if (!nv || nv.volumes.length === 0) return
    const next: [number, number, number] = [...vox]
    next[axis] = value
    nv.scene.crosshairPos = nv.vox2frac(next)
    nv.drawScene()
    nv.createOnLocationChange()
  }

  // Mirror the anatomy along one direction. The voxels change, not the header, so the viewer's
  // L/R, A/P and S/I labels stay put and any program reading the saved array sees the same brain.
  // ponytail: no flipping with a label map loaded -- NiiVue places an overlay by its own grid, so
  // mirroring both only stays aligned on an identical grid. Add when someone needs it.
  const toggleFlip = (ras: number) => {
    const nv = niivueRef.current
    const vol = nv?.volumes[0]
    if (!nv || !vol?.img || !vol.hdr || !vol.permRAS) return
    flipVoxels(vol.img, vol.hdr.dims.slice(1, 4), Math.abs(vol.permRAS[ras]) - 1)
    nv.updateGLVolume()
    nv.createOnLocationChange() // the footer's intensity at the crosshair has changed
    setFlips(current => current.map((on, i) => (i === ras ? !on : on)))
  }

  // NiiVue writes out the voxels it holds, so this file has the flips applied. The page's own
  // Download menu serves the stored original, which is why this button lives here.
  const downloadFlipped = () => {
    const vol = niivueRef.current?.volumes[0]
    if (!vol) return
    const applied = FLIP_AXES.filter(({ ras }) => flips[ras]).map(({ code }) => code).join('-')
    // Any "_orientation-uncertain" tag stays: a hand flip does not make the atlas fit reliable.
    // saveToDisk always writes NIfTI, so the extension becomes .nii.gz whatever was opened.
    vol.saveToDisk(vol.name.replace(/\.(nii(\.gz)?|mgz|mgh|nrrd)$/i, `_flipped-${applied}.nii.gz`))
  }

  // file wins over niftiUrl in toSource, so the same order decides which name describes the volume.
  const sourceName = file?.name ?? niftiUrl ?? ''
  const isGeneratedVolume = sourceName.includes('/volumes/') || sourceName.includes('mri3D')
  // Why this volume's front/back and top/bottom cannot be trusted, if they cannot.
  const unconfirmedReason = /_flipped-[A-Z-]*(AP|SI)/.test(sourceName)
    ? 'this copy was flipped by hand'
    : sourceName.includes('_orientation-uncertain')
      ? 'this brain could not be confidently matched to the atlas'
      : isUnfitted
        ? 'this brain was generated before atlas fitting was added'
        : null
  const anyFlipped = flips.some(Boolean)

  // Icon button component for toolbar
  const IconButton = ({
    icon: Icon,
    label,
    active = false,
    onClick,
    ...props
  }: {
    icon: any;
    label: string;
    active?: boolean;
    onClick?: () => void;
    [key: string]: any;
  }) => (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
      <button
        className={cn(
          "p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700",
          active && "bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400"
        )}
        onClick={onClick}
        aria-label={label}
        {...props}
      >
        <Icon size={18} />
      </button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{label}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

  return (
    <div className="flex flex-col h-full w-full">
      {/* Compact toolbar */}
      <div className="flex flex-wrap items-center bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 py-1 px-2 gap-1">
        {/* Views label and buttons */}
        <div className="flex items-center mr-2">
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400 mr-2 px-2">Views</span>
          <div className="flex items-center">
            <IconButton
              icon={Grid3X3}
              label="3D + Slices (SOCR layout)"
              active={viewType === 'multiplanar'}
              onClick={() => setViewType('multiplanar')}
            />
            <IconButton
              icon={SplitSquareVertical}
              label="Axial View"
              active={viewType === 'axial'}
              onClick={() => setViewType('axial')}
            />
            <IconButton
              icon={() => <SplitSquareVertical className="rotate-90" />}
              label="Coronal View"
              active={viewType === 'coronal'}
              onClick={() => setViewType('coronal')}
            />
            <IconButton
              icon={() => <SplitSquareVertical className="-rotate-90" />}
              label="Sagittal View"
              active={viewType === 'sagittal'}
              onClick={() => setViewType('sagittal')}
            />
            <IconButton
              icon={Box}
              label="3D Render"
              active={viewType === 'render'}
              onClick={() => setViewType('render')}
            />
          </div>
        </div>

        <Separator orientation="vertical" className="h-6 mx-1" />

        {/* Opacity control */}
        <div className="flex items-center gap-1 mr-2">
          {/* Fades the volume in every view (NiiVue 0.52 has no render-only opacity for the
              main volume), so it is labelled plainly rather than SOCR's "3D Opacity". */}
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400 mr-1 px-2">Opacity</span>
          <Layers size={16} className="text-gray-500 dark:text-gray-400" />
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={[bgOpacity]}
            onValueChange={(values) => setBgOpacity(values[0])}
            className="w-24 h-4"
            aria-label="Volume opacity"
          />
        </div>

        <Separator orientation="vertical" className="h-6 mx-1" />

        {/* Clip plane toggle */}
        <div className="flex items-center gap-1 mr-2">
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400 mr-1 px-2">Clip Plane</span>
          <TooltipProvider delayDuration={300}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  className={cn(
                    "p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700",
                    isClipPlaneEnabled && "bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400"
                  )}
                  onClick={() => setIsClipPlaneEnabled(!isClipPlaneEnabled)}
                  aria-label="Toggle Clip Plane"
                >
                  <Maximize size={18} />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Toggle Clip Plane</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        <Separator orientation="vertical" className="h-6 mx-1" />

        {/* Color and Window/Level popover */}
        <div className="flex items-center gap-1 mr-2">
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400 mr-1 px-2">Window/Level</span>
          <Popover>
            <TooltipProvider delayDuration={300}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverTrigger asChild>
                    <button className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700" aria-label="Color and Window/Level">
                      <Palette size={18} />
                    </button>
                  </PopoverTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Color and Window/Level</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          <PopoverContent className="w-80 bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700">
            <div className="space-y-4">
              <h4 className="font-medium">Color Settings</h4>

              <div>
                <Label htmlFor="color-map-select" className="text-gray-600 dark:text-gray-400 text-sm mb-1 block">
                  Color Map
                </Label>
                <Select value={colorMap} onValueChange={setColorMap}>
                  <SelectTrigger id="color-map-select" className="bg-white dark:bg-gray-800">
                    <SelectValue placeholder="Select color map" />
                  </SelectTrigger>
                  <SelectContent>
                    {COLOR_MAPS.map(map => (
                      <SelectItem key={map} value={map}>
                        {map.charAt(0).toUpperCase() + map.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-2">
                <Label htmlFor="colorbar-toggle" className="text-gray-600 dark:text-gray-400 text-sm">
                  Show Colorbar
                </Label>
                <Switch
                  id="colorbar-toggle"
                  checked={isColorbar}
                  onCheckedChange={setIsColorbar}
                />
              </div>

              <div>
                <Label className="text-gray-600 dark:text-gray-400 text-sm mb-1 block">
                  Window / Level
                </Label>
                <div className="flex gap-3 items-center">
                  <div className="flex flex-col items-center">
                    <span className="text-xs font-medium text-gray-600 dark:text-gray-300 mb-0.5">Min</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">{contrastMin.toFixed(1)}</span>
                  </div>
                  <Slider
                    min={volumeRanges[0]}
                    max={volumeRanges[1]}
                    step={(volumeRanges[1] - volumeRanges[0]) / 100}
                    value={[contrastMin, contrastMax]}
                    onValueChange={([min, max]) => {
                      setContrastMin(min);
                      setContrastMax(max);
                    }}
                    className="flex-1 h-6"
                    aria-label="Window range"
                  />
                  <div className="flex flex-col items-center">
                    <span className="text-xs font-medium text-gray-600 dark:text-gray-300 mb-0.5">Max</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">{contrastMax.toFixed(1)}</span>
                  </div>
                </div>
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1.5 text-center">
                  Window = Max − Min, Level = their midpoint
                </p>
              </div>
            </div>
          </PopoverContent>
          </Popover>
        </div>

        {/* Display options popover */}
        <div className="flex items-center gap-1">
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400 mr-1 px-2">Display</span>
          <Popover>
            <TooltipProvider delayDuration={300}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverTrigger asChild>
                    <button className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700" aria-label="Display Options">
                      <Settings size={18} />
                    </button>
                  </PopoverTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Display Options</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          <PopoverContent className="w-72 bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700">
            <div className="space-y-3">
              <h4 className="font-medium">Display Options</h4>

                  <div className="flex items-center justify-between">
                    <Label htmlFor="radiological-toggle" className="text-gray-600 dark:text-gray-400 text-sm">
                      Radiological Convention
                    </Label>
                <Switch
                  id="radiological-toggle"
                  checked={isRadiological}
                  onCheckedChange={setIsRadiological}
                />
              </div>

                  <div className="flex items-center justify-between">
                    <Label htmlFor="orientation-cube-toggle" className="text-gray-600 dark:text-gray-400 text-sm">
                      Orientation Cube
                    </Label>
                <Switch
                  id="orientation-cube-toggle"
                  checked={isOrientationCube}
                  onCheckedChange={setIsOrientationCube}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="crosshair-toggle" className="text-gray-600 dark:text-gray-400 text-sm">
                  Show Crosshair
                </Label>
                <Switch
                  id="crosshair-toggle"
                  checked={showCrosshair}
                  onCheckedChange={setShowCrosshair}
                />
              </div>
            </div>
          </PopoverContent>
          </Popover>
        </div>
      </div>

      {/* Slice position sliders (+ label map) -- the SOCR viewer's per-plane bars */}
      {dims && !isLoading && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 py-1.5 px-4">
          {SLICE_AXES.map(({ axis, label }) => (
            <div key={label} className="flex items-center gap-2 min-w-[180px] flex-1">
              <span className="text-xs font-medium text-gray-600 dark:text-gray-400 w-14">{label}</span>
              <Slider
                min={0}
                max={Math.max(0, dims[axis] - 1)}
                step={1}
                value={[Math.min(vox[axis], Math.max(0, dims[axis] - 1))]}
                onValueChange={([v]) => setSlice(axis, v)}
                className="flex-1"
                aria-label={`${label} slice`}
              />
              <span className="text-xs tabular-nums text-gray-500 dark:text-gray-400 w-14 text-right">
                {vox[axis] + 1}/{dims[axis]}
              </span>
            </div>
          ))}
          {hasOverlay && (
            <div className="flex items-center gap-2 min-w-[220px] flex-1">
              <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Label map</span>
              <Switch checked={showOverlay} onCheckedChange={setShowOverlay} aria-label="Show label map" />
              <Slider
                min={0}
                max={1}
                step={0.05}
                value={[overlayOpacity]}
                onValueChange={([v]) => setOverlayOpacity(v)}
                disabled={!showOverlay}
                className="flex-1"
                aria-label="Label map opacity"
              />
              <span className="text-xs tabular-nums text-gray-500 dark:text-gray-400 w-10 text-right">
                {Math.round(overlayOpacity * 100)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* The backend fits every generated 3D volume to an atlas to fix its anatomical direction
          (backend/inference/orient3d.py) and marks the file name when that fit was unreliable.
          Left/right can never be fitted, so every generated volume gets the manual flips. */}
      {isGeneratedVolume && !error && (
        <div className={cn(
          "px-4 py-1.5 text-xs border-b space-y-1",
          unconfirmedReason
            ? "text-amber-800 bg-amber-50 dark:text-amber-200 dark:bg-amber-950 border-amber-200 dark:border-amber-900"
            : "text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-800"
        )}>
          <p>
            {unconfirmedReason
              ? <>
                  <span className="font-medium">Orientation not confirmed:</span>{' '}
                  {unconfirmedReason}, so front/back and top/bottom may be reversed. In the sagittal
                  panel (P at left, S at top) the cerebellum belongs at the lower left. Left/right
                  cannot be determined.
                </>
              : 'Generated volume: front/back and top/bottom are fitted to an atlas. Left/right cannot be determined and follows the BraTS convention, so flip it if you need the other side.'}
          </p>
          {hasOverlay ? (
            <p>Flipping is off while a label map is loaded, because it might not stay aligned. Remove the label map to flip.</p>
          ) : (
          <div role="group" aria-label="Flip orientation" className="flex flex-wrap items-center gap-1.5">
            <span className="font-medium mr-1">Flip</span>
            {FLIP_AXES.map(({ ras, label }) => (
              <Button
                key={ras}
                size="sm"
                variant={flips[ras] ? 'default' : 'outline'}
                className="h-6 px-2 text-xs"
                aria-pressed={flips[ras]}
                disabled={isLoading}
                onClick={() => toggleFlip(ras)}
              >
                {label}
              </Button>
            ))}
            {anyFlipped && (
              <>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 px-2 text-xs"
                  onClick={() => FLIP_AXES.forEach(({ ras }) => { if (flips[ras]) toggleFlip(ras) })}
                >
                  Reset
                </Button>
                <Button size="sm" variant="outline" className="h-6 px-2 text-xs" onClick={downloadFlipped}>
                  Download flipped copy
                </Button>
                <span>Flips change this view, not the stored file. Only this button saves them; any other download gives the original.</span>
              </>
            )}
          </div>
          )}
        </div>
      )}

      {/* Canvas container */}
      <div className={cn("relative flex-grow w-full min-h-0", isLoading ? "opacity-50" : "opacity-100")}>
        <canvas
          ref={canvasRef}
          className="w-full h-full"
        ></canvas>

        {/* Error state. An overlay, not an early return: the canvas NiiVue is attached to must
            stay mounted, or the next (valid) file would load into a detached canvas. */}
        {error && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50 dark:bg-gray-950">
            <div className="text-center p-8">
              <div className="text-red-500 mb-4">⚠️</div>
              <h3 className="text-lg font-medium text-gray-800 dark:text-gray-200 mb-2">Error Loading 3D Volume</h3>
              <p className="text-gray-600 dark:text-gray-400">{error}</p>
            </div>
          </div>
        )}

        {/* Loading indicator */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-800/20 dark:bg-gray-950/40">
            <div className="bg-white p-4 rounded-lg shadow-lg">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
              <p className="mt-2 text-center text-sm font-medium">Loading volume...</p>
            </div>
          </div>
        )}
      </div>

      {/* Footer with crosshair location information */}
      {crosshairLocation && (
        <div className="bg-gray-50 dark:bg-gray-900 border-t dark:border-gray-800 p-1 text-xs text-gray-700 dark:text-gray-300 font-mono truncate">
          {crosshairLocation}
        </div>
      )}
    </div>
  )
}
