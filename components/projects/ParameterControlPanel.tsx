'use client'

import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Loader2 } from 'lucide-react'

interface ParameterControlPanelProps {
  // Model selection
  dimensionType: '2D' | '3D'
  setDimensionType: (value: '2D' | '3D') => void
  selectedModel: string
  setSelectedModel: (value: string) => void
  numImages: number
  setNumImages: (value: number) => void

  // Model parameters
  tumour: string
  setTumour: (value: string) => void
  sliceOrientation: string
  setSliceOrientation: (value: string) => void
  sliceLocation: string
  setSliceLocation: (value: string) => void
  resolution: string
  setResolution: (value: string) => void

  // Conditional-diffusion parameters (braingen_CondDiffuser_BraTS_v1)
  // These two are NEW and are read ONLY inside that model's branch of
  // renderParameterFields(), so every existing model is untouched.
  // NOTE: all props in this interface are required (no `?`), so the caller
  // GenerationPageClient.tsx MUST pass them in the same change or the setter is
  // undefined at runtime -- next.config.ts sets typescript.ignoreBuildErrors,
  // so the build will NOT catch a missing prop for you.
  lobe: string
  setLobe: (value: string) => void
  tumourSize: string
  setTumourSize: (value: string) => void

  // Which (Lobe, Slice Location) pairs the deployed backend can actually generate, from
  // GET /api/conddiff-cells. OPTIONAL, unlike the two above, and `null` means "not asked yet":
  // the conditioning bank fills only 11 of the 18 combinations (the cerebellum has no superior
  // slices; the insula can miss the lobe-area gate), and offering a missing one yields HTTP 200
  // with no images -- a "Success" toast over a blank viewer. While this is null or absent every
  // option stays enabled, so a slow backend degrades to the old behaviour rather than to a
  // panel where nothing can be clicked.
  conddiffCells?: { ready: boolean; pairs: { lobe: string; slice_location: string }[] } | null

  // WaveDiT parameters (braingen_WaveDiT_v1 (3D))
  // Age conditions the volume; steps is the ODE integration count; seed makes a sample
  // reproducible. Like lobe/tumourSize above these are REQUIRED props (no `?`), so
  // GenerationPageClient must pass them in the same change -- next.config.ts sets
  // typescript.ignoreBuildErrors, so the build will NOT catch a missing prop for you.
  age: number
  setAge: (value: number) => void
  steps: number
  setSteps: (value: number) => void
  seed: number
  setSeed: (value: number) => void

  // Generation state
  isGenerating: boolean
  onGenerate: () => void
}

export default function ParameterControlPanel({
  dimensionType,
  setDimensionType,
  selectedModel,
  setSelectedModel,
  numImages,
  setNumImages,
  tumour,
  setTumour,
  sliceOrientation,
  setSliceOrientation,
  sliceLocation,
  setSliceLocation,
  resolution,
  setResolution,
  lobe,
  setLobe,
  tumourSize,
  setTumourSize,
  age,
  setAge,
  steps,
  setSteps,
  seed,
  setSeed,
  conddiffCells = null,
  isGenerating,
  onGenerate,
}: ParameterControlPanelProps) {
  // ---- capability helpers for braingen_CondDiffuser_BraTS_v1 -------------------------------
  // `unknown` is the not-yet-fetched case; treat it as "everything allowed" so the controls are
  // never dead. Once the answer arrives we consult the pair list, because the constraint is a
  // JOINT one: Cerebellum is fine at Inferior and impossible at Superior, so neither dropdown
  // can be filtered on its own -- each must be filtered against the other's current value.
  const cellsKnown = conddiffCells != null && conddiffCells.ready
  const lobeAllowedAt = (l: string, level: string) =>
    !cellsKnown || conddiffCells!.pairs.some(p => p.lobe === l && p.slice_location === level)
  const levelAllowedFor = (level: string, l: string) => lobeAllowedAt(l, level)
  // Available models
  const models2D = [
    { id: 'braingen_GAN_seg_TCGA_v1 (2D)', name: 'GAN Segmentation TCGA v1' },
    { id: 'braingen_cGAN_Multicontrast_BraTS_v1 (2D)', name: 'cGAN Multicontrast BraTS v1' },
    { id: 'braingen_cGAN_Multicontrast_seg_BraTS_v1 (2D)', name: 'cGAN Multicontrast seg BraTS v1' },
    { id: 'braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)', name: 'Wavelet GAN Multicontrast BraTS v1' },
    // Conditional diffusion model (the paper's model). The `id` is sent to the backend
    // VERBATIM as `model_name`, so it must match the Python literal byte-for-byte --
    // including the space before '(2D)'. It also deliberately does NOT contain the
    // substring 'braingen_cGAN_Multicontrast', which is matched with .includes() below,
    // so it cannot be captured by that model's parameter branch.
    { id: 'braingen_CondDiffuser_BraTS_v1 (2D)', name: 'Conditional Diffusion BraTS v1' },
  ]

  const models3D = [
    // WaveDiT, age-conditioned wavelet flow matching. The `id` is sent to the backend VERBATIM as
    // `model_name`, so it must match image_generation.py's `elif` byte-for-byte, including the
    // space before '(3D)'. The label carries no resolution: output size is its own control below,
    // the same way the GAN 3D model does it.
    { id: 'braingen_WaveDiT_v1 (3D)', name: 'WaveDiT' },
    { id: 'braingen_gan3d_BraTS_64_v1 (3D)', name: 'GAN 3D BraTS 64 v1' },
  ]

  // Output resolutions WaveDiT offers. Only the native 128^3 checkpoint is deployed, so this is a
  // one-item list rather than an arbitrary choice -- but it is a real control, because the value
  // is sent to the backend and recorded with the image, and because a one-item list reads as
  // "this is the model's output size" instead of leaving the user to guess.
  const WAVEDIT_RESOLUTIONS = ['128']

  // Per-model ceiling on "Number of Images". The GANs are a single forward pass each, so five
  // is cheap. The conditional diffusion model runs a full DDIM sampling loop on the CPU and the
  // backend executes the n_images loop SEQUENTIALLY inside one synchronous request, so five
  // would multiply a request that is already minutes long. GenerationPageClient additionally
  // clamps the state when this model is selected; this is the control that stops the user
  // raising it again.
  //
  // WaveDiT is capped for the same reason and one more: each sample is an ODE loop over a 1.7 GB
  // checkpoint, and image_generation.py clamps to wavedit_inference.MAX_IMAGES (1) server-side,
  // so offering 5 here would let the UI promise images the backend will not produce.
  const ONE_IMAGE_ONLY = ['braingen_CondDiffuser_BraTS_v1 (2D)', 'braingen_WaveDiT_v1 (3D)']
  const maxImages = ONE_IMAGE_ONLY.includes(selectedModel) ? 1 : 5

  // Is the CURRENT conddiff selection actually generatable? Computed here at component scope,
  // not inside renderParameterFields(), because the Generate button needs it and that helper
  // returns JSX rather than a validity flag.
  //
  // This has to gate the BUTTON, not just render a warning. Radix leaves a selected value in
  // place even after that value's SelectItem becomes disabled, so choosing Cerebellum at
  // Inferior and then switching to Superior leaves an unrenderable pair selected. Nothing else
  // validates before the POST -- GenerationPageClient sends whatever the state holds -- and the
  // backend's blanket except turns the miss into HTTP 200 with an empty image list, i.e. a
  // green "Success" toast over a blank viewer. Blocking here is the only place that stops it.
  const conddiffPairUnavailable =
    selectedModel === 'braingen_CondDiffuser_BraTS_v1 (2D)' &&
    tumour !== 'Without Tumor' &&                       // no lesion -> the lobe is unused
    cellsKnown &&                                       // unknown -> allow, do not lock the UI
    !lobeAllowedAt(lobe, ['Inferior', 'Middle', 'Superior'].includes(sliceLocation)
      ? sliceLocation : 'Middle')                       // same coercion the panel displays

  // Show parameter fields based on selected model
  const renderParameterFields = () => {
    if (!selectedModel) return null

    if (selectedModel === 'braingen_GAN_seg_TCGA_v1 (2D)') {
      return (
        <div className="space-y-3 mt-4">
          <div>
            <Label htmlFor="tumour-select">Tumour</Label>
            <Select value={tumour} onValueChange={setTumour}>
              <SelectTrigger id="tumour-select">
                <SelectValue placeholder="Select tumor option" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="With Tumor">With Tumor</SelectItem>
                <SelectItem value="Without Tumor">Without Tumor</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      )
    }

    if (selectedModel === 'braingen_CondDiffuser_BraTS_v1 (2D)') {
      // Conditional diffusion model. Four user-facing axes:
      //   Tumour         -- reuses the site-wide `tumour` state so the params key and the
      //                     exact strings ("With Tumor" / "Without Tumor") stay consistent
      //                     with every other model's backend signature.
      //   Lobe           -- NEW. Atlas-guided lesion placement is this model's novel axis;
      //                     no other model on the site has it.
      //   Slice Location -- reuses `sliceLocation`; the option set is identical to the
      //                     Axial branch of getLocationOptions() below. This model is
      //                     axial-only, so NO Slice Orientation control is rendered and
      //                     `sliceOrientation` is simply never sent.
      //   Tumour Size    -- NEW.
      const noTumour = tumour === 'Without Tumor'

      // `sliceLocation` is shared with the cGAN models, where picking Sagittal/Coronal can
      // leave it at 'Left' / 'Anterior' etc. Those values have no matching SelectItem here,
      // which would render a BLANK trigger. Coerce the DISPLAYED value to 'Middle' so the
      // control always shows something valid -- GenerationPageClient.getModelParams()
      // applies the identical coercion, so what the user sees is what gets sent.
      const AXIAL_LEVELS = ['Inferior', 'Middle', 'Superior']
      const levelValue = AXIAL_LEVELS.includes(sliceLocation) ? sliceLocation : 'Middle'

      return (
        <div className="space-y-3 mt-4">
          <div>
            <Label htmlFor="tumour-select">Tumour</Label>
            <Select value={tumour} onValueChange={setTumour}>
              <SelectTrigger id="tumour-select">
                <SelectValue placeholder="Select tumor option" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="With Tumor">With Tumor</SelectItem>
                <SelectItem value="Without Tumor">Without Tumor</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Lobe and Tumour Size describe the synthesized lesion, so they are meaningless
              when there is no lesion. Disabled (rather than removed) on "Without Tumor" so
              the panel does not jump around and the user can see what the axes are. */}
          <div>
            <Label htmlFor="lobe-select">Lobe</Label>
            <Select value={lobe} onValueChange={setLobe} disabled={noTumour}>
              <SelectTrigger id="lobe-select">
                <SelectValue placeholder="Select lobe" />
              </SelectTrigger>
              <SelectContent>
                {/* Listed in the model's own atlas channel order (atlas lobe k lives at
                    9-channel stack index 3+k), so the dropdown reads the way the training
                    data is laid out. The ORDER here is cosmetic: the backend maps each value
                    through a name->name dict (conddiff_inference.UI_LOBE) and then takes
                    LOBES.index(...), so reordering these items changes nothing.
                    What IS load-bearing is the exact `value` SPELLING AND CAPITALISATION --
                    those strings are the literal keys of UI_LOBE. An unrecognised value does
                    not raise; params_to_spec silently falls back to "frontal", so writing
                    e.g. "insula" instead of "Insula" would place every lesion in the frontal
                    lobe while the UI still said Insula. Change these strings only together
                    with UI_LOBE. */}
                {/* Each item is disabled when the deployed bank has no slice for
                    (this lobe, the currently selected level) -- 7 of the 18 pairs are empty and
                    the gaps are anatomy, not a bug. Rendered-but-disabled rather than removed
                    so the user can see that the lobe exists and infer that the LEVEL is what
                    rules it out; silently dropping items would look like a shorter menu. */}
                {['Frontal', 'Parietal', 'Temporal', 'Occipital', 'Cerebellum', 'Insula'].map(l => (
                  <SelectItem key={l} value={l} disabled={!lobeAllowedAt(l, levelValue)}>
                    {l}{lobeAllowedAt(l, levelValue) ? '' : ` — not available at ${levelValue}`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {/* Unlike the other four models, this one does not invent the anatomy: the
                conditioning channels carry a real held-out BraTS subject's T1, and the model
                synthesizes only the FLAIR contrast and the lesion. Saying so here is the only
                place a user ever sees it -- the backend records it in the Supabase
                `parameters_used` column, which no component in this app renders, and in
                playground mode that row is never written at all. */}
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1.5">
              Lesion placement is guided by the ICBM452 lobar atlas warped to SRI24. The
              underlying anatomy comes from a held-out BraTS subject; the FLAIR contrast and the
              lesion are model-generated.
            </p>
          </div>

          <div>
            <Label htmlFor="location-select">Slice Location</Label>
            <Select value={levelValue} onValueChange={setSliceLocation}>
              <SelectTrigger id="location-select">
                <SelectValue placeholder="Select slice location" />
              </SelectTrigger>
              <SelectContent>
                {/* Filtered against the CURRENT lobe, the mirror of the Lobe control above.
                    Both directions are needed because the constraint is a joint one: with
                    Cerebellum selected only Inferior remains, and with Superior selected
                    Cerebellum disappears. Filtering only one dropdown would let the user walk
                    into an empty pair from the other side.
                    When there is no tumour the level is the ONLY thing that matters and every
                    level has slices, so drop the filter rather than disabling options for a
                    lobe that is not being used. */}
                {[['Inferior', 'Inferior (bottom)'], ['Middle', 'Middle'], ['Superior', 'Superior (top)']]
                  .map(([value, label]) => {
                    const ok = noTumour || levelAllowedFor(value, lobe)
                    return (
                      <SelectItem key={value} value={value} disabled={!ok}>
                        {label}{ok ? '' : ` — no ${lobe.toLowerCase()} slices`}
                      </SelectItem>
                    )
                  })}
              </SelectContent>
            </Select>
          </div>

          {/* The pair can still go stale: pick Cerebellum at Inferior, then switch the level to
              Superior, and the lobe select is left holding a value its own list now disables.
              Radix does not clear a disabled selection, and nothing else validates before the
              POST, so without this the request would go out for a cell that does not exist and
              come back as HTTP 200 with no images. Warn and block instead. */}
          {!noTumour && cellsKnown && !lobeAllowedAt(lobe, levelValue) && (
            <p className="text-xs text-amber-600 dark:text-amber-500">
              This deployment has no {lobe.toLowerCase()} conditioning slice at the{' '}
              {levelValue.toLowerCase()} level. Pick a different lobe or slice location — the
              combination cannot be generated.
            </p>
          )}

          <div>
            <Label htmlFor="tumour-size-select">Tumour Size</Label>
            <Select value={tumourSize} onValueChange={setTumourSize} disabled={noTumour}>
              <SelectTrigger id="tumour-size-select">
                <SelectValue placeholder="Select tumour size" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Small">Small</SelectItem>
                <SelectItem value="Moderate">Moderate</SelectItem>
                <SelectItem value="Large">Large</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {noTumour && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Lobe and Tumour Size are unused when generating without a tumour.
            </p>
          )}
        </div>
      )
    }

    if (selectedModel.includes('braingen_cGAN_Multicontrast') ||
        selectedModel === 'braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)') {
      // Get location options based on orientation
      const getLocationOptions = () => {
        switch (sliceOrientation) {
          case 'Axial':
            return [
              { value: 'Inferior', label: 'Inferior (bottom)' },
              { value: 'Middle', label: 'Middle' },
              { value: 'Superior', label: 'Superior (top)' }
            ];
          case 'Coronal':
            return [
              { value: 'Anterior', label: 'Anterior (front)' },
              { value: 'Middle', label: 'Middle' },
              { value: 'Posterior', label: 'Posterior (back)' }
            ];
          case 'Sagittal':
            return [
              { value: 'Left', label: 'Left' },
              { value: 'Middle', label: 'Middle' },
              { value: 'Right', label: 'Right' }
            ];
          default:
            return [
              { value: 'Middle', label: 'Middle' }
            ];
        }
      };

      const locationOptions = getLocationOptions();

      return (
        <div className="space-y-3 mt-4">
          <div>
            <Label htmlFor="tumour-select">Tumour</Label>
            <Select value={tumour} onValueChange={setTumour}>
              <SelectTrigger id="tumour-select">
                <SelectValue placeholder="Select tumor option" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="With Tumor">With Tumor</SelectItem>
                <SelectItem value="Without Tumor">Without Tumor</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="orientation-select">Slice Orientation</Label>
            <Select value={sliceOrientation} onValueChange={(value) => {
              setSliceOrientation(value);
              // Reset location to Middle when orientation changes
              setSliceLocation('Middle');
            }}>
              <SelectTrigger id="orientation-select">
                <SelectValue placeholder="Select slice orientation" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Axial">Axial</SelectItem>
                <SelectItem value="Coronal">Coronal</SelectItem>
                <SelectItem value="Sagittal">Sagittal</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="location-select">Slice Location</Label>
            <Select value={sliceLocation} onValueChange={setSliceLocation}>
              <SelectTrigger id="location-select">
                <SelectValue placeholder="Select slice location" />
              </SelectTrigger>
              <SelectContent>
                {locationOptions.map(option => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      )
    }

    if (selectedModel === 'braingen_gan3d_BraTS_64_v1 (3D)') {
      return (
        <div className="space-y-3 mt-4">
          <div>
            <Label htmlFor="resolution-select">Resolution</Label>
            <Select value={resolution} onValueChange={setResolution}>
              <SelectTrigger id="resolution-select">
                <SelectValue placeholder="Select resolution" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="32">32</SelectItem>
                <SelectItem value="64">64</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      )
    }

    if (selectedModel === 'braingen_WaveDiT_v1 (3D)') {
      // WaveDiT's own axes: age conditions the volume, steps is the ODE integration count and
      // seed makes a sample reproducible.
      return (
        <div className="space-y-4 mt-4">
          {/* Resolution */}
          <div>
            <Label htmlFor="wavedit-resolution-select">Resolution</Label>
            <Select value={resolution} onValueChange={setResolution}>
              <SelectTrigger id="wavedit-resolution-select" className="bg-white dark:bg-gray-800 mt-1">
                <SelectValue placeholder="Select resolution" />
              </SelectTrigger>
              <SelectContent>
                {WAVEDIT_RESOLUTIONS.map(r => (
                  <SelectItem key={r} value={r}>{r}³</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Native output size of the deployed checkpoint.
            </p>
          </div>

          <div>
            <Label htmlFor="age-input">Patient Age</Label>
            <Input
              id="age-input"
              type="number"
              min={6}
              max={95}
              value={age}
              onChange={(e) => setAge(Math.max(6, Math.min(95, parseInt(e.target.value) || 50)))}
              className="w-full bg-white dark:bg-gray-800 mt-1"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Conditioning range: 6 to 95 years
            </p>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1.5">
              <Label htmlFor="steps-slider">Sampling Steps</Label>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">
                {steps}
              </span>
            </div>
            <Slider
              id="steps-slider"
              min={1}
              max={50}
              step={1}
              value={[steps]}
              onValueChange={(val) => setSteps(val[0] ?? 10)}
              className="mt-1"
            />
            <div className="flex justify-between text-[10px] text-gray-400 dark:text-gray-500 mt-1">
              <span>1 (fastest)</span>
              <span>10 (default)</span>
              <span>50 (finest)</span>
            </div>
          </div>

          <div>
            <Label htmlFor="seed-input">Random Seed</Label>
            <Input
              id="seed-input"
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
              className="w-full bg-white dark:bg-gray-800 mt-1"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Same seed and steps reproduces the same volume.
            </p>
          </div>
        </div>
      )
    }

    return null
  }

  return (
    <div className="w-96 flex flex-col bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800">
      {/* Fixed Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-gray-200 dark:border-gray-800">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Generation Controls</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Configure model parameters</p>
      </div>

      {/* Scrollable Middle Section */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-6">
          {/* Model Selection Section */}
          <div className="space-y-4">
            <div>
              <Label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">Dimension Type</Label>
              <RadioGroup value={dimensionType} onValueChange={setDimensionType} className="flex gap-4">
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="2D" id="r1" />
                  <Label htmlFor="r1" className="text-sm text-gray-700 dark:text-gray-300 cursor-pointer">2D</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="3D" id="r2" />
                  <Label htmlFor="r2" className="text-sm text-gray-700 dark:text-gray-300 cursor-pointer">3D</Label>
                </div>
              </RadioGroup>
            </div>

            <div>
              <Label htmlFor="model-select" className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger id="model-select" className="bg-white dark:bg-gray-800">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {dimensionType === '2D'
                    ? models2D.map(model => (
                        <SelectItem key={model.id} value={model.id}>{model.name}</SelectItem>
                      ))
                    : models3D.map(model => (
                        <SelectItem key={model.id} value={model.id}>{model.name}</SelectItem>
                      ))
                  }
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="num-images" className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">Number of Images</Label>
              <Input
                id="num-images"
                type="number"
                min="1"
                max={maxImages}
                value={numImages}
                onChange={(e) => setNumImages(Math.max(1, Math.min(maxImages, parseInt(e.target.value) || 1)))}
                className="w-full bg-white dark:bg-gray-800"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1.5">
                Maximum: {maxImages} image{maxImages > 1 ? 's' : ''} per generation
              </p>
            </div>
          </div>

          {/* Parameters Section */}
          {selectedModel && (
            <>
              <Separator className="bg-gray-200 dark:bg-gray-700" />
              <div className="space-y-4">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">Model Parameters</h3>
                {renderParameterFields()}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Fixed Generate Button at bottom */}
      <div className="flex-shrink-0 border-t border-gray-200 dark:border-gray-800 p-6 bg-gray-50 dark:bg-gray-950">
        <Button
          className="w-full font-medium h-11 shadow-sm bg-gradient-to-r from-[var(--color-1)] via-[var(--color-3)] to-[var(--color-5)] hover:opacity-90 transition-opacity"
          onClick={onGenerate}
          disabled={!selectedModel || isGenerating || conddiffPairUnavailable}
        >
          {isGenerating ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>Generate {numImages} Image{numImages > 1 ? 's' : ''}</>
          )}
        </Button>
      </div>
    </div>
  )
}
