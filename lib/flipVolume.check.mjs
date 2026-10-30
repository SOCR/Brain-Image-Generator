// Self-check for flipVoxels: node lib/flipVolume.check.mjs (Node 23.6+ runs the .ts import directly;
// the ExperimentalWarning it prints is harmless)
import assert from 'node:assert/strict'
import { flipVoxels } from './flipVolume.ts'

const dims = [3, 4, 5] // deliberately unequal, so a mixed-up stride cannot pass by luck
const [nx, ny, nz] = dims
const frames = 2
const at = (x, y, z, t) => x + y * nx + z * nx * ny + t * nx * ny * nz
const original = Float32Array.from({ length: nx * ny * nz * frames }, (_, i) => i)

for (const axis of [0, 1, 2]) {
  const v = original.slice()
  flipVoxels(v, dims, axis)
  for (let t = 0; t < frames; t++) {
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const mirrored = [x, y, z]
          mirrored[axis] = dims[axis] - 1 - mirrored[axis]
          assert.equal(v[at(x, y, z, t)], original[at(...mirrored, t)], `axis ${axis} voxel ${x},${y},${z},${t}`)
        }
      }
    }
  }
  flipVoxels(v, dims, axis)
  assert.deepEqual(v, original, `axis ${axis}: flipping twice must restore the volume`)
}

// Even-length axes have no middle slice to leave in place.
const even = Uint8Array.from([1, 2, 3, 4])
flipVoxels(even, [4, 1, 1], 0)
assert.deepEqual(even, Uint8Array.from([4, 3, 2, 1]))

console.log('flipVolume check passed')
