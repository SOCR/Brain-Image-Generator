// Self-check for flipVoxels and saveFlippedVolume: node lib/flipVolume.check.mjs
// (Node 23.6+ runs the .ts imports directly; the ExperimentalWarning it prints is harmless)
import assert from 'node:assert/strict'
import { flipVoxels } from './flipVolume.ts'
import { saveFlippedVolume } from './download.ts'

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

// saveFlippedVolume, against a small stand-in for the Supabase client.
const BASE = 'https://x.supabase.co/storage/v1/object/public/volumes'
const fakeSupabase = (rows, uploadError = null) => {
  const log = { uploads: [], inserts: [] }
  const query = () => {
    let matches = rows
    const q = {
      select: () => q,
      eq: (column, value) => { matches = matches.filter(r => r[column] === value); return q },
      limit: () => q,
      maybeSingle: async () => ({ data: matches[0] ?? null, error: null }),
      insert: async (row) => { log.inserts.push(row); rows.push(row); return { error: null } },
    }
    return q
  }
  const volumes = {
    upload: async (path) => { log.uploads.push(path); return { error: uploadError } },
    getPublicUrl: (path) => ({ data: { publicUrl: `${BASE}/${path}` } }),
  }
  return { log, from: query, storage: { from: () => volumes } }
}
const sourceUrl = `${BASE}/u/p/mri64_x/mri3D.nii.gz?`
const sourceRow = () => ({
  name: '3D Volume - mri64_x', project_id: 'p', model_id: 'm', user_id: 'u', file_path: sourceUrl,
  parameters_used: JSON.stringify({ model_name: 'gan3d', orientation: { margin: 0.05 } }),
})
const save = (client, url = sourceUrl) =>
  saveFlippedVolume(client, url, 'mri3D_flipped-AP.nii.gz', 'AP', new Uint8Array(4))

// A first save stores the file beside the original and adds one row; the original row is kept.
const rows = [sourceRow()]
const first = fakeSupabase(rows)
assert.equal(await save(first), true)
assert.deepEqual(first.log.uploads, ['u/p/mri64_x/mri3D_flipped-AP.nii.gz'])
assert.equal(rows.length, 2)
assert.equal(rows[0].file_path, sourceUrl, 'the original row must not change')
const copy = first.log.inserts[0]
assert.equal(copy.name, '3D Volume - mri64_x_flipped-AP')
assert.equal(copy.file_path, `${BASE}/u/p/mri64_x/mri3D_flipped-AP.nii.gz`)
assert.deepEqual(copy.parameters_used, {
  model_name: 'gan3d', orientation: { margin: 0.05 }, flipped: 'AP', flipped_from: '3D Volume - mri64_x',
})
assert.ok(!JSON.stringify(copy.parameters_used).includes('supabase'), 'no storage URL in shareable params')

// Saving the same flips again adds nothing.
const again = fakeSupabase(rows, { message: 'The resource already exists' })
assert.equal(await save(again), false)
assert.equal(again.log.inserts.length, 0)

// A file left behind by an interrupted save still gets its row.
const orphan = fakeSupabase([sourceRow()], { message: 'The resource already exists' })
assert.equal(await save(orphan), true)
assert.equal(orphan.log.inserts.length, 1)

// Real upload failures stop the save, and anything not stored or not found is refused.
const denied = fakeSupabase([sourceRow()], new Error('new row violates row-level security policy'))
await assert.rejects(save(denied), /row-level security/)
assert.equal(denied.log.inserts.length, 0)
await assert.rejects(save(fakeSupabase([]), 'https://example.com/mri3D.nii.gz'), /not a stored result/)
await assert.rejects(save(fakeSupabase([])), /not in your Library/)

console.log('flipVolume check passed')
