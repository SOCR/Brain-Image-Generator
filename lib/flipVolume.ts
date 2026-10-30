/**
 * Mirror a volume in place along one voxel axis (0, 1 or 2). Doing it twice restores the volume.
 * Walking the whole array also covers 4D files, since every frame repeats the same x/y/z layout.
 */
export function flipVoxels(
  img: { [i: number]: number; length: number },
  [nx, ny, nz]: number[],
  axis: number,
) {
  const stride = [1, nx, nx * ny][axis]
  const n = [nx, ny, nz][axis]
  for (let a = 0; a < img.length; a++) {
    const i = Math.floor(a / stride) % n
    if (i >= n - 1 - i) continue // each pair is swapped once, from its near end; the middle stays
    const b = a + (n - 1 - 2 * i) * stride
    const tmp = img[a]
    img[a] = img[b]
    img[b] = tmp
  }
}
