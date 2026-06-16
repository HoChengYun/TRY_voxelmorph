# -*- coding: utf-8 -*-
"""
MNI152 2009c .nii -> atlas npz + nii.gz

Usage:
    python IXI\make_atlas.py
        --t1   mni_icbm152_t1_tal_nlin_asym_09c.nii
        --mask mni_icbm152_t1_tal_nlin_asym_09c_mask.nii
        --target-shape 192,224,192
        --method crop      <- crop/pad keeps spacing=1mm; resample=old behaviour

Output:
    atlas_mni152_09c.npz     <- for VoxelMorph train.py
    atlas_mni152_09c.nii.gz  <- for preprocess_ixi.py (with header)
"""

import os
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--t1',           required=True, help='T1 .nii path')
parser.add_argument('--mask',         required=True, help='brain mask .nii path')
parser.add_argument('--out',          default=os.path.join(_HERE, 'atlas_mni152_09c'),
                    help='output base path (no extension)')
parser.add_argument('--target-shape', default=None,
                    help='target shape e.g. 192,224,192 (must be divisible by 16)')
parser.add_argument('--method',       default='crop',
                    choices=['crop', 'pad', 'resample'],
                    help='crop: trim background (spacing stays 1mm); '
                         'pad: zero-pad (spacing stays 1mm); '
                         'resample: scale (old behaviour, spacing changes)')
args = parser.parse_args()

import ants

print('Loading T1  :', args.t1)
print('Loading mask:', args.mask)

t1   = ants.image_read(args.t1)
mask = ants.image_read(args.mask)

print('Original shape  :', t1.shape)
print('Original spacing:', t1.spacing)
print('Original origin :', t1.origin)

# skull strip
brain = ants.mask_image(t1, mask)

# normalise to [0, 1]
arr = brain.numpy().astype(np.float32)
p1, p99 = np.percentile(arr[arr > 0], [1, 99])
arr = np.clip(arr, p1, p99)
arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
arr = arr.astype(np.float32)

brain_norm = brain.new_image_like(arr)
print('After skull-strip + normalise shape:', brain_norm.shape)

# adjust shape
if args.target_shape is not None:
    target = tuple(int(x) for x in args.target_shape.split(','))
    src    = brain_norm.shape
    print(f'Adjusting shape: {src} -> {target}  method={args.method}')

    if args.method == 'resample':
        print('  WARNING: resample changes spacing (not 1mm isotropic)')
        brain_final = ants.resample_image(
            brain_norm, target, use_voxels=True, interp_type=1)

    else:
        # crop or pad -- spacing stays exactly as-is (1mm)
        arr2    = brain_norm.numpy().astype(np.float32)
        origin  = list(brain_norm.origin)
        spacing = brain_norm.spacing

        new_origin = list(origin)

        for ax in range(3):
            s, t = arr2.shape[ax], target[ax]
            if s == t:
                continue

            elif s > t:
                # crop: split evenly, extra goes to back
                diff        = s - t
                crop_front  = diff // 2
                crop_back   = diff - crop_front
                new_origin[ax] = origin[ax] + crop_front * spacing[ax]
                slc = [slice(None)] * 3
                slc[ax] = slice(crop_front, s - crop_back)
                arr2 = arr2[tuple(slc)]
                print(f'  Axis {ax}: crop {crop_front} front + {crop_back} back'
                      f'  origin {origin[ax]:.3f} -> {new_origin[ax]:.3f}')

            else:
                # pad: split evenly, extra goes to back
                diff       = t - s
                pad_front  = diff // 2
                pad_back   = diff - pad_front
                new_origin[ax] = origin[ax] - pad_front * spacing[ax]
                pw = [(0, 0)] * 3
                pw[ax] = (pad_front, pad_back)
                arr2 = np.pad(arr2, pw, mode='constant', constant_values=0)
                print(f'  Axis {ax}: pad {pad_front} front + {pad_back} back'
                      f'  origin {origin[ax]:.3f} -> {new_origin[ax]:.3f}')

        brain_final = ants.from_numpy(
            arr2,
            origin    = tuple(new_origin),
            spacing   = spacing,
            direction = brain_norm.direction,
        )
        print(f'  spacing kept: {brain_final.spacing}  (1mm isotropic)')

else:
    brain_final = brain_norm

final_arr = brain_final.numpy().astype(np.float32)

print('Final shape  :', brain_final.shape)
print('Final spacing:', brain_final.spacing)
print('Final origin :', brain_final.origin)
print('min/max      :', final_arr.min(), '/', final_arr.max())

# strip extension from --out
out_base = args.out
for ext in ['.npz', '.nii.gz', '.nii']:
    if out_base.endswith(ext):
        out_base = out_base[:-len(ext)]
        break

# save .nii.gz (with header, for preprocess_ixi.py ANTs registration)
nii_path = out_base + '.nii.gz'
ants.image_write(brain_final, nii_path)
print('\nSaved nii.gz:', nii_path, ' (with header, for --atlas in preprocess_ixi.py)')

# save .npz (array only, for train.py)
npz_path = out_base + '.npz'
np.savez_compressed(npz_path, vol=final_arr)
print('Saved npz   :', npz_path, ' (for --atlas in train.py)')

print()
print('Next steps:')
print(f'  python IXI\\preprocess_ixi.py --atlas {nii_path}')
print(f'  python voxelmorph-code\\scripts\\torch\\train.py --atlas {npz_path} ...')
