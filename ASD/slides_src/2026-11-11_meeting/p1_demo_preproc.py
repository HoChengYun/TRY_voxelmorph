# -*- coding: utf-8 -*-
"""Part 1 示範：三路去頭殼影像走一次 v3 前處理（照 ASD\\preprocess_fs.py 的步驟，加 --n4），存每一步的中間結果給流程圖用。
→ models\\deck_charts\\1111_p1_demo_<受試者>_<路>.npz（n4：N4 後（原格子）、reg：affine 後（atlas 格子）、vol：正規化後、seg：aseg 搬過去）

步驟（同 preprocess_fs.py 第 516～582 行）：讀 orig × 遮罩 → ants.n4_bias_field_correction → ants.registration(Affine) 對到
IXI\\atlas_mni152_09c_v3.nii.gz → aseg 用同一個變換、nearestNeighbor → 腦內 1～99 percentile 截斷、縮放到 [0, 1]。
只讀示範檔，不寫 data\\。ANTs 配準有隨機取樣，重跑結果會有極小差異（同 preprocess_fs.py 的說明）。
"""
import os
import sys

import numpy as np
import ants

from style import J, OUT

SUBJ = sys.argv[1:] or ['VNT045']
ARMS = {'freesurfer': 'freesurfer_strip', 'tigerbx': 'tigerbx_strip', 'synthstrip': 'synthstrip_strip'}
atlas = ants.image_read(J('IXI', 'atlas_mni152_09c_v3.nii.gz'))
for s in SUBJ:
    D = J('data', 'example_for_meeting', s)
    seg = ants.image_read(os.path.join(D, 'labels', 'aseg.nii.gz'))
    for arm, fn in ARMS.items():
        dst = os.path.join(OUT, '1111_p1_demo_%s_%s.npz' % (s, arm))
        if os.path.exists(dst):
            print('已有', dst)
            continue
        img = ants.image_read(os.path.join(D, fn + '.nii.gz'))
        n4 = ants.n4_bias_field_correction(img)
        reg = ants.registration(fixed=atlas, moving=n4, type_of_transform='Affine', verbose=False)
        r = reg['warpedmovout'].numpy().astype(np.float32)
        sr = ants.apply_transforms(fixed=atlas, moving=seg, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor').numpy()
        assert float(np.abs(sr - np.round(sr)).max()) < 1e-6
        pos = r[r > 0]
        p1, p99 = np.percentile(pos, [1, 99])
        v = np.clip(r, p1, p99)
        v = ((v - v.min()) / (v.max() - v.min() + 1e-8)).astype(np.float32)
        np.savez_compressed(dst, n4=n4.numpy().astype(np.float32), reg=r, vol=v, seg=np.round(sr).astype(np.int16))
        print('[OK]', s, arm, r.shape)
