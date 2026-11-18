# -*- coding: utf-8 -*-
"""atlas 的 aparc+aseg（或其他 FreeSurfer 標籤）-> 訓練用 atlas 空間（v3 那批，2026-11-17）

來源：data/skullstrip_compare/atlas/<標籤>/mni152_09c.nii.gz
      （FreeSurfer 端對 mni152_09c_t1_padded256 跑 recon-all 的結果，recon-all 的格子：256³ LIA）

兩步，都是零內插：
  1. recon-all 格子（LIA）-> 補零 atlas 的格子（IXI/mni152_09c_t1_padded256.nii.gz，RAS）
     README 寫的是 `mri_convert -rl <padded256> -rt nearest`；筆電沒有 FreeSurfer，
     但兩個格子之間只差軸對調／翻轉＋整數位移（下面會檢查），直接搬陣列結果一樣。
  2. 256³ -> (192,224,192)：跟 make_atlas_seg.py 同一組整數切片。
     （不直接用 make_atlas_seg.py：它檢查「labels.npz 的 30 個結構都在」，aparc+aseg
      的皮質是 68 區、沒有 3／42，會被擋下來不存檔。那支程式不改。）

檢查（全部通過才存檔）：
  a. 兩個格子的關係是純軸對調／翻轉＋整數位移
  b. aseg 用同樣方法搬完，跟現有的 ASD/atlas_out/atlas_aseg.nii.gz 逐體素相同
  c. 每個標籤的體素數，搬之前、搬之後一樣（沒掉東西）
  d. 切掉的區域沒有標籤
  e. aparc+aseg 的皮質（1000～1999 -> 3、2000～2999 -> 42）併回去之後，
     跟現在用的 IXI/atlas_mni152_09c_v3_seg.npz 逐體素相同（只有 --labels 含 aparc+aseg 時）

輸出
  ASD/atlas_out/atlas_<標籤>.nii.gz            256³、補零 atlas 的格子
  IXI/atlas_mni152_09c_v3_seg_<標籤>.npz       key: seg（int16），給 test_dice.py --atlas-seg

用法
  python ASD\\make_atlas_aparc.py                       （預設只做 aparc+aseg）
  python ASD\\make_atlas_aparc.py --labels aparc+aseg wmparc
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = r'D:\MyHome\MRI\Cdisk_voxelmorph'   # 資料都放 D 槽（11-18 起，筆電）
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = os.path.join(ROOT, 'data')      # 其他機器（AI）：repo 底下的 data\
ap = argparse.ArgumentParser()
ap.add_argument('--src-root', default=os.path.join(DATA_ROOT, 'skullstrip_compare', 'atlas'))
ap.add_argument('--labels', nargs='+', default=['aparc+aseg'])
ap.add_argument('--ref', default=os.path.join(ROOT, 'IXI', 'mni152_09c_t1_padded256.nii.gz'))
args = ap.parse_args()

OLD_ASEG256 = os.path.join(ROOT, 'ASD', 'atlas_out', 'atlas_aseg.nii.gz')
OLD_SEG_V3 = os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz')
PAD_SLICE = (slice(31, 224), slice(13, 242), slice(31, 224))   # 同 make_atlas_seg.py
CROP_SLICE = (slice(0, 192), slice(2, 226), slice(0, 192))     # 同 make_atlas_seg.py

ref = nib.load(args.ref)
fails = []


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('v' if ok else 'X', name, ('  —— ' + detail) if detail else ''))
    if not ok:
        fails.append(name)


def to_ref_grid(img):
    """recon-all 格子 -> 補零 atlas 格子，最近鄰（實際上是純搬陣列）"""
    M = np.linalg.inv(img.affine) @ ref.affine          # ref 的 voxel -> 來源的 voxel
    R, t = M[:3, :3], M[:3, 3]
    pure = np.allclose(R, np.round(R)) and np.allclose(t, np.round(t), atol=1e-3)
    d = np.asanyarray(img.dataobj)
    frac = float(np.abs(d - np.round(d)).max())
    d = np.round(d).astype(np.int16)
    i, j, k = np.meshgrid(*[np.arange(n) for n in ref.shape], indexing='ij')
    p = np.stack([i, j, k, np.ones_like(i)], 0).reshape(4, -1)
    q = np.rint(M @ p)[:3].astype(np.int64)
    inside = np.all((q >= 0) & (q < np.array(d.shape)[:, None]), 0)
    out = np.zeros(int(np.prod(ref.shape)), np.int16)
    out[inside] = d[q[0, inside], q[1, inside], q[2, inside]]
    return d, out.reshape(ref.shape), pure, frac


def same_counts(a, b):
    ua, na = np.unique(a, return_counts=True)
    ub, nb = np.unique(b, return_counts=True)
    return np.array_equal(ua, ub) and np.array_equal(na, nb), len(ua), len(ub)


print('\n== 先拿 aseg 驗證搬法 ==')
src, a256, pure, frac = to_ref_grid(nib.load(os.path.join(args.src_root, 'aseg', 'mni152_09c.nii.gz')))
check('a. 兩個格子只差軸對調／翻轉＋整數位移', pure)
check('   標籤是整數', frac == 0, '最大偏差 %g' % frac)
old = np.round(np.asanyarray(nib.load(OLD_ASEG256).dataobj)).astype(np.int16)
check('b. aseg 搬完跟現有的 atlas_out/atlas_aseg.nii.gz 逐體素相同',
      np.array_equal(a256, old), '不同 %d 個' % int((a256 != old).sum()))

for lab in args.labels:
    print('\n== %s ==' % lab)
    src, s256, pure, frac = to_ref_grid(nib.load(os.path.join(args.src_root, lab, 'mni152_09c.nii.gz')))
    n_fail0 = len(fails)
    check('a. 純軸對調／翻轉＋整數位移', pure)
    check('   標籤是整數', frac == 0, '最大偏差 %g' % frac)
    ok, n0, n1 = same_counts(src, s256)
    check('c. 每個標籤的體素數搬前搬後一樣', ok, '%d 種 -> %d 種' % (n0, n1))
    m = np.ones(s256.shape, bool)
    m[PAD_SLICE] = False
    s193 = s256[PAD_SLICE]
    m2 = np.ones(s193.shape, bool)
    m2[CROP_SLICE] = False
    seg = s193[CROP_SLICE]
    check('d. 切掉的區域沒有標籤', int((s256[m] != 0).sum()) == 0 and int((s193[m2] != 0).sum()) == 0)
    if lab == 'aparc+aseg':
        merged = seg.copy()
        merged[(seg >= 1000) & (seg < 2000)] = 3
        merged[(seg >= 2000) & (seg < 3000)] = 42
        v3 = np.load(OLD_SEG_V3)['seg']
        check('e. 皮質併回 3／42 後，跟現在的 atlas_mni152_09c_v3_seg.npz 逐體素相同',
              np.array_equal(merged, v3), '不同 %d 個' % int((merged != v3).sum()))
        ctx = np.unique(seg[seg >= 1000])
        print('      皮質分區 %d 個（左 %d、右 %d）'
              % (len(ctx), int((ctx < 2000).sum()), int((ctx >= 2000).sum())))
    if len(fails) > n_fail0 or fails:
        print('  [X] 有檢查沒過，%s 不存檔' % lab)
        continue
    out256 = os.path.join(ROOT, 'ASD', 'atlas_out', 'atlas_%s.nii.gz' % lab)
    nib.save(nib.Nifti1Image(s256, ref.affine), out256)
    out_npz = os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg_%s.npz' % lab)
    np.savez_compressed(out_npz, seg=seg)
    print('  [v] %s' % os.path.relpath(out256, ROOT))
    print('  [v] %s   seg %s int16，%d 種標籤' % (os.path.relpath(out_npz, ROOT), seg.shape, len(np.unique(seg))))

print('\n結果：%s' % ('全部通過' if not fails else '沒過：%s' % fails))
sys.exit(1 if fails else 0)
