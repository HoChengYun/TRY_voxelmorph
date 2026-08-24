"""
把 FreeSurfer 產生的 atlas aseg 切到訓練用的 atlas 空間，並驗證。

輸入
    ASD/atlas_out/atlas_aseg.nii.gz     (256,256,256) int
        來自對 mni152_09c_t1_padded256.nii.gz 跑 recon-all -all，
        再以 mri_convert -rl <padded256> -rt nearest 轉回。

切片鏈（兩段都是整數切片，零內插）
    (256,256,256) --[31:224, 13:242, 31:224]--> (193,229,193)   去掉補零
                  --[0:192,   2:226,  0:192 ]--> (192,224,192)   同 atlas_v3 的裁切

輸出
    IXI/atlas_mni152_09c_v3_seg.npz     key: seg (int16)
    ASD/atlas_out/atlas_seg_check.png   目視用疊圖

用法
    python ASD\\make_atlas_seg.py
"""

import os
import sys
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'ASD', 'atlas_out', 'atlas_aseg.nii.gz')
ATLAS_NPZ = os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz')
LABELS = os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz')
DST = os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz')
PNG = os.path.join(ROOT, 'ASD', 'atlas_out', 'atlas_seg_check.png')

PAD_SLICE = (slice(31, 224), slice(13, 242), slice(31, 224))   # 去補零
CROP_SLICE = (slice(0, 192), slice(2, 226), slice(0, 192))     # atlas_v3 的裁切

BAR = '=' * 74
print()
print(BAR)
print('  atlas aseg -> 訓練用 atlas 空間')
print(BAR)

img = nib.load(SRC)
raw = np.asarray(img.dataobj)
print('  來源 : %s' % SRC)
print('         shape=%s dtype=%s' % (raw.shape, raw.dtype))

if raw.shape != (256, 256, 256):
    sys.exit('[X] 預期 (256,256,256)，實際 %s' % (raw.shape,))

# 標籤必須是精確整數
frac = float(np.abs(raw - np.round(raw)).max())
if frac > 0:
    sys.exit('[X] 標籤有非整數值（最大偏差 %g）—— 轉檔時內插法用錯了' % frac)
seg256 = np.round(raw).astype(np.int16)
print('         標籤 %d 種，最大非整數偏差 %g' % (len(np.unique(seg256)), frac))

results = []


def check(name, ok, detail):
    results.append((name, ok))
    print('  [%s] %s' % ('v' if ok else 'X', name))
    print('      %s' % detail)


# ── 切片鏈 ───────────────────────────────────────────────────────────
print()
print('  切片鏈：')
seg193 = seg256[PAD_SLICE]
print('    (256,256,256) --[31:224, 13:242, 31:224]--> %s' % (seg193.shape,))
seg = seg193[CROP_SLICE]
print('    %s --[0:192, 2:226, 0:192]--> %s' % (seg193.shape, seg.shape))

vol = np.load(ATLAS_NPZ)['vol']
print()
check('shape 與 atlas_v3 一致', seg.shape == vol.shape,
      'seg %s  vol %s' % (seg.shape, vol.shape))

# ── 切掉的部分不該有標籤 ─────────────────────────────────────────────
m = np.ones(seg256.shape, dtype=bool)
m[PAD_SLICE] = False
lost_pad = int((seg256[m] != 0).sum())
m2 = np.ones(seg193.shape, dtype=bool)
m2[CROP_SLICE] = False
lost_crop = int((seg193[m2] != 0).sum())
check('切掉的區域沒有標籤（沒切到腦）', lost_pad == 0 and lost_crop == 0,
      '補零區 %d 個、atlas 裁切區 %d 個非零標籤' % (lost_pad, lost_crop))

# ── 評估用的 30 個結構要在 ───────────────────────────────────────────
want = np.load(LABELS)['labels'].astype(int).tolist()
have = set(int(x) for x in np.unique(seg))
miss = sorted(set(want) - have)
check('labels.npz 的 30 個結構都在', not miss,
      '有 %d 種標籤；缺少 %s' % (len(have), miss if miss else '無'))

# ── 標籤要落在腦內 ───────────────────────────────────────────────────
fg_seg, fg_vol = seg > 0, vol > 0
inter = int((fg_seg & fg_vol).sum())
r = inter / max(int(fg_seg.sum()), 1)
check('標籤幾乎全落在 atlas 影像的前景內', r > 0.97,
      '有標籤的 voxel 中 %.4f 落在 vol>0' % r)

# ── 左右定位（抓翻轉）────────────────────────────────────────────────
LR = [(2, 41, 'Cerebral-WM'), (3, 42, 'Cortex'), (4, 43, 'Lateral-Vent'),
      (10, 49, 'Thalamus'), (17, 53, 'Hippocampus'), (11, 50, 'Caudate'),
      (12, 51, 'Putamen')]


def centroid(mask):
    i = np.argwhere(mask)
    return i.mean(axis=0) if len(i) else np.array([np.nan] * 3)


print()
print('  ── 左右結構質心（軸 0 = 左右）──')
deltas = []
for lid, rid, nm in LR:
    if lid not in have or rid not in have:
        continue
    cl, cr2 = centroid(seg == lid), centroid(seg == rid)
    d = cl[0] - cr2[0]
    deltas.append(d)
    print('    %-16s L(%2d) x=%6.1f   R(%2d) x=%6.1f   Δ=%+7.1f'
          % (nm, lid, cl[0], rid, cr2[0], d))
same = len(deltas) >= 3 and (all(x > 0 for x in deltas) or all(x < 0 for x in deltas))
check('左右配對方向一致（未翻轉）', same, '%d 組配對，Δ 全部同號 = %s' % (len(deltas), same))

# ── 存檔 ─────────────────────────────────────────────────────────────
n_ok = sum(1 for _, o in results if o)
if n_ok == len(results):
    np.savez_compressed(DST, seg=seg)
    print()
    print('  [v] 已存 %s' % DST)
    print('      key=seg  shape=%s  dtype=%s' % (seg.shape, seg.dtype))
else:
    print()
    print('  [X] 有檢查未通過，不存檔')

# ── 疊圖 ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rng = np.random.default_rng(0)
    lut = np.vstack([[0, 0, 0], rng.random((int(seg.max()) + 1, 3)) * 0.7 + 0.3])
    cmap = ListedColormap(lut)
    D, H, W = vol.shape
    fig, ax = plt.subplots(2, 3, figsize=(13, 9))
    for i, (v, s, t) in enumerate([
            (vol[D // 2], seg[D // 2], 'Sagittal'),
            (vol[:, H // 2], seg[:, H // 2], 'Coronal'),
            (vol[:, :, W // 2], seg[:, :, W // 2], 'Axial')]):
        ax[0][i].imshow(v.T, cmap='gray', origin='lower')
        ax[0][i].set_title('atlas_v3 — %s' % t, fontsize=10)
        ax[1][i].imshow(v.T, cmap='gray', origin='lower')
        ax[1][i].imshow(np.ma.masked_where(s.T == 0, s.T), cmap=cmap,
                        origin='lower', alpha=0.55, interpolation='nearest')
        ax[1][i].set_title('+ aseg', fontsize=10)
        for r_ in range(2):
            ax[r_][i].axis('off')
    fig.suptitle('MNI152 atlas + FreeSurfer aseg   %s   %d labels'
                 % (str(seg.shape), len(have)), fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PNG, dpi=130, bbox_inches='tight')
    plt.close()
    print('  [v] 疊圖：%s' % PNG)
except Exception as e:
    print('  [!] 疊圖失敗：%s' % e)

print()
print(BAR)
print('  結果：%d/%d 通過' % (n_ok, len(results)))
for nm, o in results:
    if not o:
        print('    失敗：%s' % nm)
print(BAR)
sys.exit(0 if n_ok == len(results) else 1)
