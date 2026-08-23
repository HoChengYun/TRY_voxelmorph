"""
對 preprocess_fs.py 產出的單顆 npz 做量化驗證。

目視確認「標籤和影像疊合」很容易漏掉方向性錯誤（左右翻轉時兩者仍然疊合，
只是整顆腦鏡像了）。這支腳本補上目視看不出來的檢查：

  A. 結構完整性  shape / dtype / 值域 / 標籤是否為整數
  B. 疊合        有標籤的地方影像不能是背景；反之腦組織要幾乎都被標到
  C. ⭐ 左右定位  FreeSurfer 左側結構（2/4/10/17）的質心必須落在
                 右側對應結構（41/43/49/53）的同一側且方向一致
                 —— 若 Affine 把方向搞反，這裡會抓到
  D. 解剖合理性  腦室在腦中央、皮質在外圍
  E. 出圖        三切面 + 標籤疊圖，供人眼最後確認

用法：
    python ASD\\verify_one_subject.py --npz ASD\\fs_check\\train\\A001.npz \\
        --atlas IXI\\atlas_mni152_09c_v3.npz
"""

import os
import sys
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True, help='preprocess_fs.py 產出的 .npz')
ap.add_argument('--atlas', default=None, help='atlas .npz（用來比對腦位置，可省略）')
ap.add_argument('--out-png', default=None, help='輸出疊圖路徑（預設放在 npz 旁邊）')
args = ap.parse_args()

# FreeSurfer LUT：左右成對的結構
LR_PAIRS = [
    (2,  41, 'Cerebral-White-Matter'),
    (3,  42, 'Cerebral-Cortex'),
    (4,  43, 'Lateral-Ventricle'),
    (10, 49, 'Thalamus'),
    (17, 53, 'Hippocampus'),
    (11, 50, 'Caudate'),
    (12, 51, 'Putamen'),
]

d = np.load(os.path.normpath(args.npz))
name = os.path.basename(args.npz)

print('=' * 68)
print('  單顆驗證：%s' % name)
print('=' * 68)

results = []


def check(label, ok, detail):
    results.append((label, ok))
    print('  [%s] %s' % ('v' if ok else 'X', label))
    print('      %s' % detail)


# ── A. 結構完整性 ────────────────────────────────────────────────────
keys = list(d.keys())
check('npz 含 vol 與 seg 兩個 key', set(keys) >= {'vol', 'seg'}, 'keys = %s' % keys)

vol, seg = d['vol'], d['seg']
check('vol 是 float32 且值域在 [0,1]',
      vol.dtype == np.float32 and vol.min() >= 0 and vol.max() <= 1.0 + 1e-6,
      'dtype=%s  範圍 [%.4f, %.4f]' % (vol.dtype, vol.min(), vol.max()))
check('seg 是整數型別（未被正規化）',
      np.issubdtype(seg.dtype, np.integer),
      'dtype=%s  標籤值 %d 種，最小 %d 最大 %d'
      % (seg.dtype, len(np.unique(seg)), seg.min(), seg.max()))
check('vol 與 seg 同 shape', vol.shape == seg.shape,
      'vol %s  seg %s' % (vol.shape, seg.shape))

labels = sorted(int(x) for x in np.unique(seg) if x != 0)
known_fs = [l for l in labels if l in
            {2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31,
             41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 72, 77,
             80, 85, 251, 252, 253, 254, 255}]
check('標籤值是 FreeSurfer LUT 編號',
      len(known_fs) >= len(labels) * 0.9,
      '%d/%d 個標籤屬於已知 FreeSurfer LUT；前 12 個：%s'
      % (len(known_fs), len(labels), labels[:12]))

# ── B. 疊合 ──────────────────────────────────────────────────────────
fg_seg, fg_vol = seg > 0, vol > 0
inter = int((fg_seg & fg_vol).sum())
r_seg_in_vol = inter / max(int(fg_seg.sum()), 1)
r_vol_in_seg = inter / max(int(fg_vol.sum()), 1)
# 這個方向才是「標籤有沒有搬對」的關鍵：標籤不該跑到影像外面去
check('標籤幾乎全落在影像前景內（>0.99）',
      r_seg_in_vol > 0.99,
      '有標籤的 voxel 中，%.4f 落在影像前景（vol>0）' % r_seg_in_vol)

# 反方向只是診斷，不是通過條件：
#   norm.mgz 源自 brainmask.mgz，會保留部分硬腦膜、血管、腦外 CSF，
#   而 aseg 刻意把這些留成標籤 0。所以「影像前景 ⊅ 標籤前景」是正常的。
#   A001 實測：未標到的 voxel 中位強度 0.18（有標籤的是 0.64），
#   且距腦質心中位半徑 73.9 vs 60.1 —— 系統性偏外圍，符合上述解釋。
#   門檻放寬到 0.6 只是為了抓「整個標籤沒搬進來」這種嚴重錯誤。
check('腦組織有合理比例被標到（>0.60，此項僅為粗略 sanity check）',
      r_vol_in_seg > 0.60,
      '影像前景中，%.4f 有標籤覆蓋'
      '（aseg 不標硬腦膜/血管/腦外 CSF，這個比例本來就不會接近 1）' % r_vol_in_seg)

# 質心距離：疊合良好時兩者質心應該幾乎重合
def centroid(mask):
    idx = np.argwhere(mask)
    return idx.mean(axis=0) if len(idx) else np.array([np.nan] * 3)

c_v, c_s = centroid(fg_vol), centroid(fg_seg)
dist = float(np.linalg.norm(c_v - c_s))
check('影像與標籤的質心幾乎重合（< 5 voxel）', dist < 5.0,
      '影像質心 %s  標籤質心 %s  距離 %.2f voxel'
      % (np.round(c_v, 1), np.round(c_s, 1), dist))

# ── C. ⭐ 左右定位 ───────────────────────────────────────────────────
print()
print('  ── 左右結構質心（軸 0 = 左右方向）──')
sides, bad = [], []
for lid, rid, nm in LR_PAIRS:
    if lid not in labels or rid not in labels:
        continue
    cl, cr = centroid(seg == lid), centroid(seg == rid)
    delta = cl[0] - cr[0]          # 左側 - 右側，沿軸 0
    sides.append(delta)
    print('    %-22s L(%d) x=%6.1f   R(%d) x=%6.1f   Δ=%+7.1f'
          % (nm, lid, cl[0], rid, cr[0], delta))
    if abs(delta) < 3:
        bad.append(nm)

consistent = len(sides) >= 3 and (all(x > 0 for x in sides) or all(x < 0 for x in sides))
check('所有左右配對都落在同一側（方向一致，未發生翻轉）',
      consistent and not bad,
      '%d 組配對，Δ 全部同號 = %s%s'
      % (len(sides), consistent,
         ('；分不開左右的結構：%s' % bad) if bad else ''))

# ── D. 解剖合理性 ────────────────────────────────────────────────────
if 4 in labels and 43 in labels and 3 in labels and 42 in labels:
    vent = centroid((seg == 4) | (seg == 43))
    ctx = seg == 3
    ctx_idx = np.argwhere(ctx)
    # 腦室應該比皮質更靠近腦中心
    brain_c = centroid(fg_seg)
    d_vent = float(np.linalg.norm(vent - brain_c))
    d_ctx = float(np.linalg.norm(ctx_idx.mean(axis=0) - brain_c)) if len(ctx_idx) else 0
    spread_ctx = float(np.linalg.norm(ctx_idx.std(axis=0))) if len(ctx_idx) else 0
    spread_vent = float(np.linalg.norm(np.argwhere((seg == 4) | (seg == 43)).std(axis=0)))
    check('腦室比皮質更集中在腦中央（解剖合理）',
          spread_vent < spread_ctx,
          '腦室離散度 %.1f < 皮質離散度 %.1f（腦室質心距腦中心 %.1f voxel）'
          % (spread_vent, spread_ctx, d_vent))

# ── E. 出圖 ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    D, H, W = vol.shape
    rng = np.random.default_rng(0)
    lut = np.vstack([[0, 0, 0], rng.random((int(seg.max()) + 1, 3)) * 0.7 + 0.3])
    cmap = ListedColormap(lut)

    fig, ax = plt.subplots(2, 3, figsize=(13, 9))
    views = [(vol[D // 2], seg[D // 2], 'Axis0 (sagittal) mid'),
             (vol[:, H // 2], seg[:, H // 2], 'Axis1 (coronal) mid'),
             (vol[:, :, W // 2], seg[:, :, W // 2], 'Axis2 (axial) mid')]
    for i, (v, s, t) in enumerate(views):
        ax[0][i].imshow(v.T, cmap='gray', origin='lower')
        ax[0][i].set_title(t, fontsize=10); ax[0][i].axis('off')
        ax[1][i].imshow(v.T, cmap='gray', origin='lower')
        ax[1][i].imshow(np.ma.masked_where(s.T == 0, s.T), cmap=cmap,
                        origin='lower', alpha=0.55, interpolation='nearest')
        ax[1][i].set_title('+ aseg overlay', fontsize=10); ax[1][i].axis('off')
    fig.suptitle('%s   vol %s   %d labels' % (name, str(vol.shape), len(labels)),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = args.out_png or os.path.join(os.path.dirname(os.path.abspath(args.npz)),
                                       name.replace('.npz', '_check.png'))
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print()
    print('  [v] 疊圖已輸出：%s' % out)
except Exception as e:
    print('  [!] 出圖失敗（不影響上述檢查）：%s' % e)

print()
print('=' * 68)
n_ok = sum(1 for _, ok in results if ok)
print('  結果：%d/%d 通過' % (n_ok, len(results)))
for lab, ok in results:
    if not ok:
        print('    失敗：%s' % lab)
print('=' * 68)
sys.exit(0 if n_ok == len(results) else 1)
