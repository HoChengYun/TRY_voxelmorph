"""
量測 FreeSurfer conform 來回（原始網格 -> 256^3 -> -rl 轉回）造成的幾何損失。

為什麼要量
----------
recon-all 會把輸入 conform 成 256^3 / 1mm / LIA。若輸入的邊長是奇數
（MNI152 是 193/229/193），中心體素座標是半整數（96.5 / 114.5 / 96.5），
而目標 256 的中心是整數 128.0 —— 差半個體素，會觸發三線性內插。
aseg 之後還要 -rl 轉回，等於來回兩次。

標籤是離散的，小位移在 aseg 上「看不出來」，卻會在 Dice 上偽裝成
「配準效果不好」。所以要拿**影像**（連續值）來量，才量得出位移多少。

⚠️ 量測陷阱（FreeSurfer 端提醒，已納入）
----------------------------------------
conform 會把資料轉成 uchar(0-255) 並**重新縮放強度**。原檔是 float64、
值域 [-0.013, 99.169]。所以：

  * RMS 差異、梯度強度  -> 受縮放影響，**必須先各自正規化到 [0,1] 再比**
  * 質心位移、相關係數  -> 對線性縮放不變，**以這兩項為主要判準**

用法
----
    python ASD\\verify_conform_roundtrip.py --back atlas_orig_backto_mni.nii.gz

    # 參考影像預設依 --back 的 shape 自動選：
    #   (193,229,193) -> 原始 MNI152
    #   (256,256,256) -> 補零版 mni152_09c_t1_padded256.nii.gz
    # 也可以用 --ref 明確指定。
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = os.path.join(ROOT, 'IXI', 'mni_icbm152_nlin_asym_09c_nifti',
                    'mni_icbm152_t1_tal_nlin_asym_09c.nii')
PADDED = os.path.join(ROOT, 'IXI', 'mni152_09c_t1_padded256.nii.gz')

ap = argparse.ArgumentParser()
ap.add_argument('--back', required=True, help='FreeSurfer 用 -rl 轉回來的影像')
ap.add_argument('--ref', default=None, help='參考影像（不給則依 shape 自動選）')
ap.add_argument('--thr', type=float, default=0.20,
                help='產生遮罩的相對閾值（各自最大值的比例，預設 0.20）')
ap.add_argument('--out-png', default=None, help='1D 剖線圖輸出路徑')
args = ap.parse_args()

back_img = nib.load(os.path.normpath(args.back))
back = np.asarray(back_img.dataobj, dtype=np.float64)

if args.ref:
    ref_path = os.path.normpath(args.ref)
elif back.shape == (193, 229, 193):
    ref_path = ORIG
elif back.shape == (256, 256, 256):
    ref_path = PADDED
else:
    sys.exit('[X] 無法依 shape %s 自動選參考影像，請用 --ref 指定' % (back.shape,))

ref_img = nib.load(ref_path)
ref = np.asarray(ref_img.dataobj, dtype=np.float64)

BAR = '=' * 76
print()
print(BAR)
print('  conform 來回幾何損失量測')
print(BAR)
print('  轉回來的 : %s' % args.back)
print('             shape=%s dtype=%s 值域[%.3f, %.3f]'
      % (back.shape, np.asarray(back_img.dataobj).dtype, back.min(), back.max()))
print('  參考影像 : %s' % os.path.basename(ref_path))
print('             shape=%s dtype=%s 值域[%.3f, %.3f]'
      % (ref.shape, np.asarray(ref_img.dataobj).dtype, ref.min(), ref.max()))

if ref.shape != back.shape:
    sys.exit('[X] shape 不一致，無法逐 voxel 比對：%s vs %s' % (ref.shape, back.shape))

verdict = []


def norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-12)




# ── 1. 質心位移（主要判準，對強度縮放不變）────────────────────────
print()
print('  [1] 質心位移　（主要判準：對強度縮放不變）')
# ⚠️ 不能各自產生遮罩再比質心。conform 會把 float 量化成 uchar，
#    改變哪些 voxel 通過門檻 —— 光是這件事就會造出 ~0.06 voxel 的假位移。
#    （實測：各自遮罩 0.062 voxel，共用遮罩只有 0.0032 voxel。）
#    所以用「參考影像的遮罩」套在兩邊，並改用強度加權質心。
mr = norm01(ref) > args.thr
mb = norm01(back) > args.thr      # 只用來報告遮罩大小差異
idx = np.argwhere(mr)


def wcentroid(a):
    w = norm01(a)[mr]
    return (idx * w[:, None]).sum(axis=0) / w.sum()


cr, cb = wcentroid(ref), wcentroid(back)
d = cb - cr
dist = float(np.linalg.norm(d))
print('      共用遮罩   : %d voxel（各自產生的話是 %d vs %d，差異來自 uchar 量化）'
      % (mr.sum(), mr.sum(), mb.sum()))
print('      參考質心   : [%9.4f %9.4f %9.4f]' % tuple(cr))
print('      轉回後質心 : [%9.4f %9.4f %9.4f]' % tuple(cb))
print('      逐軸位移   : [%+9.4f %+9.4f %+9.4f]  合計 %.4f voxel' % (*d, dist))
if dist < 0.05:
    print('      -> ✅ 幾乎沒有位移，conform 在這個輸入上是無損的')
    verdict.append(('質心位移', True, '%.3f voxel' % dist))
elif dist < 0.6:
    print('      -> ⚠️ 有次像素位移（%.3f voxel），標籤邊界會受影響' % dist)
    verdict.append(('質心位移', False, '%.3f voxel（次像素）' % dist))
else:
    print('      -> 🔴 位移過大（%.3f voxel），可能是 header 變換鏈出錯' % dist)
    verdict.append(('質心位移', False, '%.3f voxel（過大）' % dist))

# ── 2. 相關係數（對線性縮放不變）──────────────────────────────────
print()
print('  [2] 相關係數　（對線性縮放不變）')
m = mr | mb
r = float(np.corrcoef(ref[m], back[m])[0, 1])
print('      腦內 voxel 相關係數 : %.6f' % r)
ok = r > 0.999
print('      -> %s' % ('✅ 幾乎完全相同' if ok else '⚠️ 有可見差異'))
verdict.append(('相關係數', ok, '%.6f' % r))

# ── 3. 翻轉偵測（抓最嚴重的錯誤）──────────────────────────────────
print()
print('  [3] 軸翻轉偵測　（抓 header 變換鏈出錯這種「安靜的錯」）')
best = ('無翻轉', r)
for ax, nm in [(0, '軸0(左右)'), (1, '軸1(前後)'), (2, '軸2(上下)')]:
    rf = float(np.corrcoef(ref[m], np.flip(back, axis=ax)[m])[0, 1])
    print('      翻轉 %-12s 後的相關 : %+.6f' % (nm, rf))
    if rf > best[1]:
        best = (nm, rf)
if best[0] == '無翻轉':
    print('      -> ✅ 原樣的相關最高，沒有翻轉')
    verdict.append(('無軸翻轉', True, '原樣最高'))
else:
    print('      -> 🔴 翻轉 %s 後相關更高（%.6f）—— 標籤是鏡像的！' % best)
    verdict.append(('無軸翻轉', False, '疑似 %s 翻轉' % best[0]))

# ── 4. 正規化後的 RMS 與梯度（輔助）───────────────────────────────
print()
print('  [4] 正規化後的差異　（輔助判準：兩邊各自正規化到 [0,1] 消除縮放影響）')
rn, bn = norm01(ref), norm01(back)
rms = float(np.sqrt(((rn[m] - bn[m]) ** 2).mean()))
print('      腦內 RMS 差異 : %.5f  （值域 0~1）' % rms)


def gradmag(a):
    g = np.gradient(a)
    return np.sqrt(sum(x ** 2 for x in g))


gr, gb = gradmag(rn)[m].mean(), gradmag(bn)[m].mean()
ratio = float(gb / (gr + 1e-12))
print('      梯度強度比    : %.4f  （<1 表示來回後變模糊）' % ratio)
if ratio > 0.95:
    print('      -> ✅ 幾乎沒有被平滑')
elif ratio > 0.90:
    print('      -> ⚠️ 略微平滑')
else:
    print('      -> 🔴 明顯被平滑掉細節')
verdict.append(('梯度保留', ratio > 0.90, '%.4f' % ratio))

# ── 5. 1D 剖線圖 ─────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    D, H, W = ref.shape
    fig, ax = plt.subplots(3, 1, figsize=(13, 9))
    for i, (cut, nm) in enumerate([
            ((slice(None), H // 2, W // 2), 'axis 0'),
            ((D // 2, slice(None), W // 2), 'axis 1'),
            ((D // 2, H // 2, slice(None)), 'axis 2')]):
        ax[i].plot(rn[cut], label='reference', lw=1.4)
        ax[i].plot(bn[cut], label='after round-trip', lw=1.0, ls='--')
        ax[i].set_title('1D profile through brain centre — %s' % nm, fontsize=10)
        ax[i].legend(fontsize=8)
        ax[i].grid(alpha=0.3)
    fig.suptitle('conform round-trip: centroid shift %.3f voxel, corr %.6f' % (dist, r),
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = args.out_png or os.path.splitext(os.path.splitext(args.back)[0])[0] + '_roundtrip.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print()
    print('  [5] 剖線圖已輸出：%s' % out)
    print('      次像素平移在剖線上肉眼就看得出來（曲線整體水平位移）')
except Exception as e:
    print('  [5] 剖線圖失敗（不影響上述判準）：%s' % e)

# ── 總結 ─────────────────────────────────────────────────────────
print()
print(BAR)
n_ok = sum(1 for _, o, _ in verdict if o)
print('  結論：%d/%d 通過' % (n_ok, len(verdict)))
for nm, o, v in verdict:
    print('    [%s] %-12s %s' % ('v' if o else 'X', nm, v))
print()
if n_ok == len(verdict):
    print('  -> conform 來回沒有造成幾何損失，aseg 可以直接用整數切片搬到 atlas 空間。')
else:
    print('  -> 有損失。若只是次像素平移，評估對 Dice 的影響；')
    print('     若是翻轉或大位移，**不要使用這份 aseg**，回報 FreeSurfer 端。')
print(BAR)
sys.exit(0 if n_ok == len(verdict) else 1)
