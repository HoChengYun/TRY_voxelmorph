"""
把 MNI152 原始 T1 補零成 256^3，供 FreeSurfer recon-all 使用（避免 conform 內插）。

為什麼要這樣做
--------------
FreeSurfer 的 conform 會把輸入轉成 256^3 / 1mm / LIA，並保留 c_ras。
中心體素座標定義為 width/2：

    輸入 193 -> 中心 96.5   （半整數）
    輸出 256 -> 中心 128.0  （整數）
    -> 偏移 -31.5 個體素，落在體素之間，**觸發三線性內插**

229 同理（偏移 -13.5）。三個軸都會內插。而 aseg 之後還要 -rl 轉回原始網格，
等於來回內插兩次；影像被平滑兩次，標籤走最近鄰則邊界可能位移到 1 個體素。

先補零成 256^3 之後，三軸中心都是 128.0，跟 conform 目標一致，偏移為 0，
conform 只剩下 RAS->LIA 的軸重排（純排列與翻轉，對等向立方網格是精確的）。

補零量（純整數，零內插）
    193 -> 256 : 前 31 / 後 32
    229 -> 256 : 前 13 / 後 14
    193 -> 256 : 前 31 / 後 32

拿到 aseg 之後要切回原始網格，用：
    [31:224, 13:242, 31:224]

用法
----
    python ASD\\make_padded_atlas_input.py
    # 產生 IXI\\mni152_09c_t1_padded256.nii.gz，把這個檔給 recon-all
"""

import os
import numpy as np
import nibabel as nib

SRC = os.path.join('IXI', 'mni_icbm152_nlin_asym_09c_nifti',
                   'mni_icbm152_t1_tal_nlin_asym_09c.nii')
DST = os.path.join('IXI', 'mni152_09c_t1_padded256.nii.gz')
TARGET = 256

img = nib.load(SRC)
a = np.asarray(img.dataobj)
shape = a.shape

print('來源 :', SRC)
print('  shape=%s  spacing=%s  方向=%s'
      % (shape, tuple(round(float(z), 4) for z in img.header.get_zooms()),
         nib.aff2axcodes(img.affine)))

pads = []
for ax in range(3):
    d = TARGET - shape[ax]
    if d < 0:
        raise ValueError('軸 %d 已經大於 %d，本腳本只處理補零' % (ax, TARGET))
    pads.append((d // 2, d - d // 2))
print('  補零量（前, 後）:', pads)

padded = np.pad(a, pads, mode='constant', constant_values=0)

# 補零之後 voxel 索引整體位移了，affine 要跟著調整，
# 否則腦在世界座標裡的位置會跑掉（ANTs / FreeSurfer 都靠這個定位）。
#   新索引 j = 舊索引 i + f   =>   i = j - f
#   world = A @ [i,1] = A @ ([j,1] - [f,0])
shift = np.eye(4)
shift[:3, 3] = -np.array([f for f, _ in pads], dtype=float)
new_affine = img.affine @ shift

out = nib.Nifti1Image(padded, new_affine, header=img.header)
out.header.set_data_dtype(a.dtype)
nib.save(out, DST)

# ── 驗證：世界座標有沒有跑掉 ────────────────────────────────────────
chk = nib.load(DST)
print()
print('輸出 :', DST)
print('  shape=%s  spacing=%s  方向=%s'
      % (chk.shape, tuple(round(float(z), 4) for z in chk.header.get_zooms()),
         nib.aff2axcodes(chk.affine)))

ok = True

# 1) 隨機取幾個體素，確認補零前後指向同一個世界座標
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    i = np.array([rng.integers(0, s) for s in shape], dtype=float)
    j = i + np.array([f for f, _ in pads], dtype=float)
    w_old = img.affine @ np.append(i, 1)
    w_new = chk.affine @ np.append(j, 1)
    worst = max(worst, float(np.abs(w_old - w_new).max()))
print('  [%s] 世界座標一致性：最大偏差 %.3g mm' % ('v' if worst < 1e-6 else 'X', worst))
ok &= worst < 1e-6

# 2) 切回原始網格要能完全還原
slc = tuple(slice(f, f + shape[ax]) for ax, (f, _) in enumerate(pads))
back = np.asarray(chk.dataobj)[slc]
d = float(np.abs(back.astype(np.float64) - a.astype(np.float64)).max())
print('  [%s] 切回 %s 逐 voxel 最大差異 %g' % ('v' if d == 0 else 'X', str(back.shape), d))
ok &= d == 0

# 3) 補的區域必須全是 0（沒有蓋到腦）
m = np.ones(chk.shape, dtype=bool)
m[slc] = False
mx = float(np.asarray(chk.dataobj)[m].max()) if m.any() else 0.0
print('  [%s] 補零區域最大值 %g（應為 0）' % ('v' if mx == 0 else 'X', mx))
ok &= mx == 0

print()
print('=' * 74)
print('  拿到 aseg 之後，切回原始 (193,229,193) 網格用這組切片：')
print('      [%d:%d, %d:%d, %d:%d]'
      % (slc[0].start, slc[0].stop, slc[1].start, slc[1].stop, slc[2].start, slc[2].stop))
print('  再接 atlas_v3 的裁切 [0:192, 2:226, 0:192]，就到訓練用的 atlas 空間。')
print('  兩段都是整數切片，全程零內插。')
print('=' * 74)
print('  結果：%s' % ('全部通過' if ok else '有檢查未通過，不要使用這個檔案'))
raise SystemExit(0 if ok else 1)
