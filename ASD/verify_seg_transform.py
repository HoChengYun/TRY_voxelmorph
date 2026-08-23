"""
驗證「用同一個 Affine 變換搬標籤」這件事在本機 ANTsPy 上真的成立。

交接文件第 5 節的程式碼作者自陳「沒在你們環境跑過」，這支腳本補上驗證。
不需要 FreeSurfer 資料，用現有的 IXI 影像 + 合成標籤即可。

測試設計（重點是「能給出確定答案」，不是看圖猜）：

  1. 拿一張真實 IXI T1，用強度門檻把它切成 4 個標籤 -> 合成 aseg
  2. reg = ants.registration(atlas, img, 'Affine')
  3. img_nn = apply_transforms(..., 影像, fwdtransforms, nearestNeighbor)
     seg_nn = apply_transforms(..., 標籤, fwdtransforms, nearestNeighbor)
  4. ⭐ 關鍵斷言：quantize(img_nn) 必須「逐 voxel 完全等於」seg_nn
     最近鄰對影像和標籤取的是同一顆來源 voxel，若變換真的共用，
     兩者必然 bit-exact 相同。有任何一個 voxel 不同就代表變換沒共用。
  5. 反證：同樣的標籤改用 linear 內插，必須出現非整數值
     （證明 preprocess_fs.py 裡那個整數檢查抓得到用錯內插法的情況）

用法：
    python fs_pipeline\\verify_seg_transform.py ^
        --img   IXI\\IXI-T1\\IXI002-Guys-0828-T1.nii.gz ^
        --atlas IXI\\atlas_mni152_09c_v3.nii.gz
"""

import os
import sys
import time
import argparse
import numpy as np
import ants

ap = argparse.ArgumentParser()
ap.add_argument('--img', required=True, help='任一張 T1 .nii.gz')
ap.add_argument('--atlas', required=True)
args = ap.parse_args()

print("=" * 66)
print("  驗證：aseg 標籤沿用影像的 Affine 變換 + 最近鄰內插")
print("=" * 66)
print(f"  ANTsPy : {ants.__version__}")
print(f"  影像   : {args.img}")
print(f"  atlas  : {args.atlas}\n")

atlas = ants.image_read(os.path.normpath(args.atlas))
img = ants.image_read(os.path.normpath(args.img))
print(f"  atlas shape={atlas.shape}  spacing={tuple(round(s, 4) for s in atlas.spacing)}")
print(f"  影像  shape={img.shape}  spacing={tuple(round(s, 4) for s in img.spacing)}\n")

# ── 1. 合成標籤：用強度分位數把影像切成 4 類 ─────────────────────────
arr = img.numpy()
pos = arr[arr > 0]
t1, t2, t3 = np.percentile(pos, [40, 70, 90])
lab = np.zeros(arr.shape, dtype=np.float32)
lab[arr > t1] = 10       # 故意用 FreeSurfer 風格的不連續標籤值
lab[arr > t2] = 17
lab[arr > t3] = 41
seg = img.new_image_like(lab)


def quantize(a):
    """跟合成標籤完全相同的規則，套用在任意影像陣列上。"""
    out = np.zeros(a.shape, dtype=np.int16)
    out[a > t1] = 10
    out[a > t2] = 17
    out[a > t3] = 41
    return out


print(f"  合成標籤：值 = {sorted(np.unique(lab).astype(int).tolist())}"
      f"（門檻 {t1:.1f} / {t2:.1f} / {t3:.1f}）\n")

# ── 2. Affine 對位 ───────────────────────────────────────────────────
print("  執行 ants.registration(type_of_transform='Affine') ...")
t0 = time.time()
reg = ants.registration(fixed=atlas, moving=img, type_of_transform='Affine', verbose=False)
print(f"    完成，耗時 {time.time() - t0:.1f}s")
print(f"    fwdtransforms = {[os.path.basename(t) for t in reg['fwdtransforms']]}")

# 確認變換不是恆等（否則這個測試沒有鑑別力）
tx = ants.read_transform(reg['fwdtransforms'][0])
params = np.asarray(tx.parameters, dtype=float)
lin = params[:9].reshape(3, 3)
off = params[9:12]
dev = float(np.abs(lin - np.eye(3)).max())
print(f"    線性部分偏離單位矩陣：{dev:.4f}   平移："
      f"{tuple(round(float(v), 2) for v in off)}")
if dev < 1e-3 and np.abs(off).max() < 1e-3:
    print("    [!] 變換接近恆等，這個測試的鑑別力會下降")
print()

# ── 3. 用同一個變換搬影像與標籤（都用最近鄰）─────────────────────────
img_nn = ants.apply_transforms(fixed=atlas, moving=img,
                               transformlist=reg['fwdtransforms'],
                               interpolator='nearestNeighbor')
seg_nn = ants.apply_transforms(fixed=atlas, moving=seg,
                               transformlist=reg['fwdtransforms'],
                               interpolator='nearestNeighbor')

img_nn_np = img_nn.numpy()
seg_nn_np = seg_nn.numpy()

results = []


def check(name, passed, detail):
    results.append((name, passed, detail))
    print(f"  [{'v' if passed else 'X'}] {name}")
    print(f"      {detail}")


# 測試 A：shape 與 atlas 一致
check("標籤 shape 等於 atlas",
      seg_nn_np.shape == atlas.shape,
      f"seg={seg_nn_np.shape}  atlas={atlas.shape}")

# 測試 B：值仍為整數
frac = float(np.abs(seg_nn_np - np.round(seg_nn_np)).max())
check("最近鄰搬完後標籤值仍是整數",
      frac == 0.0,
      f"與最近整數的最大偏差 = {frac:.6g}")

# 測試 C：沒有跑出原本不存在的標籤值
before = set(np.unique(lab).round().astype(int).tolist())
after = set(np.unique(seg_nn_np).round().astype(int).tolist())
check("沒有產生原本不存在的標籤值",
      after.issubset(before),
      f"搬之前 {sorted(before)} -> 搬之後 {sorted(after)}"
      + (f"   多出來：{sorted(after - before)}" if after - before else ""))

# 測試 D ⭐ 核心：影像與標籤取到同一顆來源 voxel
derived = quantize(img_nn_np)
actual = np.round(seg_nn_np).astype(np.int16)
n_diff = int((derived != actual).sum())
check("影像與標籤逐 voxel 完全對齊（bit-exact）",
      n_diff == 0,
      f"不一致的 voxel：{n_diff} / {actual.size}"
      + ("   -> 兩者取到同一顆來源 voxel，變換確實共用" if n_diff == 0 else
         "   -> 變換沒有共用，或內插法不同"))

# 測試 E（反證）：linear 內插必須產生非整數值
seg_lin = ants.apply_transforms(fixed=atlas, moving=seg,
                                transformlist=reg['fwdtransforms'],
                                interpolator='linear').numpy()
lin_frac = float(np.abs(seg_lin - np.round(seg_lin)).max())
bogus = sorted(set(np.unique(seg_lin).tolist()) - {0.0, 10.0, 17.0, 41.0})
# 取分佈中段當例子；bogus 由小到大排序，開頭全是逼近 0 的值，印出來看不出問題
mid = [round(bogus[len(bogus) * k // 6], 3) for k in range(1, 6)] if bogus else []
check("反證：改用 linear 會插出不存在的標籤值（守門檢查有效）",
      lin_frac > 1e-6,
      f"最大小數偏差 = {lin_frac:.4f}；捏造出的值例如 {mid}"
      f"（共 {len(bogus)} 種不存在於原標籤集的值）")

# 測試 F：腦組織體積沒有明顯流失（sanity check）
vol_before = int((lab > 0).sum())
vol_after = int((actual > 0).sum())
ratio = vol_after / max(vol_before, 1)
check("標籤總體積變化在合理範圍（Affine 會縮放，非 1.0 屬正常）",
      0.5 < ratio < 2.0,
      f"{vol_before} -> {vol_after} voxel（比值 {ratio:.3f}）")

print("\n" + "=" * 66)
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"  結果：{n_pass}/{len(results)} 通過")
for name, ok, _ in results:
    if not ok:
        print(f"    失敗：{name}")
print("=" * 66)
sys.exit(0 if n_pass == len(results) else 1)
