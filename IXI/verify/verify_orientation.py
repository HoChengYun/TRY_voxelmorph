"""
視覺化驗證：確認 preprocessed npz 的方向是否與 atlas 一致。

做法：取 atlas + 5 個 npz，畫出 axial / coronal / sagittal 切面，
      看腦的朝向是否一致。如果方向跑掉，會一眼看出來（左右顛倒、上下翻轉等）。
"""

import os
import sys
import glob
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── 載入 atlas ────────────────────────────────────────────────────────
atlas_path = os.path.join(_HERE, 'atlas_mni152_09c.npz')
atlas_vol = np.load(atlas_path)['vol'].astype(np.float32)

# atlas 可能是 (193,229,193)，resize 到 (192,224,192) 以匹配 npz
from scipy.ndimage import zoom
target_shape = (192, 224, 192)
if atlas_vol.shape != target_shape:
    factors = tuple(t / s for t, s in zip(target_shape, atlas_vol.shape))
    atlas_vol = zoom(atlas_vol, factors, order=1)

# ── 載入幾個 npz（分別從不同醫院抽樣）────────────────────────────────
npz_dir = os.path.join(_HERE, 'IXI_preprocessed', 'train')
all_npz = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))

# 抽一個 Guys, 一個 HH, 一個 IOP
samples = []
for keyword in ['Guys', 'HH', 'IOP']:
    for f in all_npz:
        if keyword in os.path.basename(f) and f not in samples:
            samples.append(f)
            break

# 再隨機補 2 個
random.seed(42)
remaining = [f for f in all_npz if f not in samples]
samples.extend(random.sample(remaining, min(2, len(remaining))))

print(f"Atlas shape: {atlas_vol.shape}")
print(f"Selected {len(samples)} samples for visualization\n")

# ── 畫圖：每行一個影像，三列 = axial / coronal / sagittal ───────────
n_rows = 1 + len(samples)  # atlas + samples
fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.5 * n_rows))

def plot_slices(ax_row, vol, title):
    """在一行 3 個子圖上畫 axial / coronal / sagittal 中央切面"""
    d, h, w = vol.shape
    
    # Axial (xy plane, z=mid) - 從上往下看
    ax_row[0].imshow(vol[d//2, :, :].T, cmap='gray', origin='lower')
    ax_row[0].set_title(f'{title}\nAxial (dim0={d//2})')
    ax_row[0].set_xlabel('dim1')
    ax_row[0].set_ylabel('dim2')
    
    # Coronal (xz plane, y=mid) - 從前往後看
    ax_row[1].imshow(vol[:, h//2, :].T, cmap='gray', origin='lower')
    ax_row[1].set_title(f'Coronal (dim1={h//2})')
    ax_row[1].set_xlabel('dim0')
    ax_row[1].set_ylabel('dim2')
    
    # Sagittal (yz plane, x=mid) - 從側面看
    ax_row[2].imshow(vol[:, :, w//2].T, cmap='gray', origin='lower')
    ax_row[2].set_title(f'Sagittal (dim2={w//2})')
    ax_row[2].set_xlabel('dim0')
    ax_row[2].set_ylabel('dim1')
    
    for ax in ax_row:
        ax.axis('off')

# 畫 Atlas
plot_slices(axes[0], atlas_vol, 'ATLAS (MNI152)')

# 畫 samples
for i, f in enumerate(samples):
    name = os.path.basename(f).replace('.npz', '')
    vol = np.load(f)['vol']
    print(f"  [{i+1}] {name}  shape={vol.shape}")
    plot_slices(axes[i + 1], vol, name)

plt.tight_layout()
out_path = os.path.join(_HERE, 'orientation_check.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()

print(f"\n[OK] 已儲存：{out_path}")
print("""
如何判讀：
  - 所有行的腦部朝向應該一致（鼻子朝同一邊、左右腦對稱方式相同）
  - 如果某一行翻轉了，代表那張影像的方向跑掉了
  - 特別注意 Sagittal 切面：鼻子應該都朝同一個方向
""")
