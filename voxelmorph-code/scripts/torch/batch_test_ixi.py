"""
逐 epoch 評估腳本（完整版）
跑 models/ 資料夾裡所有 .pt，畫出 NCC / SSIM / 負Jacobian比例 / Smoothness vs Epoch 曲線
自動標示綜合最佳 epoch（★）

指標說明：
  NCC   → 灰值相關，越高越好
  SSIM  → 結構相似度，越高越好
  %|J|≤0 → 形變場折疊比例，越低越好（0% = 完美）
  Smooth → 形變場梯度能量，越低越好（形變越平滑）

用法：
    python voxelmorph-code/scripts/torch/batch_test_ixi.py \
        --model-dir models/exp2_IXI \
        --atlas     IXI/atlas_mni152_09c_resize.npz \
        --test-dir  IXI/IXI_preprocessed/test \
        --out-dir   draw-img/output \
        --gpu       0

    # 只跑部分 epoch（例如每 5 個）
    python voxelmorph-code/scripts/torch/batch_test_ixi.py ... --step 5
"""

import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND']     = 'pytorch'

import argparse
import numpy as np
import torch
import voxelmorph as vxm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from skimage.metrics import structural_similarity as ssim_fn

# ── 參數 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', required=True,  help='存放 .pt 的資料夾')
parser.add_argument('--atlas',     required=True)
parser.add_argument('--test-dir',  required=True)
parser.add_argument('--out-dir',   required=True)
parser.add_argument('--step',      type=int, default=1,  help='每幾個 epoch 評估一次（預設全跑）')
parser.add_argument('--gpu',       default='0')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
print(f'裝置：{device}')

# ── 載入 atlas ────────────────────────────────────────────────────────
atlas_vol    = np.load(args.atlas)['vol'].astype(np.float32)
atlas_tensor = torch.from_numpy(atlas_vol)[None, None].to(device)

# ── 測試檔案清單 ──────────────────────────────────────────────────────
test_files = sorted([
    os.path.join(args.test_dir, f)
    for f in os.listdir(args.test_dir) if f.endswith('.npz')
])
print(f'測試筆數：{len(test_files)}')

# ── 找出所有 .pt 並排序 ───────────────────────────────────────────────
pt_files = sorted([
    f for f in os.listdir(args.model_dir) if f.endswith('.pt')
])
# 每 step 個取一個
pt_files = pt_files[::args.step]
print(f'評估 epoch 數：{len(pt_files)}')

# ── 指標函數 ─────────────────────────────────────────────────────────
def ncc(a, b):
    """Normalized Cross-Correlation"""
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8))

def jacobian_negative_ratio(flow_np):
    """
    計算形變場的負 Jacobian determinant 比例
    flow_np: (3, D, H, W)
    回傳：折疊比例（0~1），越低越好
    """
    # 計算各方向的梯度
    # flow_np[0] = displacement along D, [1] along H, [2] along W
    # Jacobian 是 3x3 矩陣，對角線加上 identity（因為是 displacement 不是 position）
    du_dd = np.gradient(flow_np[0], axis=0)  # ∂u/∂d
    du_dh = np.gradient(flow_np[0], axis=1)  # ∂u/∂h
    du_dw = np.gradient(flow_np[0], axis=2)  # ∂u/∂w
    dv_dd = np.gradient(flow_np[1], axis=0)  # ∂v/∂d
    dv_dh = np.gradient(flow_np[1], axis=1)  # ∂v/∂h
    dv_dw = np.gradient(flow_np[1], axis=2)  # ∂v/∂w
    dw_dd = np.gradient(flow_np[2], axis=0)  # ∂w/∂d
    dw_dh = np.gradient(flow_np[2], axis=1)  # ∂w/∂h
    dw_dw = np.gradient(flow_np[2], axis=2)  # ∂w/∂w

    # Jacobian = I + grad(displacement)
    # det(J) = det([[1+du/dd, du/dh, du/dw],
    #               [dv/dd, 1+dv/dh, dv/dw],
    #               [dw/dd, dw/dh, 1+dw/dw]])
    j11 = 1.0 + du_dd;  j12 = du_dh;        j13 = du_dw
    j21 = dv_dd;         j22 = 1.0 + dv_dh;  j23 = dv_dw
    j31 = dw_dd;         j32 = dw_dh;         j33 = 1.0 + dw_dw

    det_j = (j11 * (j22 * j33 - j23 * j32)
           - j12 * (j21 * j33 - j23 * j31)
           + j13 * (j21 * j32 - j22 * j31))

    neg_ratio = float((det_j <= 0).sum() / det_j.size)
    return neg_ratio

def flow_smoothness(flow_np):
    """
    計算形變場的梯度能量（Smoothness）
    flow_np: (3, D, H, W)
    回傳：平均梯度能量，越低越好
    """
    total = 0.0
    for c in range(3):
        for axis in range(3):
            grad = np.gradient(flow_np[c], axis=axis)
            total += float((grad ** 2).mean())
    return total

# ── 逐 epoch 評估 ────────────────────────────────────────────────────
epochs     = []
ncc_means  = []
ssim_means = []
jneg_means = []
smooth_means = []
all_rows   = []

for pt_name in pt_files:
    epoch_str = os.path.splitext(pt_name)[0]
    try:
        epoch_num = int(epoch_str)
    except ValueError:
        epoch_num = epoch_str

    model_path = os.path.join(args.model_dir, pt_name)
    print(f'\n── epoch {epoch_num}  ({pt_name}) ──')

    model = vxm.networks.VxmDense.load(model_path, device)
    model.to(device)
    model.eval()

    ncc_vals, ssim_vals, jneg_vals, smooth_vals = [], [], [], []

    with torch.no_grad():
        for fpath in test_files:
            vol = np.load(fpath)['vol'].astype(np.float32)
            vol_tensor = torch.from_numpy(vol)[None, None].to(device)
            moved, flow = model(vol_tensor, atlas_tensor, registration=True)
            moved_np = moved[0, 0].cpu().numpy()
            flow_np  = flow[0].cpu().numpy()   # (3, D, H, W)

            # NCC
            ncc_vals.append(ncc(moved_np, atlas_vol))
            # SSIM
            dr = max(moved_np.max(), atlas_vol.max()) - min(moved_np.min(), atlas_vol.min())
            ssim_vals.append(ssim_fn(moved_np, atlas_vol, data_range=dr))
            # 負 Jacobian 比例
            jneg_vals.append(jacobian_negative_ratio(flow_np))
            # Smoothness
            smooth_vals.append(flow_smoothness(flow_np))

    ncc_mean    = np.mean(ncc_vals)
    ssim_mean   = np.mean(ssim_vals)
    jneg_mean   = np.mean(jneg_vals)
    smooth_mean = np.mean(smooth_vals)

    print(f'  NCC={ncc_mean:.4f}  SSIM={ssim_mean:.4f}  '
          f'%|J|≤0={jneg_mean*100:.3f}%  Smooth={smooth_mean:.4f}')

    epochs.append(epoch_num)
    ncc_means.append(ncc_mean)
    ssim_means.append(ssim_mean)
    jneg_means.append(jneg_mean)
    smooth_means.append(smooth_mean)
    all_rows.append({
        'epoch': epoch_num,
        'ncc': ncc_mean,
        'ssim': ssim_mean,
        'jneg_pct': jneg_mean * 100,
        'smoothness': smooth_mean,
    })

# ── 選出最佳 epoch ──────────────────────────────────────────────────
# 綜合評分：NCC 和 SSIM 正規化到 [0,1]（越高越好），jneg 和 smooth 反轉（越低越好）
# score = 0.4*NCC_norm + 0.3*SSIM_norm + 0.15*(1 - jneg_norm) + 0.15*(1 - smooth_norm)
def normalize_01(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

ncc_norm    = normalize_01(ncc_means)
ssim_norm   = normalize_01(ssim_means)
jneg_norm   = normalize_01(jneg_means)
smooth_norm = normalize_01(smooth_means)

scores = 0.4 * ncc_norm + 0.3 * ssim_norm + 0.15 * (1 - jneg_norm) + 0.15 * (1 - smooth_norm)
best_idx = int(np.argmax(scores))
best_epoch = epochs[best_idx]

print(f'\n★ 綜合最佳 epoch = {best_epoch}')
print(f'  NCC={ncc_means[best_idx]:.4f}  SSIM={ssim_means[best_idx]:.4f}  '
      f'%|J|≤0={jneg_means[best_idx]*100:.3f}%  Smooth={smooth_means[best_idx]:.4f}')

# ── 存 CSV ────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out_dir, 'epoch_curve.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch', 'ncc', 'ssim', 'jneg_pct', 'smoothness'])
    writer.writeheader()
    writer.writerows(all_rows)
print(f'\nCSV 已存：{csv_path}')

# ── 畫曲線圖 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('#0D1117')

# 四個子圖的設定
plot_cfgs = [
    {'data': ncc_means,    'color': '#58A6FF', 'title': 'NCC vs Epoch',        'ylabel': 'NCC',     'better': 'higher'},
    {'data': ssim_means,   'color': '#3FB950', 'title': 'SSIM vs Epoch',       'ylabel': 'SSIM',    'better': 'higher'},
    {'data': [v*100 for v in jneg_means], 'color': '#F85149', 'title': '%|J|≤0 vs Epoch',   'ylabel': '%|J|≤0',  'better': 'lower'},
    {'data': smooth_means, 'color': '#F0883E', 'title': 'Smoothness vs Epoch', 'ylabel': 'Gradient Energy', 'better': 'lower'},
]

for ax, cfg in zip(axes.flat, plot_cfgs):
    ax.set_facecolor('#161B22')
    ax.tick_params(colors='#8B949E')
    ax.xaxis.label.set_color('#8B949E')
    ax.yaxis.label.set_color('#8B949E')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')

    data = cfg['data']
    ax.plot(epochs, data, color=cfg['color'], linewidth=1.8, marker='o', markersize=3)
    ax.set_title(cfg['title'], color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(cfg['ylabel'])
    ax.grid(True, color='#30363D', linestyle='--', alpha=0.5)

    # 標示最佳 epoch
    best_val = data[best_idx]
    ax.axvline(x=best_epoch, color='#5EEAD4', linestyle='--', linewidth=1, alpha=0.6)
    ax.plot(best_epoch, best_val, marker='*', color='#5EEAD4', markersize=14, zorder=5)
    # 標注文字
    ax.annotate(f'★ epoch {best_epoch}\n{best_val:.4f}',
                xy=(best_epoch, best_val),
                xytext=(15, 10 if cfg['better'] == 'higher' else -25),
                textcoords='offset points',
                color='#5EEAD4', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#5EEAD4', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#161B22', edgecolor='#5EEAD4', alpha=0.9))

fig.suptitle(
    f'Epoch Evaluation   |   ★ Best = epoch {best_epoch}   |   '
    f'NCC={ncc_means[best_idx]:.4f}  SSIM={ssim_means[best_idx]:.4f}  '
    f'%|J|≤0={jneg_means[best_idx]*100:.3f}%',
    color='white', fontsize=12, y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(args.out_dir, 'epoch_curve.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'圖片已存：{out_path}')
