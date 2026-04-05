"""
配準視覺化腳本（完整版）
輸出三張圖：
  1. 三切面四格圖：Source / Atlas / Warped / Difference（axial + coronal + sagittal）
  2. Checkerboard：Warped 與 Atlas 棋盤格交錯，檢查邊界對齊
  3. Warped Grid：形變場網格圖，顯示模型學到的形變方向與大小

用法：
    python draw-img/visualize_reg_ixi.py \
        --model    models/exp2_IXI/0100.pt \
        --atlas    IXI/atlas_mni152_09c_resize.npz \
        --subject  IXI/IXI_preprocessed/test/IXI017-Guys-0698-T1.npz \
        --out-dir  draw-img/output \
        --gpu      0

    # 不指定 --subject 則隨機從 test-dir 選一張
    python draw-img/visualize_reg_ixi.py \
        --model    models/exp2_IXI/0100.pt \
        --atlas    IXI/atlas_mni152_09c_resize.npz \
        --test-dir IXI/IXI_preprocessed/test \
        --out-dir  draw-img/output \
        --gpu      0
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
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim_fn

# ── 參數 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',    required=True)
parser.add_argument('--atlas',    required=True)
parser.add_argument('--subject',  default=None,  help='指定單張 npz，不指定則從 test-dir 隨機選')
parser.add_argument('--test-dir', default=None)
parser.add_argument('--out-dir',  required=True)
parser.add_argument('--gpu',      default='0')
parser.add_argument('--checker-block', type=int, default=16, help='Checkerboard 方塊大小（pixel）')
parser.add_argument('--grid-spacing', type=int, default=4, help='Warped Grid 網格間距（pixel）')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')

# ── 載入 atlas ────────────────────────────────────────────────────────
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_tensor = torch.from_numpy(atlas_vol)[None, None].to(device)

# ── 載入模型 ──────────────────────────────────────────────────────────
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

# ── 選擇受試者 ────────────────────────────────────────────────────────
if args.subject is not None:
    subject_path = args.subject
else:
    import random
    files = [os.path.join(args.test_dir, f)
             for f in os.listdir(args.test_dir) if f.endswith('.npz')]
    subject_path = random.choice(files)

subject_name = os.path.splitext(os.path.basename(subject_path))[0]
print(f'受試者：{subject_name}')

vol = np.load(subject_path)['vol'].astype(np.float32)
vol_tensor = torch.from_numpy(vol)[None, None].to(device)

# ── 推論 ──────────────────────────────────────────────────────────────
with torch.no_grad():
    moved, flow = model(vol_tensor, atlas_tensor, registration=True)

moved_np = moved[0, 0].cpu().numpy()
flow_np  = flow[0].cpu().numpy()   # (3, D, H, W)

# ── 計算 3D 指標 ─────────────────────────────────────────────────────
def ncc(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8))

ncc_val  = ncc(moved_np, atlas_vol)
dr       = max(moved_np.max(), atlas_vol.max()) - min(moved_np.min(), atlas_vol.min())
ssim_val = ssim_fn(moved_np, atlas_vol, data_range=dr)

model_name = os.path.splitext(os.path.basename(args.model))[0]
print(f'NCC={ncc_val:.4f}  SSIM={ssim_val:.4f}')

# ── 工具函數 ─────────────────────────────────────────────────────────
D, H, W = vol.shape

def get_slice(vol3d, axis, idx):
    """取 2D 切面"""
    if axis == 'axial':    return vol3d[idx, :, :]
    if axis == 'coronal':  return vol3d[:, idx, :]
    if axis == 'sagittal': return vol3d[:, :, idx]

def get_flow_slice(flow3d, axis, idx):
    """取 flow field 的 2D 切面，回傳 (u, v) 兩個分量"""
    # flow3d shape: (3, D, H, W) → 取兩個面內方向
    if axis == 'axial':    return flow3d[1, idx, :, :], flow3d[2, idx, :, :]   # H, W
    if axis == 'coronal':  return flow3d[0, :, idx, :], flow3d[2, :, idx, :]   # D, W
    if axis == 'sagittal': return flow3d[0, :, :, idx], flow3d[1, :, :, idx]   # D, H

def get_mid(axis):
    return {'axial': D//2, 'coronal': H//2, 'sagittal': W//2}[axis]

def flip_for_display(sl, axis, vol_type):
    """
    根據之前測試得到的翻轉規則：
    - atlas 和 diff 需要 flipud
    - source 和 warped 不翻
    此規則在 axial 切面已驗證，其他切面保持一致
    """
    if vol_type in ('atlas', 'diff'):
        return np.flipud(sl)
    return sl

def make_checkerboard(img_a, img_b, block_size=16):
    """棋盤格交錯：偶數格用 img_a，奇數格用 img_b"""
    h, w = img_a.shape
    result = np.empty_like(img_a)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            yy = min(y + block_size, h)
            xx = min(x + block_size, w)
            # 棋盤判斷
            if ((y // block_size) + (x // block_size)) % 2 == 0:
                result[y:yy, x:xx] = img_a[y:yy, x:xx]
            else:
                result[y:yy, x:xx] = img_b[y:yy, x:xx]
    return result

def draw_warped_grid(ax, flow_u, flow_v, spacing=4, color='#5EEAD4', linewidth=0.4):
    """在 ax 上畫扭曲網格"""
    h, w = flow_u.shape
    # 水平線
    for y in range(0, h, spacing):
        xs = np.arange(w, dtype=float)
        ys = np.full(w, y, dtype=float) + flow_u[y, :]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.7)
    # 垂直線
    for x in range(0, w, spacing):
        ys = np.arange(h, dtype=float)
        xs = np.full(h, x, dtype=float) + flow_v[:, x]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.7)

# ── 暗色主題設定 ─────────────────────────────────────────────────────
BG_COLOR = '#0D1117'
CARD_COLOR = '#161B22'

# ════════════════════════════════════════════════════════════════════════
# 圖 1：三切面四格圖  (3 rows × 4 cols)
#   row = axial / coronal / sagittal
#   col = Source / Atlas / Warped / Difference
# ════════════════════════════════════════════════════════════════════════
print('繪製三切面四格圖...')
fig1 = plt.figure(figsize=(18, 14))
fig1.patch.set_facecolor(BG_COLOR)

gs1 = gridspec.GridSpec(3, 4, figure=fig1, wspace=0.06, hspace=0.15,
                        left=0.02, right=0.95, top=0.92, bottom=0.02,
                        width_ratios=[1, 1, 1, 1.12])

axes_list = ['axial', 'coronal', 'sagittal']
col_titles = ['Source', 'Atlas', 'Warped', 'Difference  |Warped − Atlas|']

for row_i, axis in enumerate(axes_list):
    mid = get_mid(axis)
    s_sl = get_slice(vol,       axis, mid)
    a_sl = get_slice(atlas_vol, axis, mid)
    w_sl = get_slice(moved_np,  axis, mid)
    d_sl = np.abs(w_sl - a_sl)

    # 翻轉
    s_sl = flip_for_display(s_sl, axis, 'source')
    a_sl = flip_for_display(a_sl, axis, 'atlas')
    w_sl = flip_for_display(w_sl, axis, 'warped')
    d_sl = flip_for_display(d_sl, axis, 'diff')

    mad_val  = float(np.mean(d_sl))
    diff_max = float(d_sl.max())

    for col_i, (sl, cmap) in enumerate(zip(
            [s_sl, a_sl, w_sl, d_sl],
            ['gray', 'gray', 'gray', 'hot'])):
        ax = fig1.add_subplot(gs1[row_i, col_i])
        vmax = 1.0 if cmap == 'gray' else d_sl.max()
        ax.imshow(sl.T, cmap=cmap, origin='lower', aspect='equal', vmin=0, vmax=vmax)

        # 第一列標題
        if row_i == 0:
            ax.set_title(col_titles[col_i], color='white', fontsize=12,
                         fontweight='bold', pad=6)
        # 第一行標注切面名
        if col_i == 0:
            ax.text(-0.05, 0.5, f'{axis}\nslice={mid}',
                    transform=ax.transAxes, color='#8B949E', fontsize=10,
                    ha='right', va='center', rotation=0)

        # Difference 加 colorbar + 統計
        if col_i == 3:
            im = ax.images[0]
            cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='#8B949E', labelcolor='#8B949E')
            cbar.outline.set_edgecolor('#30363D')
            ax.text(0.05, 0.04,
                    f'MAD = {mad_val:.4f}\nmax = {diff_max:.4f}',
                    transform=ax.transAxes, color='white', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD_COLOR, alpha=0.8))
        ax.axis('off')

fig1.suptitle(
    f'{subject_name}   |   model: {model_name}   |   '
    f'NCC={ncc_val:.4f}   SSIM={ssim_val:.4f}',
    color='white', fontsize=13, y=0.97
)

out1 = os.path.join(args.out_dir, f'reg_{subject_name}_{model_name}_triplanar.png')
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.close(fig1)
print(f'✓ 儲存：{out1}')

# ════════════════════════════════════════════════════════════════════════
# 圖 2：Checkerboard  (1 row × 3 cols: axial / coronal / sagittal)
#   每格 = Warped 與 Atlas 的棋盤格交錯
# ════════════════════════════════════════════════════════════════════════
print('繪製 Checkerboard...')
fig2 = plt.figure(figsize=(16, 5.5))
fig2.patch.set_facecolor(BG_COLOR)

gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.08,
                        left=0.03, right=0.97, top=0.85, bottom=0.03)

for i, axis in enumerate(axes_list):
    mid = get_mid(axis)
    a_sl = get_slice(atlas_vol, axis, mid)
    w_sl = get_slice(moved_np,  axis, mid)

    # 翻轉（checkerboard 中 warped 和 atlas 都要與原圖一致）
    a_sl = flip_for_display(a_sl, axis, 'atlas')
    w_sl = flip_for_display(w_sl, axis, 'warped')

    checker = make_checkerboard(w_sl.T, a_sl.T, block_size=args.checker_block)

    ax = fig2.add_subplot(gs2[i])
    ax.imshow(checker, cmap='gray', origin='lower', aspect='equal', vmin=0, vmax=1)
    ax.set_title(f'{axis}  (slice={get_mid(axis)})', color='white',
                 fontsize=12, fontweight='bold', pad=6)
    ax.axis('off')

fig2.suptitle(
    f'Checkerboard:  Warped ↔ Atlas   |   {subject_name}   |   '
    f'block={args.checker_block}px',
    color='white', fontsize=13, y=0.96
)

out2 = os.path.join(args.out_dir, f'checker_{subject_name}_{model_name}.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f'✓ 儲存：{out2}')

# ════════════════════════════════════════════════════════════════════════
# 圖 3：Warped Grid  (1 row × 3 cols: axial / coronal / sagittal)
#   底圖 = Source（灰色），上面畫扭曲網格（綠色線條）
# ════════════════════════════════════════════════════════════════════════
print('繪製 Warped Grid...')
fig3 = plt.figure(figsize=(16, 5.5))
fig3.patch.set_facecolor(BG_COLOR)

gs3 = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.08,
                        left=0.03, right=0.97, top=0.85, bottom=0.03)

for i, axis in enumerate(axes_list):
    mid = get_mid(axis)
    s_sl = get_slice(vol, axis, mid)
    s_sl = flip_for_display(s_sl, axis, 'source')

    # 取 flow 切面
    fu, fv = get_flow_slice(flow_np, axis, mid)

    ax = fig3.add_subplot(gs3[i])
    # 底圖：source 灰色（降低亮度讓網格更清楚）
    ax.imshow(s_sl.T * 0.4, cmap='gray', origin='lower', aspect='equal', vmin=0, vmax=1)
    # 畫扭曲網格
    draw_warped_grid(ax, fu, fv, spacing=args.grid_spacing,
                     color='#5EEAD4', linewidth=0.3)
    ax.set_title(f'{axis}  (slice={get_mid(axis)})', color='white',
                 fontsize=12, fontweight='bold', pad=6)
    ax.set_xlim(0, s_sl.shape[1])
    ax.set_ylim(0, s_sl.shape[0])
    ax.axis('off')

fig3.suptitle(
    f'Warped Grid (Deformation Field)   |   {subject_name}   |   '
    f'spacing={args.grid_spacing}px',
    color='white', fontsize=13, y=0.96
)

out3 = os.path.join(args.out_dir, f'grid_{subject_name}_{model_name}.png')
plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
plt.close(fig3)
print(f'✓ 儲存：{out3}')

print(f'\n完成！共輸出 3 張圖到 {args.out_dir}/')


'''
這些視覺化方式在配準領域的幾篇經典和知名論文中都能看到：

**1. VoxelMorph 原始論文（Balakrishnan et al., 2019）**
論文的 Figure 裡就有多切面（axial/coronal/sagittal）的 Warped vs Atlas 比較，以及形變場的顏色編碼圖。這是最基本的呈現方式。

**2. TransMorph（Chen et al., 2022, Medical Image Analysis）**
這篇是用 Transformer 做配準，Figure 裡面用了 checkerboard（棋盤格）來比較 Warped 和 Atlas 的邊界對齊情況，也有 warped grid（扭曲網格圖）來視覺化形變場，還有 Jacobian determinant map 來檢查是否有折疊。這篇的視覺化最完整，很值得參考。

**3. SyN / ANTs 的經典評估論文（Avants et al., 2011, NeuroImage）**
ANTs SyN 的方法論文，用了 deformation field 的 RGB 顏色編碼（把 x/y/z 三個方向的位移分別映射到 R/G/B），以及 Jacobian determinant map（紅色表示負值 = 形變折疊）。

**4. 各種 MICCAI 論文的共通做法**
基本上 MICCAI 投稿的配準論文，標配是三切面比較 + warped grid + Jacobian map，有的還會加 checkerboard。

所以你現在的四格圖（Source/Atlas/Warped/Difference）是最基礎的，如果要讓結果更有說服力，至少加上三切面和 checkerboard 會好很多。

Sources:
- [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/abs/1809.05231)
- [TransMorph: Transformer for unsupervised medical image registration](https://pmc.ncbi.nlm.nih.gov/articles/PMC9999483/)
- [Evaluation of volume-based and surface-based brain image registration methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC2862732/)
'''
