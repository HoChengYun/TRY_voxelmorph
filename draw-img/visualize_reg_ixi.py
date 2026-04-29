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
    python draw-img/visualize_reg_ixi.py `
        --model    models/exp2_IXI/0100.pt `
        --atlas    IXI/atlas_mni152_09c_resize.npz `
        --test-dir IXI/IXI_preprocessed/test `
        --out-dir  draw-img/output `
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
parser.add_argument('--save-nii', action='store_true', help='是否將配準結果與變形場儲存為 .nii.gz 實體檔案')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')

# ── 載入 atlas ────────────────────────────────────────────────────────
atlas_vol, atlas_affine = vxm.py.utils.load_volfile(args.atlas, add_batch_axis=False, add_feat_axis=False, ret_affine=True)
atlas_vol = atlas_vol.astype(np.float32)
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
             for f in os.listdir(args.test_dir) 
             if f.endswith('.npz') or f.endswith('.nii.gz') or f.endswith('.nii')]
    subject_path = random.choice(files)

subject_name = os.path.splitext(os.path.basename(subject_path))[0]
if subject_name.endswith('.nii'):
    subject_name = subject_name[:-4]
print(f'受試者：{subject_name}')

vol, vol_affine = vxm.py.utils.load_volfile(subject_path, add_batch_axis=False, add_feat_axis=False, ret_affine=True)
vol = vol.astype(np.float32)
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

def jacobian_negative_ratio(flow_np):
    du_dd = np.gradient(flow_np[0], axis=0)
    du_dh = np.gradient(flow_np[0], axis=1)
    du_dw = np.gradient(flow_np[0], axis=2)
    dv_dd = np.gradient(flow_np[1], axis=0)
    dv_dh = np.gradient(flow_np[1], axis=1)
    dv_dw = np.gradient(flow_np[1], axis=2)
    dw_dd = np.gradient(flow_np[2], axis=0)
    dw_dh = np.gradient(flow_np[2], axis=1)
    dw_dw = np.gradient(flow_np[2], axis=2)
    j11 = 1.0 + du_dd;  j12 = du_dh;        j13 = du_dw
    j21 = dv_dd;        j22 = 1.0 + dv_dh;  j23 = dv_dw
    j31 = dw_dd;        j32 = dw_dh;        j33 = 1.0 + dw_dw
    det_j = (j11 * (j22 * j33 - j23 * j32)
           - j12 * (j21 * j33 - j23 * j31)
           + j13 * (j21 * j32 - j22 * j31))
    neg_ratio = float((det_j <= 0).sum() / det_j.size)
    return neg_ratio, det_j

ncc_val_source = ncc(vol, atlas_vol)
dr_source = max(vol.max(), atlas_vol.max()) - min(vol.min(), atlas_vol.min())
ssim_val_source = ssim_fn(vol, atlas_vol, data_range=dr_source)

ncc_val_warped  = ncc(moved_np, atlas_vol)
dr_warped       = max(moved_np.max(), atlas_vol.max()) - min(moved_np.min(), atlas_vol.min())
ssim_val_warped = ssim_fn(moved_np, atlas_vol, data_range=dr_warped)

jneg_val, jacobian_map = jacobian_negative_ratio(flow_np)

model_name = os.path.splitext(os.path.basename(args.model))[0]
print(f'Warped NCC={ncc_val_warped:.4f}  SSIM={ssim_val_warped:.4f}  %|J|<=0={jneg_val*100:.3f}%')

# ── 工具函數 ─────────────────────────────────────────────────────────
D, H, W = vol.shape

# ── 論文風格主題設定 ─────────────────────────────────────────────────────
BG_COLOR = 'white'
CARD_COLOR = '#F8F9FA'
TEXT_MAIN = 'black'
TEXT_SUB = '#495057'
ACCENT = '#0056B3'  # 學術深藍色
BORDER = '#DEE2E6'

def get_slice(vol3d, axis, idx):
    """取 2D 切面"""
    if axis == 'sagittal': return vol3d[idx, :, :]
    if axis == 'coronal':  return vol3d[:, idx, :]
    if axis == 'axial':    return vol3d[:, :, idx]

def get_flow_slice(flow3d, axis, idx):
    """取 flow field 的 2D 切面，回傳 (u, v) 兩個分量"""
    if axis == 'sagittal': return flow3d[1, idx, :, :], flow3d[2, idx, :, :]   # H, W
    if axis == 'coronal':  return flow3d[0, :, idx, :], flow3d[2, :, idx, :]   # D, W
    if axis == 'axial':    return flow3d[0, :, :, idx], flow3d[1, :, :, idx]   # D, H

def get_mid(axis):
    return {'sagittal': D//2, 'coronal': H//2, 'axial': W//2}[axis]

def flip_for_display(sl, axis, vol_type):
    """
    統一翻轉規則：因為輸入的 NPZ 已經對齊過，所有影像(Source, Atlas, Warped)
    在 Numpy Array 裡的空間方向是完全一致的，不應該對 atlas 特別翻轉。
    我們統一回傳原始切片，交由 imshow(origin='lower') 處理。
    """
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

def draw_warped_grid(ax, flow_u, flow_v, spacing=4, color=ACCENT, linewidth=0.5):
    """在 ax 上畫扭曲網格 (配合 imshow(sl.T) 轉置座標)"""
    H_dim, W_dim = flow_u.shape
    
    # 畫水平線 (Y 軸固定，X 軸變動)
    for y_idx in range(0, W_dim, spacing):
        xs = np.arange(H_dim, dtype=float)
        dx = flow_u[:, y_idx]
        dy = flow_v[:, y_idx]
        ax.plot(xs + dx, y_idx + dy, color=color, linewidth=linewidth, alpha=0.7)
        
    # 畫垂直線 (X 軸固定，Y 軸變動)
    for x_idx in range(0, H_dim, spacing):
        ys = np.arange(W_dim, dtype=float)
        dx = flow_u[x_idx, :]
        dy = flow_v[x_idx, :]
        ax.plot(x_idx + dx, ys + dy, color=color, linewidth=linewidth, alpha=0.7)


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
            ['gray', 'gray', 'gray', 'magma'])):
        ax = fig1.add_subplot(gs1[row_i, col_i])
        vmax = 1.0 if cmap == 'gray' else d_sl.max()
        ax.imshow(sl.T, cmap=cmap, origin='lower', aspect='equal', vmin=0, vmax=vmax)

        # 第一列標題
        if row_i == 0:
            ax.set_title(col_titles[col_i], color=TEXT_MAIN, fontsize=12,
                         fontweight='bold', pad=6)
        # 第一行標注切面名
        if col_i == 0:
            ax.text(-0.05, 0.5, f'{axis}\nslice={mid}',
                    transform=ax.transAxes, color=TEXT_SUB, fontsize=10,
                    ha='right', va='center', rotation=0)

        # Difference 加 colorbar + 統計
        if col_i == 3:
            im = ax.images[0]
            cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB)
            cbar.outline.set_edgecolor(BORDER)
            ax.text(0.05, 0.04,
                    f'MAD = {mad_val:.4f}\nmax = {diff_max:.4f}',
                    transform=ax.transAxes, color=TEXT_MAIN, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD_COLOR, edgecolor=BORDER, alpha=0.9))
        ax.axis('off')

fig1.suptitle(
    f'{subject_name}   |   model: {model_name}   |   '
    f'NCC={ncc_val_warped:.4f}   SSIM={ssim_val_warped:.4f}',
    color=TEXT_MAIN, fontsize=14, y=0.97, fontweight='bold'
)

out1 = os.path.join(args.out_dir, f'reg_{subject_name}_{model_name}_triplanar.png')
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.close(fig1)
print(f'[OK] 儲存：{out1}')

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
    ax.set_title(f'{axis}  (slice={get_mid(axis)})', color=TEXT_MAIN,
                 fontsize=12, fontweight='bold', pad=6)
    ax.axis('off')

fig2.suptitle(
    f'Checkerboard:  Warped ↔ Atlas   |   {subject_name}   |   '
    f'block={args.checker_block}px',
    color=TEXT_MAIN, fontsize=14, y=0.96, fontweight='bold'
)

out2 = os.path.join(args.out_dir, f'checker_{subject_name}_{model_name}.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f'[OK] 儲存：{out2}')

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
                     color=ACCENT, linewidth=0.5)
    ax.set_title(f'{axis}  (slice={get_mid(axis)})', color=TEXT_MAIN,
                 fontsize=12, fontweight='bold', pad=6)
    ax.axis('off')

fig3.suptitle(
    f'Warped Grid (Deformation Field)   |   {subject_name}   |   '
    f'spacing={args.grid_spacing}px',
    color=TEXT_MAIN, fontsize=14, y=0.96, fontweight='bold'
)

out3 = os.path.join(args.out_dir, f'grid_{subject_name}_{model_name}.png')
plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
plt.close(fig3)
print(f'[OK] 儲存：{out3}')

# ════════════════════════════════════════════════════════════════════════
# 圖 4：Overlay  (2 rows × 3 cols: axial / coronal / sagittal)
#   row 1 = Source + Atlas Overlay
#   row 2 = Warped + Atlas Overlay
# ════════════════════════════════════════════════════════════════════════
print('繪製 Overlay...')
fig4 = plt.figure(figsize=(16, 11))
fig4.patch.set_facecolor(BG_COLOR)

gs4 = gridspec.GridSpec(2, 3, figure=fig4, wspace=0.08, hspace=0.15,
                        left=0.18, right=0.97, top=0.82, bottom=0.03)

row_titles = ['Linear Registration\n(Source)\n+ Atlas (Red)', 
              'VoxelMorph Registration\n(Warped)\n+ Atlas (Red)']

for row_i in range(2):
    for i, axis in enumerate(axes_list):
        mid = get_mid(axis)
        a_sl = get_slice(atlas_vol, axis, mid)
        a_sl = flip_for_display(a_sl, axis, 'atlas')
        
        if row_i == 0:
            sl = get_slice(vol, axis, mid)
            sl = flip_for_display(sl, axis, 'source')
        else:
            sl = get_slice(moved_np, axis, mid)
            sl = flip_for_display(sl, axis, 'warped')

        ax = fig4.add_subplot(gs4[row_i, i])
        ax.imshow(sl.T, cmap='gray', origin='lower', aspect='equal', vmin=0, vmax=1)
        
        # 疊加紅色 atlas (背景透明)
        cmap_reds = plt.get_cmap('Reds')
        norm = plt.Normalize(vmin=0, vmax=a_sl.max())
        rgba = cmap_reds(norm(a_sl.T))
        rgba[..., 3] = np.clip(a_sl.T / a_sl.max(), 0, 1) * 0.85  # alpha blending (調高讓紅色更深)
        ax.imshow(rgba, origin='lower', aspect='equal')
        ax.axis('off')

        if i == 0:
            color = TEXT_SUB if row_i == 0 else ACCENT
            ax.text(-0.05, 0.5, row_titles[row_i],
                    transform=ax.transAxes, color=color, fontsize=12,
                    ha='right', va='center', rotation=0, fontweight='bold')
        
        if row_i == 0:
            ax.set_title(f'{axis}  (slice={get_mid(axis)})', color=TEXT_MAIN,
                         fontsize=12, fontweight='bold', pad=6)

# 標題與指標
info_text = (
    f"Linear (Source)          |  NCC: {ncc_val_source:.4f}   |   SSIM: {ssim_val_source:.4f}\n"
    f"VoxelMorph (Warped)  |  NCC: {ncc_val_warped:.4f}   |   SSIM: {ssim_val_warped:.4f}   |   %|J|≤0: {jneg_val*100:.3f}%"
)
fig4.suptitle(
    f'Registration Overlay vs Atlas   |   {subject_name}   |   model: {model_name}',
    color=TEXT_MAIN, fontsize=14, y=0.97, fontweight='bold'
)
fig4.text(0.5, 0.88, info_text, ha='center', va='bottom', color=ACCENT, fontsize=12,
          bbox=dict(boxstyle='round,pad=0.6', facecolor=CARD_COLOR, edgecolor=BORDER, alpha=0.9))

out4 = os.path.join(args.out_dir, f'overlay_{subject_name}_{model_name}.png')
plt.savefig(out4, dpi=150, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.close(fig4)
print(f'[OK] 儲存：{out4}')

# ════════════════════════════════════════════════════════════════════════
# 圖 5：Jacobian Determinant Map  (1 row × 3 cols: axial / coronal / sagittal)
#   熱圖：紅色(>1)表示放大，藍色(<1)表示縮小，接近0或負值代表摺疊
# ════════════════════════════════════════════════════════════════════════
print('繪製 Jacobian Determinant Map...')
fig5 = plt.figure(figsize=(16, 5.5))
fig5.patch.set_facecolor(BG_COLOR)

gs5 = gridspec.GridSpec(1, 3, figure=fig5, wspace=0.08,
                        left=0.03, right=0.97, top=0.85, bottom=0.03)

for i, axis in enumerate(axes_list):
    mid = get_mid(axis)
    j_sl = get_slice(jacobian_map, axis, mid)
    j_sl = flip_for_display(j_sl, axis, 'diff')
    
    ax = fig5.add_subplot(gs5[i])
    # 發散色階 (bwr) 中心點設為 1.0 (無體積變化)
    im = ax.imshow(j_sl.T, cmap='bwr', origin='lower', aspect='equal', vmin=0.5, vmax=1.5)
    ax.set_title(f'{axis}  (slice={get_mid(axis)})', color=TEXT_MAIN,
                 fontsize=12, fontweight='bold', pad=6)
    ax.axis('off')
    
    if i == 2:
        cbar = fig5.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB)
        cbar.outline.set_edgecolor(BORDER)
        cbar.set_label('Jacobian Determinant (1.0 = No Volume Change)', color=TEXT_MAIN)

fig5.suptitle(
    f'Jacobian Determinant Map   |   {subject_name}   |   '
    f'%|J|≤0 (Folding) = {jneg_val*100:.3f}%',
    color=TEXT_MAIN, fontsize=14, y=0.96, fontweight='bold'
)

out5 = os.path.join(args.out_dir, f'jacobian_{subject_name}_{model_name}.png')
plt.savefig(out5, dpi=150, bbox_inches='tight', facecolor=fig5.get_facecolor())
plt.close(fig5)
print(f'[OK] 儲存：{out5}')

print(f'\n完成！共輸出 5 張圖到 {args.out_dir}/')

# ── 儲存 NIfTI 實體檔案 ───────────────────────────────────────────────────
if args.save_nii or subject_path.endswith('.nii') or subject_path.endswith('.nii.gz'):
    print('\n儲存 NIfTI 實體檔案 (Inference Mode)...')
    # 取 subject_affine，如果沒有就借用 atlas_affine
    affine = vol_affine if vol_affine is not None else atlas_affine
    
    moved_out = os.path.join(args.out_dir, f'warped_{subject_name}_{model_name}.nii.gz')
    vxm.py.utils.save_volfile(moved_np, moved_out, affine)
    print(f'[OK] Warped Image: {moved_out}')
    
    warp_out = os.path.join(args.out_dir, f'warp_{subject_name}_{model_name}.nii.gz')
    # warp field 必須從 (3, D, H, W) 轉回 (D, H, W, 3) 才能存成標準 NIfTI
    flow_to_save = np.transpose(flow_np, (1, 2, 3, 0))
    vxm.py.utils.save_volfile(flow_to_save, warp_out, affine)
    print(f'[OK] Deformation Field: {warp_out}')


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
嗨
'''
