# -*- coding: utf-8 -*-
"""把 exp2 / exp4 / exp3 的圖排成三版對照，給簡報用。

輸出到 models/deck_charts/：
    curve_exp234.png      三顆的 val Dice 曲線 + 擠爆比例（從 dice_curve_val.csv 重畫）
    jacobian_exp234.png   三顆的 Jacobian 圖疊成一張（讀既有的 vis_T054 輸出）
    grid_exp234.png       三顆的形變網格，簡報專用的清楚版（2026-09-29 起重畫，不再堆 vis 的 PNG）

⚠️ jacobian 那張是把 visualize 出來的 PNG 直接堆起來，不是重新計算。
   所以 vis_T054 那些圖要先存在（見手冊 §13 的視覺化速查）。

📌 grid 那張是拿三顆模型對 T054 重跑一次推論，取同一個軸狀切面（跟 visualize_reg_ixi.py 一樣：
   轉成 RAS、取中間那片）再自己畫：格子放疏、線加粗換亮色、只框腦的範圍。
   推論結果存成 grid_T054_flow.npz（在 deck_charts 裡），之後重畫直接讀；要重算就刪掉它。
   models\<實驗>\vis_T054\ 裡原本的 grid 圖不動。
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
os.makedirs(OUT, exist_ok=True)

INK, MUTED, RULE, PAPER = '#141A1D', '#5F6A6B', '#D9D9D2', '#FAFAF8'
TEAL, RUST_L, RUST = '#0E7C7B', '#D9895A', '#A34F1B'

# (實驗, 最佳 epoch, 標籤, 顏色)
EXPS = [
    ('mix_exp2', '0240', '速度場版・平滑權重 2', TEAL),
    ('mix_exp4', '0230', '位移場版・平滑權重 2', RUST_L),
    ('mix_exp3', '0240', '位移場版・平滑權重 1', RUST),
]


def curve(exp):
    p = os.path.join(ROOT, 'models', exp, 'dice_curve_val.csv')
    with open(p, encoding='utf-8') as f:
        r = sorted([(int(x['epoch']), float(x['dice_mean']), float(x['jneg_pct']))
                    for x in csv.DictReader(f)])
    return np.array(r)


# ── 1. 三顆的訓練曲線 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                         gridspec_kw={'height_ratios': [1.35, 1]})
for exp, ep, lab, col in EXPS:
    r = curve(exp)
    axes[0].plot(r[:, 0], r[:, 1], color=col, lw=2, marker='o', ms=4, label=lab)
    axes[1].plot(r[:, 0], r[:, 2], color=col, lw=2, marker='o', ms=4, label=lab)
    b = r[np.argmax(r[:, 1])]
    axes[0].plot(b[0], b[1], '*', color=col, ms=18, markeredgecolor=INK, zorder=5)

axes[0].set_ylabel('Dice（驗證集 51 位）', fontsize=11)
axes[0].set_title('對得多準：星號是選中的那一輪', fontsize=12.5, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=11)
axes[0].set_ylim(0.67, 0.815)

axes[1].axhline(0.366, ls='--', color='#C0392B', lw=1.2)
axes[1].text(248, 0.375, '論文的同版本 0.366%', ha='right', color='#C0392B', fontsize=9.5)
axes[1].set_ylabel('擠爆的比例', fontsize=11)
axes[1].set_xlabel('訓練輪數', fontsize=11)
axes[1].set_title('有沒有擠爆', fontsize=12.5, fontweight='bold')
axes[1].set_ylim(-0.02, 0.45)

for ax in axes:
    ax.grid(alpha=.3)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'curve_exp234.png'), dpi=130)
plt.close(fig)
print('->', os.path.join(OUT, 'curve_exp234.png'))


# ── 2 & 3. 把既有的視覺化 PNG 堆成三版對照 ───────────────────────────
def stack(kind, subject, out_name, xcrop=(0.035, 0.295), ycrop=(0.145, 1.0)):
    """三顆橫著排，只取軸狀切面那一格（原圖是三個切面並排，太寬塞不進投影片）。"""
    fig = plt.figure(figsize=(13, 5.2), facecolor=PAPER)
    for k, (exp, ep, lab, col) in enumerate(EXPS):
        p = os.path.join(ROOT, 'models', exp, 'vis_' + subject, '%s_%s_%s.png' % (kind, subject, ep))
        if not os.path.exists(p):
            raise SystemExit('[X] 找不到 %s' % p)
        im = mpimg.imread(p)
        h, w = im.shape[:2]
        im = im[int(h * ycrop[0]):int(h * ycrop[1]), int(w * xcrop[0]):int(w * xcrop[1])]
        ax = fig.add_axes([0.02 + k * 0.327, 0.12, 0.31, 0.80])
        ax.imshow(im)
        ax.axis('off')
        ax.text(0.5, -0.045, lab, transform=ax.transAxes, ha='center', va='top',
                fontsize=14, fontweight='bold', color=col)
    fig.savefig(os.path.join(OUT, out_name), dpi=130, facecolor=PAPER)
    plt.close(fig)
    print('->', os.path.join(OUT, out_name))


stack('jacobian', 'T054', 'jacobian_exp234.png')


# ── 3. 形變網格：重跑推論、自己畫清楚版 ─────────────────────────────────
def grid_slices(subject):
    """三顆模型對 subject 的形變場，取軸狀中間那片。有快取就讀快取。"""
    cache = os.path.join(OUT, 'grid_%s_flow.npz' % subject)
    if os.path.exists(cache):
        return dict(np.load(cache))

    os.environ['NEURITE_BACKEND'] = 'pytorch'
    os.environ['VXM_BACKEND'] = 'pytorch'
    import torch
    import voxelmorph as vxm
    sys.path.insert(0, os.path.join(ROOT, 'ASD'))
    from orient import canonical_axes, to_ras, flow_to_ras

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    atlas = np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
    seg = np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg']
    vol = np.load(os.path.join(ROOT, 'data', 'mixed_preprocessed_v2', 'test', subject + '.npz'))['vol']
    vol = vol.astype(np.float32)
    perm, flip = canonical_axes(seg.astype(np.int32))      # 跟 visualize_reg_ixi.py 一樣轉成 RAS 再切

    src = to_ras(vol, perm, flip)
    mid = src.shape[2] // 2                                  # 軸狀面取中間那片（同 visualize_reg_ixi.py）
    out = {'src': src[:, :, mid]}
    a_t = torch.from_numpy(atlas)[None, None].to(device)
    v_t = torch.from_numpy(vol)[None, None].to(device)
    for exp, ep, lab, col in EXPS:
        model = vxm.networks.VxmDense.load(os.path.join(ROOT, 'models', exp, ep + '.pt'), device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            _, flow = model(v_t, a_t, registration=True)
        f = flow_to_ras(flow[0].cpu().numpy(), perm, flip)
        out[exp + '_u'] = f[0][:, :, mid]
        out[exp + '_v'] = f[1][:, :, mid]
        print('   %s：位移最大 %.1f 格' % (exp, float(np.abs(f[:2, :, :, mid]).max())))
        del model, flow
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    np.savez_compressed(cache, **out)
    print('-> 快取', cache)
    return out


def grid_clear(subject, out_name, spacing=6, line='#FFD23F', lw=1.0, dim=0.55, margin=6):
    """三顆並排的形變網格。格線＝把 atlas 上的正方格子照形變場搬到受試者身上的樣子。"""
    d = grid_slices(subject)
    src = d['src']                                           # [左右, 前後]，imshow 用轉置畫
    # 只框腦的範圍，四周黑底裁掉，腦才放得大
    lr = np.nonzero(src.max(axis=1) > 0.02)[0]               # 第 0 軸 = 左右（畫在橫軸）
    ap = np.nonzero(src.max(axis=0) > 0.02)[0]               # 第 1 軸 = 前後（畫在縱軸）
    xl = (lr.min() - margin, lr.max() + margin)
    yl = (ap.min() - margin, ap.max() + margin)
    fig = plt.figure(figsize=(13, 5.2), facecolor=PAPER)
    for k, (exp, ep, lab, col) in enumerate(EXPS):
        u, v = d[exp + '_u'], d[exp + '_v']
        ax = fig.add_axes([0.02 + k * 0.327, 0.12, 0.31, 0.80])
        ax.imshow(src.T * dim, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='bilinear')
        n0, n1 = u.shape
        for j in range(0, n1, spacing):                      # 橫線
            ax.plot(np.arange(n0) + u[:, j], j + v[:, j], color=line, lw=lw, alpha=0.95, solid_capstyle='round')
        for i in range(0, n0, spacing):                      # 直線
            ax.plot(i + u[i, :], np.arange(n1) + v[i, :], color=line, lw=lw, alpha=0.95, solid_capstyle='round')
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(0.5, -0.045, lab, transform=ax.transAxes, ha='center', va='top',
                fontsize=14, fontweight='bold', color=col)
    fig.savefig(os.path.join(OUT, out_name), dpi=130, facecolor=PAPER)
    plt.close(fig)
    print('->', os.path.join(OUT, out_name))


grid_clear('T054', 'grid_exp234.png')
