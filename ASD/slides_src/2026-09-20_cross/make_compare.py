# -*- coding: utf-8 -*-
"""把四顆模型的圖排成四版對照，給簡報用。

--set ablation（預設）：exp2 / exp5 / exp4 / exp3，一次只改一件事：exp2 → exp5 只換解析度、
    exp5 → exp4 只換版本、exp4 → exp3 只換平滑權重（手冊 §20.5；mix_exp5 是 2026-09-30 補進來的，之前是三版）
--set lambda（2026-10-04，10/14 簡報用）：速度場全尺寸的平滑權重 2 / 1 / 0.5（exp5 / exp6 / exp7）
    ＋位移場權重 1（exp3）當對照
--set wide（2026-10-05，10/14 簡報用）：版本 × 寬度（exp3 / mix_wide / exp6 / mix_wide_vel），
    全部全尺寸、平滑權重 1

輸出到 models/deck_charts/（<後綴> = ablation 是 exp2345、lambda 是 lambda、wide 是 wide）：
    curve_<後綴>.png      四顆的 val Dice 曲線 + 擠爆比例（從 dice_curve_val.csv 重畫）
    jacobian_<後綴>.png   四顆的 Jacobian 圖疊成一張（讀既有的 vis_T054 輸出）
    grid_<後綴>.png       四顆的形變網格，簡報專用的清楚版（2026-09-29 起重畫，不再堆 vis 的 PNG）

⚠️ jacobian 那張是把 visualize 出來的 PNG 直接堆起來，不是重新計算。
   所以 vis_T054 那些圖要先存在（見手冊 §13 的視覺化速查）。

📌 grid 那張是拿模型對 T054 重跑一次推論，取同一個軸狀切面（跟 visualize_reg_ixi.py 一樣：
   轉成 RAS、取中間那片）再自己畫：格子放疏、線加粗換亮色、只框腦的範圍。
   推論結果存成 grid_T054_flow.npz（在 deck_charts 裡），之後重畫直接讀；
   新加的實驗快取裡沒有，會自動補算；要全部重算就刪掉它。
   models\<實驗>\vis_T054\ 裡原本的 grid 圖不動。
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.image as mpimg

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
os.makedirs(OUT, exist_ok=True)

INK, MUTED, RULE, PAPER = '#141A1D', '#5F6A6B', '#D9D9D2', '#FAFAF8'
TEAL, TEAL_M, RUST_L, RUST = '#0E7C7B', '#3A9E9C', '#D9895A', '#A34F1B'

# (實驗, 最佳 epoch, 標籤, 顏色)
SETS = {
    'ablation': ('exp2345', [
        ('mix_exp2', '0240', '速度場・半解析度・權重 2', TEAL),
        ('mix_exp5', '0150', '速度場・全尺寸・權重 2', TEAL_M),
        ('mix_exp4', '0230', '位移場・全尺寸・權重 2', RUST_L),
        ('mix_exp3', '0240', '位移場・全尺寸・權重 1', RUST),
    ]),
    # lambda、wide 兩組是 10/14 簡報用的：2026-10-07 起標籤改正式用語（使用者要求）；ablation 是 09-20 那份的，照舊
    'lambda': ('lambda', [
        ('mix_exp5', '0150', 'SVF・λ = 2', '#5BB8B6'),
        ('mix_exp6', '0190', 'SVF・λ = 1', TEAL),
        ('mix_exp7', '0250', 'SVF・λ = 0.5', '#0A4F4E'),
        ('mix_exp3', '0240', 'Displacement・λ = 1', RUST),
    ]),
    'wide': ('wide', [
        ('mix_exp3', '0240', 'Displacement・default width', RUST_L),
        ('mix_wide', '0225', 'Displacement・2× width', RUST),
        ('mix_exp6', '0190', 'SVF・default width', TEAL_M),
        ('mix_wide_vel', '0240', 'SVF・2× width', TEAL),
    ]),
}
SET = sys.argv[sys.argv.index('--set') + 1] if '--set' in sys.argv else 'ablation'
TAG, EXPS = SETS[SET]
PANEL = lambda k: [0.012 + k * 0.247, 0.12, 0.235, 0.80]      # 四格橫排
FIGSIZE = (16, 5.2)


# folding（2026-10-07 使用者決定：跟論文比較一律用 voxel 數）：論文 Table I 的百分比分母是固定的 5.2 M voxel，
#   我們 atlas 非背景只有 1.87 M，百分比不能直接比；論文同一張表也列了 folding voxel 數（VoxelMorph (CC) 19,077），
#   voxel 都是 1 mm³，數量可以直接比。10/14 簡報的兩組（lambda、wide）畫每位平均 folding voxel 數
#   ＝ jneg_pct × 整個影像大小（精確值，不用換算）。09-20 那份（ablation）照舊畫百分比
VOL = 192 * 224 * 192
PAPER_N = 19077                                           # 論文 Table I，VoxelMorph (CC) 之 folding voxel 數


def curve(exp):
    p = os.path.join(ROOT, 'models', exp, 'dice_curve_val.csv')
    with open(p, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if SET == 'ablation':
        fold = lambda x: float(x['jneg_pct'])
    else:
        fold = lambda x: float(x['jneg_pct']) / 100 * VOL
    return np.array(sorted([(int(x['epoch']), float(x['dice_mean']), fold(x)) for x in rows]))


# ── 1. 四顆的訓練曲線 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                         gridspec_kw={'height_ratios': [1.35, 1]})
for exp, ep, lab, col in EXPS:
    r = curve(exp)
    axes[0].plot(r[:, 0], r[:, 1], color=col, lw=2, marker='o', ms=4, label=lab)
    axes[1].plot(r[:, 0], r[:, 2], color=col, lw=2, marker='o', ms=4, label=lab)
    b = r[np.argmax(r[:, 1])]
    axes[0].plot(b[0], b[1], '*', color=col, ms=18, markeredgecolor=INK, zorder=5)

axes[0].set_ylabel('Dice（validation, n = 51）', fontsize=11)
axes[0].set_title('Validation Dice（★：選定之 epoch）', fontsize=12.5, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=11)
axes[0].set_ylim(0.67, 0.815)

if SET == 'ablation':                      # 09-20 那份照舊
    axes[1].axhline(0.366, ls='--', color='#C0392B', lw=1.2)
    axes[1].text(248, 0.375, 'VoxelMorph（TMI 2019, Table I）0.366%', ha='right', color='#C0392B', fontsize=9.5)
    axes[1].set_ylabel('Folding ratio（%）', fontsize=11)
else:
    # 10/14 簡報的兩組：每位平均 folding voxel 數，論文 VoxelMorph (CC) 的 19,077 畫成虛線
    top = max(PAPER_N, max(float(curve(e)[:, 2].max()) for e, *_ in EXPS)) * 1.12
    pl = axes[1].axhline(PAPER_N, ls='--', color='#C0392B', lw=1.2)
    axes[1].legend([pl], ['VoxelMorph (CC)（TMI 2019, Table I）19,077'], loc='upper right', fontsize=9.5, frameon=False)   # 字放圖例，不壓到曲線
    axes[1].set_ylabel('Folding voxels（每位平均）', fontsize=10)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '{:,.0f}'.format(v)))
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_title('Folding ratio（%|J| ≤ 0）' if SET == 'ablation' else 'Folding voxels（|J| ≤ 0）', fontsize=12.5, fontweight='bold')
axes[1].set_ylim(-0.02, 0.45 if SET == 'ablation' else top)

for ax in axes:
    ax.grid(alpha=.3)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'curve_%s.png' % TAG), dpi=130)
plt.close(fig)
print('->', os.path.join(OUT, 'curve_%s.png' % TAG))


# ── 2. 把既有的視覺化 PNG 堆成四版對照 ───────────────────────────────
def stack(kind, subject, out_name, xcrop=(0.035, 0.295), ycrop=(0.145, 1.0)):
    """四顆橫著排，只取軸狀切面那一格（原圖是三個切面並排，太寬塞不進投影片）。"""
    fig = plt.figure(figsize=FIGSIZE, facecolor=PAPER)
    for k, (exp, ep, lab, col) in enumerate(EXPS):
        p = os.path.join(ROOT, 'models', exp, 'vis_' + subject, '%s_%s_%s.png' % (kind, subject, ep))
        if not os.path.exists(p):
            raise SystemExit('[X] 找不到 %s' % p)
        im = mpimg.imread(p)
        h, w = im.shape[:2]
        im = im[int(h * ycrop[0]):int(h * ycrop[1]), int(w * xcrop[0]):int(w * xcrop[1])]
        ax = fig.add_axes(PANEL(k))
        ax.imshow(im)
        ax.axis('off')
        ax.text(0.5, -0.045, lab, transform=ax.transAxes, ha='center', va='top',
                fontsize=14, fontweight='bold', color=col)
    fig.savefig(os.path.join(OUT, out_name), dpi=130, facecolor=PAPER)
    plt.close(fig)
    print('->', os.path.join(OUT, out_name))


stack('jacobian', 'T054', 'jacobian_%s.png' % TAG)


# ── 3. 形變網格：重跑推論、自己畫清楚版 ─────────────────────────────────
def grid_slices(subject):
    """每顆模型對 subject 的形變場，取軸狀中間那片。快取裡有的直接讀，沒有的才算。"""
    cache = os.path.join(OUT, 'grid_%s_flow.npz' % subject)
    out = dict(np.load(cache)) if os.path.exists(cache) else {}
    todo = [e for e in EXPS if e[0] + '_u' not in out]
    if 'src' in out and not todo:
        return out

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
    out['src'] = src[:, :, mid]
    a_t = torch.from_numpy(atlas)[None, None].to(device)
    v_t = torch.from_numpy(vol)[None, None].to(device)
    for exp, ep, lab, col in todo:
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
    """四顆並排的形變網格。格線＝把 atlas 上的正方格子照形變場搬到受試者身上的樣子。"""
    d = grid_slices(subject)
    src = d['src']                                           # [左右, 前後]，imshow 用轉置畫
    # 只框腦的範圍，四周黑底裁掉，腦才放得大
    lr = np.nonzero(src.max(axis=1) > 0.02)[0]               # 第 0 軸 = 左右（畫在橫軸）
    ap = np.nonzero(src.max(axis=0) > 0.02)[0]               # 第 1 軸 = 前後（畫在縱軸）
    xl = (lr.min() - margin, lr.max() + margin)
    yl = (ap.min() - margin, ap.max() + margin)
    fig = plt.figure(figsize=FIGSIZE, facecolor=PAPER)
    for k, (exp, ep, lab, col) in enumerate(EXPS):
        u, v = d[exp + '_u'], d[exp + '_v']
        ax = fig.add_axes(PANEL(k))
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


grid_clear('T054', 'grid_%s.png' % TAG)
