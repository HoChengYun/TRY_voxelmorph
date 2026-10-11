# -*- coding: utf-8 -*-
"""10/14 簡報：mix_cascade（⑥ Step 1）與半監督 mix_exp8／9（⑦）的圖（2026-10-11，AI 帶回結果後）。
數字讀 deck_data.json（gather.py 的 cascade、semisup）；兩張受試者圖要重跑推論（GPU），結果快取在 deck_charts。

  1014_cascade_res.png   ⑥ cascade vs mix_exp6、mix_wide_vel：Dice、沒參與訓練之結構 Dice、影像 NCC、HD95、folding
  1014_semi_res.png      ⑦ γ = 0／0.5／5（＋affine）：同上五項（HD95 換成 SDlogJ）
  1014_semi_struct.png   ⑦ 每個結構相對 mix_exp6 的 ΔDice：訓練時使用的 17 種結構 ＋ 沒參與訓練的 3 種
  1014_semi_grid.png     ⑦ 形變網格（受試者空間）＋ log|J|（atlas 空間），四個模型同一冠狀切面
  1014_semi_heldout.png  ⑦ 沒參與訓練之結構放大：Accumbens、Inf-Lat-Vent（atlas 輪廓 vs 搬過去的標籤）
  1014_semi_warped.png   ⑦ 搬過去的影像 m∘φ：m、f、四個模型的 m∘φ 並排（上：整個冠狀切面；下：深部放大）——同論文 Fig. 10 的排法
                         （使用者 2026-10-11：「視覺化放一些和捏出來影像有關的照片」）

使用者 2026-10-11：「簡報加網格對照圖，除了網格對照圖視覺化還可以加其他的」。
🔴 受試者：mix_exp9 的「沒參與訓練之結構 Dice」下降量最接近 51 位中位數的那一位（不挑最誇張的；T054 下降 0.255，偏大）。
網格畫法同 2026-09-20_cross/make_compare.py 的 grid_clear（atlas 上的正方格子照形變場搬到受試者身上），切面改冠狀面；
模型用 ASD/arch.py 的 load_model（cascade 也讀得到）。
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
J = lambda *p: os.path.join(ROOT, *p)
D = json.load(open(os.path.join(HERE, 'deck_data.json'), encoding='utf-8'))
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#D9D9D2'
TEAL, BLUE, AMBER, RED, GRAY = '#0E7C7B', '#2F7FD0', '#D08C2B', '#B03A2E', '#9A9A93'
W = 12.13


def save(fig, name, dpi=130):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def clean(ax):
    ax.grid(alpha=.3, axis='x')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def dot_chart(name, rows, panels, h):
    """列 = 模型、欄 = 指標（同 make_charts.py 的 1014_metrics.png）。rows: [(標籤, 設定, 顏色, {指標: 值})]。
    panels: [(標題, 小字說明, 指標, 'dot'|'bar', (xmin, xmax), 格式, 參考線)]。標題兩行（指標名＋灰色說明），不會互相壓到。"""
    fig, axes = plt.subplots(1, len(panels), figsize=(W, h), facecolor=PAPER, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, top=1 - 0.62 / h, bottom=0.3 / h, wspace=0.14)
    yv = -np.arange(len(rows), dtype=float)
    for ax, (title, note, k, kind, xl, fmt, ref) in zip(axes, panels):
        span = xl[1] - xl[0]
        if ref is not None:
            ax.axvline(ref[0], color=MUTED, ls='--', lw=1.1)
            ax.text(ref[0] + 0.01 * span, yv[0] + 0.5, ref[1], ha='left', va='top', fontsize=8.5, color=MUTED)
        for yy, (lab, cfg, col, val) in zip(yv, rows):
            v = val.get(k)
            if v is None:
                ax.text(xl[0] + 0.02 * span, yy, '—', va='center', fontsize=10, color=MUTED)
                continue
            if kind == 'dot':
                ax.plot([xl[0], v], [yy, yy], color=col, lw=1, alpha=0.25)
                ax.plot(v, yy, 'o', color=col, ms=8)
            else:
                ax.barh(yy, v, height=0.6, color=col)
            inside = kind == 'bar' and v > 0.62 * span
            ax.text(v + (-0.015 if inside else (0.045 if kind == 'dot' else 0.02)) * span, yy, fmt(v), va='center',
                    ha='right' if inside else 'left', fontsize=10.5, color='white' if inside else INK)
        ax.set_xlim(*xl)
        ax.text(0, 1.0 + 0.36 / h, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=11.5, fontweight='bold',
                color=INK)
        if note:
            ax.text(0, 1.0 + 0.06 / h, note, transform=ax.transAxes, ha='left', va='bottom', fontsize=9.5, color=MUTED)
        ax.tick_params(axis='x', labelsize=9.5)
        ax.tick_params(axis='y', length=0)
        clean(ax)
    tr = axes[0].get_yaxis_transform()
    for yy, (lab, cfg, col, _) in zip(yv, rows):
        axes[0].text(-0.04, yy + 0.17, lab, transform=tr, ha='right', va='center', fontsize=11, fontweight='bold', color=INK)
        axes[0].text(-0.04, yy - 0.2, cfg, transform=tr, ha='right', va='center', fontsize=9.5, color=col)
    axes[0].set_yticks([])
    axes[0].set_ylim(yv[-1] - 0.75, yv[0] + 0.55)
    save(fig, name)


f2, f3, f4 = (lambda v: '%.2f' % v), (lambda v: '%.3f' % v), (lambda v: '%.4f' % v)
fN = lambda v: '0' if v == 0 else ('%.1f' % v if v < 10 else '{:,.0f}'.format(v))

# ── ⑥ cascade 結果 ────────────────────────────────────────────────────
C = D['cascade']
crow = [('mix_exp6', 'SVF · default width', TEAL), ('mix_wide_vel', 'SVF · 2× width', '#5BB8B6'),
        ('mix_cascade', 'SVF · 2 stages（cascade）', BLUE)]
rows = [(e, cfg, col, {'dice': C['dice'][e], 'held': C['heldout'][e], 'ncc': C['ncc'][e], 'hd95': C['hd95'][e],
                       'fold': C['fold_n'][e]}) for e, cfg, col in crow]
lo = lambda k, pad: min(r[3][k] for r in rows) - pad
hi = lambda k, pad: max(r[3][k] for r in rows) + pad
dot_chart('1014_cascade_res.png', rows, [
    ('(a) Dice ↑', '30 結構；affine %.3f' % D['baseline'], 'dice', 'dot', (lo('dice', 0.004), hi('dice', 0.006)), f3, None),
    ('(b) Dice ↑', '未參與訓練之結構；affine %.3f' % C['heldout']['affine'], 'held', 'dot',
     (lo('held', 0.012), hi('held', 0.014)), f3, None),
    ('(c) 影像 NCC ↑', r'$m\circ\phi$' + ' 與 atlas；affine %.3f' % C['ncc']['affine'], 'ncc', 'dot', (lo('ncc', 0.012), hi('ncc', 0.016)), f3, None),
    ('(d) HD95（mm）↓', '30 結構；affine %.2f' % C['hd95']['affine'], 'hd95', 'dot', (lo('hd95', 0.03), hi('hd95', 0.05)), f2, None),
    ('(e) Folding voxels ↓', '每位平均', 'fold', 'bar', (0, hi('fold', 0) * 1.45), fN, None),
], 2.85)

# ── ⑦ 半監督：五項 ─────────────────────────────────────────────────────
S = D['semisup']
srow = [('affine', '形變配準前', GRAY), ('mix_exp6', 'γ = 0（unsupervised）', TEAL), ('mix_exp8', 'γ = 0.5', AMBER),
        ('mix_exp9', 'γ = 5', RED)]
rows = [(e, cfg, col, {'dice': S[e]['dice'], 'held': S[e]['heldout'], 'ncc': S[e]['ncc'], 'sdlogj': S[e]['sdlogj'],
                       'fold': S[e]['fold_n']}) for e, cfg, col in srow]
dot_chart('1014_semi_res.png', rows, [
    ('(a) Dice ↑', '訓練時使用之 30 結構', 'dice', 'dot', (0.66, 0.99), f3, None),
    ('(b) Dice ↑', '未參與訓練之結構', 'held', 'dot', (0.40, 0.72), f3, (S['affine']['heldout'], 'affine')),
    ('(c) 影像 NCC ↑', r'$m\circ\phi$' + ' 與 atlas（局部，窗格 9）', 'ncc', 'dot', (0.06, 0.27), f3, None),
    ('(d) SDlogJ ↓', 'log|J| 之標準差', 'sdlogj', 'bar', (0, 1.45), f2, None),
    ('(e) Folding voxels ↓', '每位平均', 'fold', 'bar', (0, S['mix_exp9']['fold_n'] * 1.45), fN, None),
], 3.4)

# ── ⑦ 每個結構的 ΔDice（相對 mix_exp6）─────────────────────────────────────
EN = D.get('struct_en', {})
st8, st9 = S['mix_exp8']['struct'], S['mix_exp9']['struct']
names = sorted(st9, key=lambda n: st9[n])
fig = plt.figure(figsize=(W, 5.6), facecolor=PAPER)
ax = fig.add_axes([0.25, 0.1, 0.42, 0.83])
y = np.arange(len(names))
ax.barh(y + 0.2, [st8[n] for n in names], height=0.38, color=AMBER, label='γ = 0.5（mix_exp8）')
ax.barh(y - 0.2, [st9[n] for n in names], height=0.38, color=RED, label='γ = 5（mix_exp9）')
for yy, n in zip(y, names):
    ax.text(st9[n] + 0.006, yy - 0.2, '%+.3f' % st9[n], va='center', fontsize=9, color=RED)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(y)
ax.set_yticklabels(['%s  %s' % (n, EN.get(n, '')) for n in names], fontsize=10)
ax.set_xlim(-0.03, 0.43)
ax.set_xlabel('ΔDice 相對 γ = 0（mix_exp6）（左右平均）', fontsize=10.5)
ax.set_title('(a) 訓練時使用之結構（30 個，左右合併為 17 種）', loc='left', fontsize=11.5, fontweight='bold')
ax.legend(fontsize=10, frameon=False, loc='lower right')
clean(ax)
ax.grid(axis='y', alpha=0)
ax2 = fig.add_axes([0.76, 0.42, 0.22, 0.51])
hn = list(S['mix_exp8']['held_d'])
y2 = np.arange(len(hn))
d8 = [S['mix_exp8']['held_d'][n] for n in hn]
d9 = [S['mix_exp9']['held_d'][n] for n in hn]
ax2.barh(y2 + 0.2, d8, height=0.38, color=AMBER)
ax2.barh(y2 - 0.2, d9, height=0.38, color=RED)
for yy, a, b in zip(y2, d8, d9):
    ax2.text(a + (0.008 if a >= 0 else -0.008), yy + 0.2, '%+.3f' % a, va='center', ha='left' if a >= 0 else 'right', fontsize=9, color=AMBER)
    ax2.text(b - 0.008, yy - 0.2, '%+.3f' % b, va='center', ha='right', fontsize=9, color=RED)
ax2.axvline(0, color=INK, lw=1)
ax2.set_yticks(y2)
HL = {'Accumbens': 'Accumbens\n（26、58）', 'Inf-Lat-Vent': 'Inf-Lat-Vent\n（5、44）', '胼胝體': '胼胝體 CC\n（251～255）'}
ax2.set_yticklabels([HL.get(n, n) for n in hn], fontsize=10.5)
ax2.set_xlim(-0.45, 0.12)
ax2.set_title('(b) 未參與訓練之結構', loc='left', fontsize=11.5, fontweight='bold')
ax2.set_xlabel('ΔDice 相對 γ = 0', fontsize=10)
clean(ax2)
ax2.grid(axis='y', alpha=0)
fig.text(0.735, 0.30, 'FreeSurfer 有標註、但不在 30 個\n評估結構內者（括號內為標籤編號）', fontsize=9, color=MUTED,
         va='top', linespacing=1.5)
save(fig, '1014_semi_struct.png')


# ── ⑦ 受試者圖：推論（GPU，快取）──────────────────────────────────────────
def pick_subject():
    """mix_exp9 的「沒參與訓練之結構 Dice」下降量最接近中位數的受試者。"""
    rd = lambda p: {r['file'][:-4]: float(r['heldout_mean']) for r in csv.DictReader(open(p, encoding='utf-8'))}
    a, b = rd(J('models', 'mix_exp6', 'semisup_check_0190.csv')), rd(J('models', 'mix_exp9', 'semisup_check_0245.csv'))
    d = {k: b[k] - a[k] for k in a}
    med = float(np.median(list(d.values())))
    k = min(d, key=lambda x: abs(d[x] - med))
    print('   受試者 %s：沒參與訓練之結構 ΔDice %+.3f（51 位中位數 %+.3f）' % (k, d[k], med))
    return k, d[k], med


SUBJ, SUBJ_D, SUBJ_MED = pick_subject()
VIS = [('affine', '0', 'affine（形變配準前）', GRAY), ('mix_exp6', '0190', 'γ = 0（mix_exp6）', TEAL),
       ('mix_cascade', '0170', 'Cascade（mix_cascade）', BLUE), ('mix_exp8', '0250', 'γ = 0.5（mix_exp8）', AMBER),
       ('mix_exp9', '0245', 'γ = 5（mix_exp9）', RED)]
# (名稱, 標籤, 鄰近的訓練結構, 鄰近名稱, 'auto' = 只看單側, 放大半徑)
HELD_ROWS = [('Accumbens', (26, 58), (11, 50, 12, 51), 'Caudate、Putamen', None, None),
             ('Inf-Lat-Vent', (5, 44), (17, 53, 18, 54), 'Hippocampus、Amygdala', 'auto', 15)]


def infer():
    cache = os.path.join(OUT, 'semi_%s_cache.npz' % SUBJ)
    out = dict(np.load(cache)) if os.path.exists(cache) else {}
    todo = [v for v in VIS if v[0] != 'affine' and v[0] + '_moved' not in out]
    if 'src' in out and not todo:
        return out
    os.environ['NEURITE_BACKEND'] = 'pytorch'
    os.environ['VXM_BACKEND'] = 'pytorch'
    import torch
    sys.path.insert(0, J('ASD'))
    sys.path.insert(0, J('voxelmorph-code'))
    import voxelmorph as vxm
    from arch import load_model
    from orient import canonical_axes, to_ras, flow_to_ras
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    atlas = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
    aseg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
    d = np.load(J('data', 'mixed_preprocessed_v2', 'test', SUBJ + '.npz'))
    vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
    perm, flip = canonical_axes(aseg)
    R = lambda x: to_ras(x, perm, flip)
    out.update({'src': R(vol), 'atlas': R(atlas), 'aseg': R(aseg), 'affine_seg': R(seg)})
    a_t = torch.from_numpy(atlas)[None, None].to(dev)
    v_t = torch.from_numpy(vol)[None, None].to(dev)
    s_t = torch.from_numpy(seg.astype(np.float32))[None, None].to(dev)
    warp_nn = vxm.torch.layers.SpatialTransformer(atlas.shape, mode='nearest').to(dev)
    warp_lin = vxm.torch.layers.SpatialTransformer(atlas.shape, mode='bilinear').to(dev)
    for e, ep, _, _ in todo:
        model = load_model(J('models', e, ep + '.pt'), dev).to(dev)
        model.eval()
        with torch.no_grad():
            _, flow = model(v_t, a_t, registration=True)
            sw = warp_nn(s_t, flow)[0, 0].cpu().numpy()
            mv = warp_lin(v_t, flow)[0, 0].cpu().numpy()
        fl = flow[0].cpu().numpy()
        g = [[np.gradient(fl[c], axis=a) for a in range(3)] for c in range(3)]
        det = ((1 + g[0][0]) * ((1 + g[1][1]) * (1 + g[2][2]) - g[1][2] * g[2][1])
               - g[0][1] * (g[1][0] * (1 + g[2][2]) - g[1][2] * g[2][0])
               + g[0][2] * (g[1][0] * g[2][1] - (1 + g[1][1]) * g[2][0]))
        out[e + '_seg'] = R(np.round(sw).astype(np.int32))
        out[e + '_moved'] = R(mv.astype(np.float32))
        out[e + '_det'] = R(det.astype(np.float32))
        out[e + '_flow'] = flow_to_ras(fl, perm, flip).astype(np.float32)
        print('   %s：folding %d voxels' % (e, int((det <= 0).sum())))
        del model, flow
        if dev == 'cuda':
            torch.cuda.empty_cache()
    np.savez_compressed(cache, **out)
    return out


X = infer()
SRC, ATL, ASEG = X['src'], X['atlas'], X['aseg']
ycor = int(np.round(np.nonzero(np.isin(ASEG, (11, 50, 12, 51)))[1].mean()))      # 冠狀面：通過紋狀體中心


def crop_box(img2d, margin=6):
    xs = np.nonzero(img2d.max(axis=1) > 0.02)[0]
    zs = np.nonzero(img2d.max(axis=0) > 0.02)[0]
    return (xs.min() - margin, xs.max() + margin), (zs.min() - margin, zs.max() + margin)


# ── ⑦ 形變網格＋log|J|：四個模型（不含 affine）同一冠狀切面 ─────────────────────────
M4 = [v for v in VIS if v[0] != 'affine']
fig = plt.figure(figsize=(W, 6.0), facecolor=PAPER)
pw_, gap, x0 = 0.222, 0.012, 0.035
src2 = SRC[:, ycor, :]
xl, zl = crop_box(src2)
atl2 = ATL[:, ycor, :]
xa, za = crop_box(atl2)
cmap = plt.get_cmap('RdBu_r')
for k, (e, ep, lab, col) in enumerate(M4):
    fl = X[e + '_flow']
    u, w = fl[0][:, ycor, :], fl[2][:, ycor, :]                  # 左右、上下分量
    ax = fig.add_axes([x0 + k * (pw_ + gap), 0.5, pw_, 0.42])
    ax.imshow(src2.T * 0.7, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='bilinear')
    n0, n1 = u.shape
    for j in range(0, n1, 7):
        ax.plot(np.arange(n0) + u[:, j], j + w[:, j], color='#FFD23F', lw=0.9, alpha=0.95)
    for i in range(0, n0, 7):
        ax.plot(i + u[i, :], np.arange(n1) + w[i, :], color='#FFD23F', lw=0.9, alpha=0.95)
    ax.set_xlim(*xa)                                         # 跟下排同一個範圍（格線畫在 atlas 格點搬過去的位置）
    ax.set_ylim(*za)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(lab, fontsize=12, fontweight='bold', color=col)
    det = X[e + '_det'][:, ycor, :]
    fg = (ATL[:, ycor, :] > 0) | (ASEG[:, ycor, :] > 0)          # 腦室在 atlas 影像上是 0，用標籤補回來
    lj = np.where(fg, np.log(np.clip(det, 1e-3, None)), np.nan)
    ax = fig.add_axes([x0 + k * (pw_ + gap), 0.04, pw_, 0.42])
    ax.imshow(np.where(fg, atl2, np.nan).T, cmap='gray', origin='lower', vmin=0, vmax=1.6)
    im = ax.imshow(lj.T, cmap=cmap, origin='lower', vmin=-1, vmax=1, alpha=0.9, interpolation='nearest')
    zx, zz = np.nonzero(fg & (det <= 0))
    ax.plot(zx, zz, 's', color='black', ms=2.2, mew=0)
    ax.set_xlim(*xa)
    ax.set_ylim(*za)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0.03, 0.03, 'folding：%d voxels（此切面）' % len(zx), transform=ax.transAxes, fontsize=9, color=INK,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
cax = fig.add_axes([0.946, 0.08, 0.009, 0.34])
cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(labelsize=8.5)
cb.set_label(r'$\log|J_\phi|$', fontsize=10)
fig.text(0.006, 0.71, '形變網格', rotation=90, va='center', fontsize=11, fontweight='bold', color=INK)
fig.text(0.006, 0.25, r'$\log|J_\phi|$', rotation=90, va='center', fontsize=12, color=INK)
save(fig, '1014_semi_grid.png')


# ── ⑦ 沒參與訓練之結構放大 ──────────────────────────────────────────────
def dice_lr(a, b, labs):
    x, y = np.isin(a, labs), np.isin(b, labs)
    return 2.0 * (x & y).sum() / max(x.sum() + y.sum(), 1)


def contour(ax, mask, **kw):
    if mask.any():
        ax.contour(mask.T.astype(float), levels=[0.5], **kw)


SEGS = {'affine': X['affine_seg'], **{e: X[e + '_seg'] for e, *_ in M4}}
VH = [v for v in VIS if v[0] != 'mix_cascade']
# 2026-10-11 使用者：訓練時使用之鄰近結構與未參與訓練之結構都要畫，且各自都要有 atlas 的輪廓對照
#   線型 = 來源（虛線：atlas、實線：受試者經形變搬至 atlas）；顏色 = 類別（紅：未參與訓練、青：訓練時使用）
NB_SUBJ, NB_ATL = '#2BB5AE', '#B8F3EF'
fig = plt.figure(figsize=(W, 6.1), facecolor=PAPER)
for r, (name, labs, nb, nb_name, side, rad) in enumerate(HELD_ROWS):
    if side == 'auto':                     # 很細的結構：只對準單側（atlas 上 voxel 較多的那側），放大一點
        one = max(labs, key=lambda l: int((ASEG == l).sum()))
        yc = int(np.round(np.nonzero(ASEG == one)[1].mean()))
        xs, zs = np.nonzero(ASEG[:, yc, :] == one)
    else:
        yc = int(np.round(np.nonzero(np.isin(ASEG, labs))[1].mean()))
        xs, zs = np.nonzero(np.isin(ASEG[:, yc, :], labs))
    m_at = np.isin(ASEG[:, yc, :], labs)
    cx, cz = xs.mean(), zs.mean()
    half_x = rad if rad else max(30, (xs.max() - xs.min()) / 2 + 14)
    half_z = rad if rad else 24
    for k, (e, ep, lab, col) in enumerate(VH):
        ax = fig.add_axes([0.155 + k * 0.204, 0.555 - r * 0.45, 0.198, 0.385])
        ax.imshow(ATL[:, yc, :].T, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='bilinear')
        sw = SEGS[e][:, yc, :]
        contour(ax, np.isin(sw, nb), colors=[NB_SUBJ], linewidths=1.4)                        # 訓練結構：受試者
        contour(ax, np.isin(ASEG[:, yc, :], nb), colors=[NB_ATL], linewidths=1.2, linestyles='--')   # 訓練結構：atlas
        ax.imshow(np.ma.masked_where(~np.isin(sw, labs), np.ones_like(sw)).T, cmap=ListedColormap([RED]), origin='lower',
                  alpha=0.45, vmin=0, vmax=1)
        contour(ax, np.isin(sw, labs), colors=[RED], linewidths=1.6)                           # 未參與訓練：受試者
        contour(ax, m_at, colors=['white'], linewidths=1.6, linestyles='--')                   # 未參與訓練：atlas
        ax.set_xlim(cx - half_x, cx + half_x)
        ax.set_ylim(cz - half_z, cz + half_z)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(2)
        dsc = dice_lr(SEGS[e], ASEG, labs)
        ax.text(0.03, 0.04, '%s Dice %.2f' % (name, dsc), transform=ax.transAxes, fontsize=10, fontweight='bold', color=RED,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
        if r == 0:
            ax.set_title(lab, fontsize=11.5, fontweight='bold', color=col)
    fig.text(0.145, 0.8 - r * 0.45, '%s\n（%s）' % (name, '、'.join(map(str, labs))), ha='right', va='center',
             fontsize=11, fontweight='bold', color=RED, linespacing=1.4)
    fig.text(0.145, 0.68 - r * 0.45, '鄰近之訓練結構：\n%s' % nb_name, ha='right', va='center', fontsize=9.5, color=NB_SUBJ,
             linespacing=1.4)
    if side == 'auto':
        fig.text(0.145, 0.59 - r * 0.45, '（僅顯示單側）', ha='right', va='center', fontsize=9, color=MUTED)
handles = [Line2D([], [], color='white', lw=1.6, ls='--', label='未參與訓練之結構：atlas'),
           Patch(fc=RED, ec=RED, alpha=0.6, label='未參與訓練之結構：受試者（經形變搬至 atlas）'),
           Line2D([], [], color=NB_ATL, lw=1.4, ls='--', label='訓練時使用之結構：atlas'),
           Line2D([], [], color=NB_SUBJ, lw=1.6, label='訓練時使用之結構：受試者（經形變搬至 atlas）')]
leg = fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9.5, frameon=True, bbox_to_anchor=(0.56, -0.005),
                 columnspacing=2.5)
leg.get_frame().set_facecolor('#2B2F31')
for t in leg.get_texts():
    t.set_color('white')
save(fig, '1014_semi_heldout.png')
json.dump({'subject': SUBJ, 'd': SUBJ_D, 'median': SUBJ_MED, 'ycor': ycor},
          open(os.path.join(OUT, 'semi_subject.json'), 'w', encoding='utf-8'))
print('   受試者資訊 -> semi_subject.json')


# ── ⑦ 搬過去的影像 m∘φ（同論文 Fig. 10：m、f、m∘φ 並排）────────────────────────────────
def ncc_of(e):
    p = J('models', 'semisup_check_affine.csv') if e == 'affine' else J('models', e, 'semisup_check_%s.csv' % CHK_EP[e])
    return {r['file'][:-4]: float(r['ncc']) for r in csv.DictReader(open(p, encoding='utf-8'))}[SUBJ]


CHK_EP = {'mix_exp6': '0190', 'mix_cascade': '0170', 'mix_exp8': '0250', 'mix_exp9': '0245'}
COLS = [('m（受試者，affine 後）', SRC, None, INK), ('f（atlas）', ATL, None, INK)] +        [(r'$m\circ\phi$' + '：' + lab.split('（')[0], X[e + '_moved'], e, col) for e, ep, lab, col in M4]
mid = np.isin(ASEG[:, ycor, :], (11, 50, 12, 51, 13, 52, 4, 43))         # 放大範圍：紋狀體＋側腦室
zx, zz = np.nonzero(mid)
zbox = ((zx.min() - 8, zx.max() + 8), (zz.min() - 10, zz.max() + 10))
fig = plt.figure(figsize=(W, 4.95), facecolor=PAPER)
n = len(COLS)
pw2 = 0.158
for k, (lab, vol, e, col) in enumerate(COLS):
    sl = vol[:, ycor, :]
    for r, (box, yb, hh) in enumerate(((crop_box(ATL[:, ycor, :]), 0.47, 0.43), (zbox, 0.03, 0.41))):
        ax = fig.add_axes([0.012 + k * (pw2 + 0.006), yb, pw2, hh])
        ax.imshow(sl.T, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='bilinear')
        if r == 0:
            ax.add_patch(plt.Rectangle((zbox[0][0], zbox[1][0]), zbox[0][1] - zbox[0][0], zbox[1][1] - zbox[1][0],
                                       fill=False, ec='#FFD23F', lw=1.2))
        ax.set_xlim(*box[0])
        ax.set_ylim(*box[1])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col if e else RULE)
            sp.set_linewidth(2 if e else 1)
        if r == 0:
            ax.set_title(lab, fontsize=10.5, fontweight='bold', color=col)
            if e is not None or k == 0:
                ax.text(0.5, -0.02, 'NCC %.3f' % ncc_of(e or 'affine'), transform=ax.transAxes, ha='center', va='top',
                        fontsize=9.5, color=INK)
save(fig, '1014_semi_warped.png')



# ── ⑦ 訓練時使用之結構：五個模型的輪廓圖並排（ASD/visualize_dice.py 的 contours_<受試者>_<epoch>.png）──────────────
# 2026-10-11 使用者：「要不要每個 model 的 contours 圖放在一起比較」。原圖：上排 affine（BEFORE）、下排配準後（AFTER），
#   三欄是矢狀、冠狀、軸狀；背景與實線 = atlas、虛線 = 受試者。這裡取 affine 一次 ＋ 四個模型的 AFTER，切面 × 模型排成一張。
#   要先跑：python ASD\visualize_dice.py --model models\<實驗>\<epoch>.pt --test-dir data\mixed_preprocessed_v2\test
#           --subject <受試者> --out-dir models\<實驗>\vis_<受試者>（四個模型各一次）
from PIL import Image


def panels_of(p):
    """回傳 {(列, 欄): 裁好的腦部影像}，列 0 = affine、1 = 配準後；欄 0～2 = 矢狀、冠狀、軸狀。另回傳圖例。"""
    im = np.asarray(Image.open(p).convert('RGB'))
    dark = im.astype(int).sum(2) < 60
    def runs(v, minlen=200):
        idx = np.nonzero(v > 0.05)[0]
        out, s = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                out.append((s, a))
                s = b
        out.append((s, idx[-1]))
        return [r for r in out if r[1] - r[0] >= minlen]
    rr, cc = runs(dark.mean(1)), runs(dark.mean(0))
    if len(rr) != 2 or len(cc) != 3:
        raise SystemExit('[X] %s 的版面不是 2 × 3 格：%s %s' % (p, rr, cc))
    out = {}
    for i, (y0, y1) in enumerate(rr):
        for j, (x0, x1) in enumerate(cc):
            sub = im[y0:y1 + 1, x0:x1 + 1]
            ys, xs = np.nonzero(sub.astype(int).sum(2) < 60)            # 黑底面板本身（帶裡上下可能有白邊、標題字）
            out[(i, j)] = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    leg = im[rr[1][1] + 25:, :]                                       # 圖例（最下面那條），上下左右的白邊裁掉
    ink = leg.astype(int).sum(2) < 600
    ky, kx = np.nonzero(ink.mean(1) > 0.002)[0], np.nonzero(ink.mean(0) > 0.0)[0]
    if len(ky) and len(kx):
        leg = leg[max(ky.min() - 6, 0):ky.max() + 6, max(kx.min() - 10, 0):kx.max() + 10]
    return out, leg


def colored_box(imgs, margin=28):
    """幾張同版面的面板裡，彩色輪廓（各結構）的聯集外框：所有模型裁同一塊，才能互相比較。"""
    ys, xs = [], []
    for a in imgs:
        a = a.astype(int)
        y, x = np.nonzero(a.max(2) - a.min(2) > 70)
        ys.append(y)
        xs.append(x)
    ys, xs = np.concatenate(ys), np.concatenate(xs)
    h, w = imgs[0].shape[:2]
    return max(ys.min() - margin, 0), min(ys.max() + margin, h), max(xs.min() - margin, 0), min(xs.max() + margin, w)


CT = [('affine', None, 'affine（形變配準前）', GRAY), ('mix_exp6', '0190', 'γ = 0（mix_exp6）', TEAL),
      ('mix_cascade', '0170', 'Cascade（mix_cascade）', BLUE), ('mix_exp8', '0250', 'γ = 0.5（mix_exp8）', AMBER),
      ('mix_exp9', '0245', 'γ = 5（mix_exp9）', RED)]
paths = {e: J('models', e, 'vis_' + SUBJ, 'contours_%s_%s.png' % (SUBJ, ep)) for e, ep, _, _ in CT if ep}
if all(os.path.exists(p) for p in paths.values()):
    P = {e: panels_of(p) for e, p in paths.items()}
    VIEWS = ['矢狀面', '冠狀面', '軸狀面']
    pan = lambda e, j: P['mix_exp6'][0][(0, j)] if e == 'affine' else P[e][0][(1, j)]   # affine 取 BEFORE（四張都一樣）
    JV = [1, 2]                                      # 只放冠狀、軸狀（三個切面五欄放不下一頁；矢狀面在 vis_<受試者> 原圖裡）
    BOX = {j: colored_box([pan(e, j) for e, *_ in CT]) for j in JV}                  # 每個切面：五欄裁同一塊（放大深部結構）
    x0, cw = 0.055, 0.189
    cw_in = cw * W
    rows_in = {j: cw_in * (BOX[j][1] - BOX[j][0]) / (BOX[j][3] - BOX[j][2]) for j in JV}   # 每列高度（吋），寬固定、高照比例
    head, leg_in = 0.45, 0.42
    FH = head + sum(rows_in.values()) + 0.05 * len(JV) + leg_in
    fig = plt.figure(figsize=(W, FH), facecolor=PAPER)
    ytop = FH - head
    for j in JV:
        y0b, y1b, x0b, x1b = BOX[j]
        for k, (e, ep, lab, col) in enumerate(CT):
            img = pan(e, j)[y0b:y1b, x0b:x1b]
            ax = fig.add_axes([x0 + k * cw, (ytop - rows_in[j]) / FH, cw - 0.006, rows_in[j] / FH])
            ax.imshow(img, interpolation='lanczos')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(col)
                sp.set_linewidth(2 if e != 'affine' else 1.2)
            if j == JV[0]:
                ax.set_title(lab, fontsize=11.5, fontweight='bold', color=col)
        fig.text(x0 - 0.008, (ytop - rows_in[j] / 2) / FH, VIEWS[j], rotation=90, ha='right', va='center', fontsize=11,
                 fontweight='bold', color=INK)
        ytop -= rows_in[j] + 0.05
    lg = P['mix_exp6'][1]
    lw_frac = min(0.8, leg_in / FH * lg.shape[1] / lg.shape[0] * FH / W)                # 圖例照原比例、最寬 80%
    lax = fig.add_axes([0.5 - lw_frac / 2 + 0.03, 0.005, lw_frac, (leg_in - 0.03) / FH])
    lax.imshow(lg, interpolation='lanczos')
    lax.axis('off')
    save(fig, '1014_semi_contours.png', dpi=150)
else:
    print('[!] 缺 vis_%s 的 contours 圖，1014_semi_contours.png 沒畫：%s' % (SUBJ, [p for p in paths.values() if not os.path.exists(p)]))
