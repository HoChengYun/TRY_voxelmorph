# -*- coding: utf-8 -*-
"""10/14 簡報「頭頂殘留怎麼量」四頁（2026-10-06）：每一步一張圖＋一個公式區塊。
-> models/deck_charts/1014_method_{1..4}.png（圖）、1014_method_eq{1..4}.png（編號公式＋「其中」符號說明）

使用者：「放進簡報取代第 12 頁、一定要放公式、希望可以和 paper 一樣好閱讀」。
- 公式用 LaTeX 的字型（mathtext 的 Computer Modern）畫成圖，編號 (1)～(5) 靠右，下面用「其中」解釋每個符號
- 圖的大小就是投影片上的大小（寬 12.13 吋），字不會被縮小
- 符號沿用使用者已經看懂的寫法：z_top、τ、t、V、residue；# ＝ 數有幾個、|V| ＝ V 裡有幾根
例子是 sub-0043（test 裡頭頂殘留最多的一位），第 87 片，吸管 A（x=74，乾淨）、B（x=101，6 mm、中間隔一格暗的）：
這兩根每一格都離門檻至少 0.04，不會卡在邊緣看糊塗。
算法照 ASD/check_skullstrip.py 的 measure_top；圖上的數字都從資料算，並跟 measure_top 的結果核對（assert）。
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'ASD'))
from check_skullstrip import find, measure_top, VERTEX_BAND

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'          # 公式用 LaTeX 的字型，跟 paper 一樣
OUT = os.path.join(ROOT, 'models', 'deck_charts')
DATA = os.path.join(ROOT, 'data', 'mixed_preprocessed_v2')
W, DPI = 12.13, 200                              # 投影片上圖的寬度（吋）
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
BLUE_C, RED_C, DARK_C, YEL, CYAN, RUST, VBLUE = '#8EB8F0', '#E8473A', '#2E2E2E', '#FFD23F', '#4FC3F7', '#A34F1B', '#2F7FD0'
BLUE, GREEN = np.array([0.30, 0.55, 0.95]), np.array([0.20, 0.72, 0.35])


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def titles(fig, items, size=12.5):
    for x, t in items:
        fig.text(x, 0.975, t, ha='center', va='top', fontsize=size, fontweight='bold', color=INK, linespacing=1.4)


# ── 資料 ─────────────────────────────────────────────────────────────
EX, PY, STRAWS = 'sub-0043', 87, [('A', 74), ('B', 101)]
d = np.load(find(DATA, EX)[0])
vol, seg = d['vol'], d['seg']
m, tissue, thick, has = measure_top(vol, seg)
nx, ny, nz = vol.shape
brain = seg > 0
ztop = np.where(has, (brain * np.arange(nz)[None, None, :]).max(axis=2), -1)
ctx = np.isin(seg, [3, 42])
gm = float(np.median(vol[ctx]))
tau = 0.5 * gm
water = float(np.median(vol[np.isin(seg, [4, 43])]))       # 側腦室＝腦脊髓液（水）
wm = float(np.median(vol[np.isin(seg, [2, 41])]))
zmax = int(ztop[has].max())
zcut = zmax - VERTEX_BAND
vertex = has & (ztop >= zcut)
tV = thick[vertex].astype(int)
mean = tV.sum() / vertex.sum()
assert abs(mean - m['top_vertex_mm']) < 0.005                 # 跟 measure_top 一樣
assert all(vertex[x, PY] for _, x in STRAWS)
ok = has[:, PY]
top = int(ztop[ok, PY].max())
xs = np.arange(nx)
EXT = [-0.5, nx - 0.5, -0.5, nz - 0.5]
XL = (40, 150)
img = vol[:, PY, :].T
fmt = lambda n: format(int(n), ',')

# test 51 人各自的門檻（「每個人用自己的皮質」）
taus = []
for p in sorted(glob.glob(os.path.join(DATA, 'test', '*.npz'))):
    dd = np.load(p)
    taus.append(0.5 * float(np.median(dd['vol'][np.isin(dd['seg'], [3, 42])])))

# 「為什麼是 25 mm」（2026-10-06 使用者問）：25 是 09-22 設計這個量法時定的，沒有調過。
# 換成別的範圍，170 人「頭頂殘留厚度 vs 皮質 Dice 進步」的相關會不會變（同 gather.py 的 residue_mm）
from scipy.stats import spearmanr
from skullstrip_label_dice_pooled import load as load_pooled
BANDS = [10, 15, 20, 25, 30, 40, 60]
# 門檻也一起試（2026-10-06 使用者圈出幾塊「看起來是殘留、卻沒變紅」的地方：亮度 0.20～0.30，比門檻暗一點；
# 使用者決定維持 0.5 倍，但在第 13 頁寫出換門檻結論一樣）：τ ＝ 皮質中位數 × f
FACTORS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
rows = load_pooled()
cx = lambda row: float(np.mean([float(row['label_%d' % l]) for l in (3, 42)]))
gain = [cx(r['after']) - cx(r['before']) for r in rows]
band_x = {b: [] for b in BANDS}
band_n = {b: [] for b in BANDS}
tau_x = {f: [] for f in FACTORS}
for r in rows:
    dd = np.load(r['npz'])
    vv, sg = dd['vol'], dd['seg']
    _, _, th, hs = measure_top(vv, sg)
    bb = sg > 0
    zz = np.arange(bb.shape[2])[None, None, :]
    zt = np.where(hs, (bb * zz).max(axis=2), -1)
    for b in BANDS:
        v = hs & (zt >= zt[hs].max() - b)
        band_x[b].append(float(th[v].mean()))
        band_n[b].append(int(v.sum()))
    above = (zz > zt[:, :, None]) & hs[:, :, None]
    v25 = hs & (zt >= zt[hs].max() - VERTEX_BAND)
    g = float(np.median(vv[np.isin(sg, [3, 42])]))
    for f in FACTORS:
        tau_x[f].append(float((above & (vv >= f * g)).sum(axis=2)[v25].mean()))
band_r = {b: spearmanr(band_x[b], gain) for b in BANDS}
tau_r = {f: spearmanr(tau_x[f], gain) for f in FACTORS}
assert abs(tau_r[0.5][0] - band_r[VERTEX_BAND][0]) < 1e-9          # f = 0.5 就是現在的算法
with open(os.path.join(ROOT, 'models', 'skullstrip_check', 'top_tau_sensitivity.csv'), 'w', encoding='utf-8') as f_:
    f_.write('tau_factor,median_residue_mm,spearman_r,p,n\n')
    for f in FACTORS:
        f_.write('%.2f,%.3f,%.4f,%.3g,%d\n' % (f, np.median(tau_x[f]), tau_r[f][0], tau_r[f][1], len(rows)))
trmin, trmax = min(tau_r[f][0] for f in FACTORS), max(tau_r[f][0] for f in FACTORS)
assert abs(band_r[VERTEX_BAND][0] - spearmanr([float(r['res']['top_vertex_mm']) for r in rows], gain)[0]) < 0.01
with open(os.path.join(ROOT, 'models', 'skullstrip_check', 'top_band_sensitivity.csv'), 'w', encoding='utf-8') as f:
    f.write('band_mm,mean_columns,median_residue_mm,spearman_r,p,n\n')
    for b in BANDS:
        f.write('%d,%.0f,%.3f,%.4f,%.3g,%d\n' % (b, np.mean(band_n[b]), np.median(band_x[b]), band_r[b][0], band_r[b][1], len(rows)))
rmin, rmax = min(band_r[b][0] for b in BANDS), max(band_r[b][0] for b in BANDS)
pmax = max(band_r[b][1] for b in BANDS)


def slice_rgb(overlay=None, color=None):
    rgb = np.dstack([img] * 3)
    if overlay is not None:
        o = overlay[:, PY, :].T
        rgb[o] = 0.45 * rgb[o] + 0.55 * color
    return rgb


def show_slice(ax, rgb, zlim=None):
    ax.imshow(rgb, origin='lower', extent=EXT, interpolation='nearest')
    ax.set_xlim(*XL)
    ax.set_ylim(*(zlim or (top - 40, top + 8)))
    ax.axis('off')


# ── 圖 ①：找腦的頂邊 ───────────────────────────────────────────────────
fig = plt.figure(figsize=(W, 3.6), facecolor=PAPER)
pos = [[0.005, 0.06, 0.235, 0.66], [0.25, 0.06, 0.235, 0.66], [0.505, 0.03, 0.215, 0.75], [0.745, 0.06, 0.235, 0.66]]
ax = [fig.add_axes(p) for p in pos]
blue = slice_rgb(brain, BLUE)
show_slice(ax[0], slice_rgb())
show_slice(ax[1], blue)
ZX0, ZW = 95, 14                                            # 放大吸管 B 附近 14 排
zw = ztop[ZX0:ZX0 + ZW, PY]
ZZ0 = int(zw.min()) - 3
ZH = int(zw.max()) - ZZ0 + 5
ax[1].add_patch(Rectangle((ZX0 - 0.5, ZZ0 - 0.5), ZW, ZH, fill=False, ec=CYAN, lw=1.8))
a = ax[2]
a.imshow(blue[ZZ0:ZZ0 + ZH, ZX0:ZX0 + ZW], origin='lower', extent=[-0.5, ZW - 0.5, -0.5, ZH - 0.5], interpolation='nearest')
for i in range(ZW + 1):
    a.axvline(i - 0.5, color='#777777', lw=0.5)
for j in range(ZH + 1):
    a.axhline(j - 0.5, color='#777777', lw=0.5)
for i in range(ZW):
    a.add_patch(Rectangle((i - 0.5, int(zw[i]) - ZZ0 - 0.5), 1, 1, fill=False, ec=YEL, lw=2.5))
for i in (1, 6, 12):                                        # 從最上面往下，停在第一個藍格子
    a.annotate('', xy=(i, int(zw[i]) - ZZ0 + 0.55), xytext=(i, ZH - 0.6), arrowprops=dict(arrowstyle='->', color='white', lw=2))
a.set_xlim(-0.5, ZW - 0.5)
a.set_ylim(-0.5, ZH - 0.5)
a.set_aspect('equal')
a.axis('off')
show_slice(ax[3], blue)
ax[3].plot(xs[ok], ztop[ok, PY], color=YEL, lw=2)
ax[3].text(47, ztop[47, PY] + 5, r'$z_{\mathrm{top}}$', color=YEL, fontsize=15, ha='center', va='bottom',
           bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.6))
titles(fig, [(0.1225, '圖一　原始影像\n（頭頂，從正面切開）'), (0.3675, '圖二　藍＝FreeSurfer\n標成「腦」的格子'),
             (0.6125, '圖三　放大圖二的框：每一直排\n從上往下，第一個藍格子（黃框）'),
             (0.8625, '圖四　黃框連起來＝黃線\n＝腦的頂邊 ' + r'$z_{\mathrm{top}}$')])
save(fig, '1014_method_1.png')

# ── 圖 ②：「亮」的門檻 ─────────────────────────────────────────────────
fig = plt.figure(figsize=(W, 3.55), facecolor=PAPER)       # 這頁公式多一行（換門檻的說明），圖矮一點
a1 = fig.add_axes([0.005, 0.06, 0.30, 0.66])
a2 = fig.add_axes([0.37, 0.20, 0.24, 0.50])
a3 = fig.add_axes([0.665, 0.30, 0.32, 0.22])
show_slice(a1, slice_rgb(ctx, GREEN))
a2.hist(vol[ctx], bins=60, color='#B9BDB9')
a2.axvline(gm, color=INK, lw=1.8)
a2.axvline(tau, color=RUST, lw=2.6)
h = a2.get_ylim()[1]
a2.text(gm + 0.02, h * 0.97, '中位數\n%.2f' % gm, ha='left', va='top', fontsize=11.5, color=INK)
a2.text(tau - 0.02, h * 0.97, '一半\n' + r'$\tau$' + ' ＝ %.2f' % tau, ha='right', va='top', fontsize=12, fontweight='bold', color=RUST)
a2.set_xlabel('亮度 ' + r'$I$' + '（0＝全黑，1＝最亮）', fontsize=11)
a2.set_yticks([])
a2.tick_params(labelsize=10.5)
for sp in ('top', 'right', 'left'):
    a2.spines[sp].set_visible(False)
a3.imshow(np.linspace(0, 1, 256)[None, :], cmap='gray', extent=[0, 1, 0, 1], aspect='auto', vmin=0, vmax=1)
for x, name in ((0.0, '背景'), (water, '水（腦室）'), (gm, '皮質'), (wm, '白質')):
    a3.plot([x, x], [1.0, 1.25], color=INK, lw=1.3, clip_on=False)
    a3.text(x, 1.3, '%s\n%.2f' % (name, x), ha='center', va='bottom', fontsize=11, color=INK, clip_on=False)
a3.plot([tau, tau], [-0.3, 1.0], color=RUST, lw=3, clip_on=False)
a3.text(tau, -0.35, r'$\tau$' + ' ＝ %.2f' % tau, ha='center', va='top', fontsize=13, fontweight='bold', color=RUST, clip_on=False)
a3.text(tau / 2, 0.5, '比 ' + r'$\tau$' + ' 暗\n不算', ha='center', va='center', fontsize=10.5, color='white')
a3.text(0.66, 0.5, '至少 ' + r'$\tau$' + ' 這麼亮＝有東西，算', ha='center', va='center', fontsize=11.5, fontweight='bold', color=INK)
a3.set_xlim(-0.02, 1.02)
a3.set_ylim(0, 1)
a3.set_xticks([])
a3.set_yticks([])
for sp in a3.spines.values():
    sp.set_visible(False)
titles(fig, [(0.155, '圖一　綠＝FreeSurfer 標成\n大腦皮質的格子（cortex）'),
             (0.49, '圖二　cortex 每一格的亮度\n取中位數，再取一半 ＝ ' + r'$\tau$'),
             (0.825, '圖三　' + r'$\tau$' + ' 夾在「水」和「皮質」中間')])
save(fig, '1014_method_2.png')

# ── 圖 ③：每根吸管數幾格 ────────────────────────────────────────────────
BELOW, ABOVE = 2, 10
fig = plt.figure(figsize=(W, 3.6), facecolor=PAPER)


def straws_slice(a, red):
    rgb = np.dstack([img] * 3)
    if red:
        rgb[tissue[:, PY, :].T] = [0.91, 0.28, 0.23]
    a.imshow(rgb, origin='lower', extent=EXT, interpolation='nearest')
    a.plot(xs[ok], ztop[ok, PY], color=YEL, lw=1.8)
    for name, x in STRAWS:
        zt = ztop[x, PY]
        a.add_patch(Rectangle((x - 0.5, zt - BELOW + 0.5), 1, BELOW + ABOVE, fill=False, ec=CYAN, lw=1.6))
        a.text(x, zt + ABOVE + 1.5, name, color=CYAN, ha='center', va='bottom', fontsize=13, fontweight='bold')
    a.set_xlim(*XL)
    a.set_ylim(top - 40, top + ABOVE + 7)
    a.axis('off')


straws_slice(fig.add_axes([0.005, 0.06, 0.30, 0.68]), red=False)
for k, (name, x) in enumerate(STRAWS):
    a = fig.add_axes([0.335 + k * 0.13, 0.16, 0.08, 0.60])
    zt = ztop[x, PY]
    n, gap = 0, None
    for i, z in enumerate(range(zt - BELOW + 1, zt + ABOVE + 1)):
        v = float(vol[x, PY, z])
        if z <= zt:
            fc, tc = (BLUE_C, INK) if seg[x, PY, z] > 0 else ('#D9D9D2', INK)
        elif v >= tau:
            fc, tc = RED_C, 'white'
            n += 1
        else:
            fc, tc = DARK_C, '#BBBBBB'
            if n and gap is None and any(vol[x, PY, zz] >= tau for zz in range(z + 1, zt + ABOVE + 1)):
                gap = i
        a.add_patch(Rectangle((0, i), 2.2, 1, fc=fc, ec=PAPER, lw=1.2))
        a.text(1.1, i + 0.5, '%.2f' % v, ha='center', va='center', fontsize=9.5, color=tc, fontweight='bold')
    assert n == int(thick[x, PY])                         # 跟 measure_top 一樣
    a.plot([-0.2, 2.4], [BELOW, BELOW], color=YEL, lw=3)
    if gap is not None:
        a.text(2.5, gap + 0.5, '← 暗的不算，\n　上面亮的照算', ha='left', va='center', fontsize=10, color=INK, clip_on=False)
    a.set_xlim(-0.4, 2.6)
    a.set_ylim(-0.2, BELOW + ABOVE + 0.2)
    a.set_aspect('equal')
    a.axis('off')
    a.text(1.1, BELOW + ABOVE + 0.4, '吸管 ' + name, ha='center', va='bottom', fontsize=12.5, fontweight='bold', color=CYAN)
    a.text(1.1, -0.5, '紅 %d 格 → ' % n + r'$t=%d$' % n, ha='center', va='top', fontsize=12.5, fontweight='bold',
           color=RED_C if n else INK)
straws_slice(fig.add_axes([0.685, 0.06, 0.30, 0.68]), red=True)
titles(fig, [(0.155, '圖一　挑兩根吸管 A、B'),
             (0.475, '圖二　放大：黃線以上，亮度至少 ' + r'$\tau$' + ' 的格子（紅）數一數'),
             (0.835, '圖三　每一根吸管都這樣數\n紅的＝沒切乾淨的東西')])
fig.text(0.475, 0.89, '藍＝腦　紅＝至少 ' + r'$\tau$' + '（算）　黑＝比 ' + r'$\tau$' + ' 暗（不算）　一格＝1 mm',
         ha='center', va='top', fontsize=10.5, color=MUTED)
save(fig, '1014_method_3.png')

# ── 圖 ④：頭頂那一塊取平均 ──────────────────────────────────────────────
fig = plt.figure(figsize=(W, 2.95), facecolor=PAPER)      # 這頁公式多一行（25 mm 的說明），圖矮一點
a1 = fig.add_axes([0.005, 0.03, 0.33, 0.70])
a2 = fig.add_axes([0.36, 0.02, 0.22, 0.74])
a3 = fig.add_axes([0.655, 0.18, 0.33, 0.53])
a1.imshow(img, origin='lower', cmap='gray', vmin=0, vmax=1, extent=EXT)
inV = vertex[:, PY]
a1.plot(xs[ok & ~inV], ztop[ok & ~inV, PY], 's', color=YEL, ms=2.2)
a1.plot(xs[ok & inV], ztop[ok & inV, PY], 's', color=VBLUE, ms=2.2)
a1.axhline(zmax, color='white', ls='--', lw=1.2)
a1.axhline(zcut, color=CYAN, ls='--', lw=1.8)
a1.annotate('', xy=(46, zcut), xytext=(46, zmax), arrowprops=dict(arrowstyle='<->', color='white', lw=1.4))
a1.text(48, (zmax + zcut) / 2, '25 mm', color='white', ha='left', va='center', fontsize=11.5, fontweight='bold')
a1.text(148, zmax + 1, '最高點（頭頂）', color='white', ha='right', va='bottom', fontsize=10.5)
a1.text(148, zcut + 1, '頭頂往下 25 mm', color=CYAN, ha='right', va='bottom', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.75))
for name, x in STRAWS:
    a1.plot([x, x], [ztop[x, PY] + 1, ztop[x, PY] + 8.5], color=CYAN, lw=1.4)
    a1.text(x, ztop[x, PY] + 9, name, color=CYAN, ha='center', va='bottom', fontsize=12.5, fontweight='bold')
a1.set_xlim(*XL)
a1.set_ylim(zcut - 30, zmax + 9)
a1.axis('off')
idx = np.argwhere(has)
a2.imshow(np.where(has, 1.0, np.nan).T, origin='lower', cmap='Greys', vmin=0, vmax=3)
a2.imshow(np.where(vertex, 1.0, np.nan).T, origin='lower', cmap='Blues', vmin=0, vmax=1.3)
a2.axhline(PY, color=INK, ls=':', lw=1.1)
for name, x in STRAWS:
    a2.plot(x, PY, 'o', ms=6.5, mfc=CYAN, mec=INK, mew=1)
    a2.text(x, PY + 4, name, color=INK, ha='center', va='bottom', fontsize=11.5, fontweight='bold')
cy, cx = np.argwhere(vertex.T).mean(axis=0)
a2.text(cx, cy + 22, r'$V$', color='white', ha='center', va='center', fontsize=22, fontweight='bold')
a2.set_xlim(idx[:, 0].min() - 3, idx[:, 0].max() + 3)
a2.set_ylim(idx[:, 1].min() - 3, idx[:, 1].max() + 3)
a2.axis('off')
vals, cnt = np.unique(tV, return_counts=True)
a3.bar(vals, cnt, width=0.85, color='#9DB9DA')
a3.axvline(mean, color=RUST, lw=2.6)
a3.text(mean + 0.4, cnt.max() * 0.98, '平均 ＝ %.2f mm' % mean, color=RUST, ha='left', va='top', fontsize=12.5, fontweight='bold')
for name, x in STRAWS:                                    # A、B 在哪一條：字母標在長條上面（標文字會壓到平均線）
    v = int(thick[x, PY])
    c = int(cnt[vals == v][0])
    a3.text(v, c + cnt.max() * 0.03, name, ha='center', va='bottom', fontsize=13, fontweight='bold', color='#1B8FC4')
a3.set_xlabel(r'$t$' + '（這根吸管數到幾 mm）', fontsize=11)
a3.set_ylabel('吸管根數', fontsize=11)
a3.set_xticks(range(0, 21, 2))
a3.set_xlim(-0.8, 20.5)
a3.tick_params(labelsize=10)
for sp in ('top', 'right'):
    a3.spines[sp].set_visible(False)
a3.grid(axis='y', alpha=0.3)
titles(fig, [(0.17, '圖一　側面看：腦頂在頭頂往下 25 mm 以內的\n吸管才算（藍點），兩側比較低的不算（黃點）'),
             (0.47, '圖二　從上往下看：藍色那塊 ＝ ' + r'$V$' + '\n共 %s 根吸管（點線＝圖一那一片）' % fmt(vertex.sum())),
             (0.82, '圖三　' + r'$V$' + ' 裡 %s 根吸管各自的 ' % fmt(vertex.sum()) + r'$t$' + '\n全部加起來 ÷ 根數 ＝ 平均')])
save(fig, '1014_method_4.png')


# ── 公式區塊：編號公式（置中、編號靠右）＋「其中」符號說明 ──────────────────────
def eq_block(name, eqs, lines, h, gap=0.33):
    """eqs：[(公式, 編號)]；lines：「其中」那幾行。位置用「從上往下幾吋」算，h 是區塊高度（吋）、gap 是行距（吋）"""
    fig = plt.figure(figsize=(W, h), facecolor=PAPER)
    y = 0.36
    for eq, num in eqs:
        fig.text(0.5, 1 - y / h, eq, ha='center', va='center', fontsize=21, color=INK)
        fig.text(0.985, 1 - y / h, r'$(%d)$' % num, ha='right', va='center', fontsize=17, color=INK)
        y += 0.66
    y -= 0.66 - 0.48
    for i, ln in enumerate(lines):
        fig.text(0.02, 1 - (y + i * gap) / h, ln, ha='left', va='top', fontsize=13, color=INK)
    save(fig, name)


IN = '其中　'
SP = '　　　'
eq_block('1014_method_eq1.png',
         [(r'$z_{\mathrm{top}}(x,y)\;=\;\max\;\{\,z\;:\;\mathrm{seg}(x,y,z)>0\,\}$', 1)],
         [IN + r'$(x,y)$' + ' ＝ 一根直立的吸管（從上往下看的位置）；' + r'$z$' + ' ＝ 高度（1 格 ＝ 1 mm）',
          SP + r'$\mathrm{seg}(x,y,z)>0$' + ' ＝ 這一格被 FreeSurfer 標成腦（圖二的藍色）',
          SP + r'$z_{\mathrm{top}}(x,y)$' + ' ＝ 這根吸管最上面的腦格子；所有吸管連起來就是黃線（圖四）'], 1.85)
eq_block('1014_method_eq2.png',
         [(r'$\tau\;=\;\frac{1}{2}\;\mathrm{median}\;I(\mathrm{cortex})$', 2)],
         [IN + r'$I$' + ' ＝ 亮度（0 ＝ 全黑，1 ＝ 最亮）；cortex ＝ FreeSurfer 標成大腦皮質的格子（標籤 3、42，圖一的綠色）',
          SP + r'$\tau$' + ' ＝ 門檻：黃線以上的格子，亮度至少 ' + r'$\tau$' + ' 才算「有東西」；比 ' + r'$\tau$' + ' 暗的（背景、水）不算',
          SP + '每個人用自己的皮質算：sub-0043 是 ' + r'$\tau$' + ' ＝ %.2f；test %d 人的 ' % (tau, len(taus)) + r'$\tau$'
          + ' 從 %.2f 到 %.2f' % (min(taus), max(taus)),
          SP + '比 ' + r'$\tau$' + ' 暗一點的殘留會漏掉；門檻改成皮質的 %.1f～%.1f 倍，紅色範圍會變，但 %d 人的結論都一樣（'
          % (FACTORS[0], FACTORS[-1], len(rows)) + r'$r$' + ' ＝ ' + r'$%.2f$' % trmax + '～' + r'$%.2f$' % trmin + '）'],
         2.02, gap=0.31)
eq_block('1014_method_eq3.png',
         [(r'$t(x,y)\;=\;\#\;\{\,z>z_{\mathrm{top}}(x,y)\;:\;I(x,y,z)\geq\tau\,\}$', 3)],
         [IN + r'$\#$' + ' ＝ 數有幾個：黃線以上（' + r'$z>z_{\mathrm{top}}$' + '）、而且亮度至少 ' + r'$\tau$' + ' 的格子（圖二的紅色）',
          SP + r'$t(x,y)$' + ' ＝ 這根吸管的殘留厚度（格數 ＝ mm）。吸管 A：' + r'$t=%d$' % int(thick[74, PY])
          + '；吸管 B：' + r'$t=%d$' % int(thick[101, PY]),
          SP + '中間隔著暗的格子，上面亮的照樣算（腦膜和腦之間本來就隔一層水）'], 1.85)
eq_block('1014_method_eq4.png',
         [(r'$V\;=\;\{\,(x,y)\;:\;z_{\mathrm{top}}(x,y)\geq\max\,z_{\mathrm{top}}-%d\,\}$' % VERTEX_BAND, 4),
          (r'$\mathrm{residue}\;=\;\frac{1}{|V|}\sum_{(x,y)\in V}t(x,y)$', 5)],
         [IN + r'$V$' + ' ＝ 頭頂那一塊的吸管：腦頂在整顆腦最高點往下 %d mm 以內（圖一藍點、圖二藍色）；' % VERTEX_BAND
          + r'$|V|$' + ' ＝ 有幾根',
          SP + '兩側不算：頭皮是斜的，直立的吸管會斜斜穿過去，數到的會比真正的厚度多',
          SP + '%d mm 是設計時定的範圍；改成 %d～%d mm，%d 人的結論都一樣（殘留越厚、皮質 Dice 進步越少，'
          % (VERTEX_BAND, BANDS[0], BANDS[-1], len(rows)) + r'$r$' + ' ＝ ' + r'$%.2f$' % rmax + '～' + r'$%.2f$' % rmin + '）',
          SP + 'residue ＝ 這個人的頭頂殘留厚度（散佈圖的橫軸）。sub-0043：%s mm ÷ %s 根 ＝ %.2f mm'
          % (fmt(tV.sum()), fmt(vertex.sum()), mean)], 2.66, gap=0.31)
