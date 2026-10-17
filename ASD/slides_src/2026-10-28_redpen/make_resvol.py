# -*- coding: utf-8 -*-
"""下次 meeting：去顱骨殘留改用「體積」（10/14 老師紅字 p12、p16、p18）的圖與數字。

2026-10-13 使用者：「可以做下次 meeting 報告順便放裡面」「先做這一 part 就好」→ 只做殘留體積這三頁。
殘留體積的量法在 ASD/skullstrip_volume.py（先跑 --scan --analyze --r 5）。只用半徑 5 mm（使用者：「取一個半徑就好」）。

  deck_data.json             投影片上的數字（build.js 讀，不手打）
  1028_resvol_demo.png       定義：一位受試者三個方向，黃線 S、藍線 B、紅色殘留
  1028_resvol_eq.png         公式 (1)(2)
  1028_resvol_result.png     左：殘留體積 vs ΔDice（170 位）；右：殘留體積 vs Dice 數值（affine、配準後各一組點）
                             2026-10-13 使用者：「170 位不用特別區分顏色」「dice 數值沒辦法用散布點來表示嗎」→ 原本右邊的長條圖改成散佈圖

數字的算法照 10/14 第 16 頁（2026-10-05 起的原始數值版）：橫軸原始體積、縱軸 ΔDice 不扣批次平均、170 位直接合起來算 Spearman r；
分組也用原始體積的上下 1/4。批內百分位的版本（skullstrip_volume.py --analyze）一併記在 json 的 pct_r，兩者應接近。
模型：mix_exp3 第 240 輪（同 10/14 的殘留分析）。
"""
import os
import sys
import csv
import json
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'ASD'))
sys.argv = sys.argv[:1]
import skullstrip_volume as sv
from skullstrip_label_dice import NAME
from skullstrip_label_dice_pooled import load as load_thick

OUT = os.path.join(ROOT, 'models', 'deck_charts')
SUBJECT, RADII, MAIN_R = 'sub-0043', (5,), 5            # 2026-10-13 使用者：「取一個半徑就好」
COH = ['test', 'val', 'MRS']
ID = {v: k for k, v in NAME.items()}


def labs_of(r):
    """「殘留旁邊的結構」：skullstrip_volume.py --analyze 挑的（全部殘留那一列）。"""
    rows = list(csv.DictReader(open(os.path.join(sv.SK, 'residue_volume_r%d_summary.csv' % r), encoding='utf-8')))
    return [ID[n] for n in rows[0]['labels'].split('、')]


def avg(row, labs):
    return float(row['dice_mean']) if labs is None else float(np.nanmean([float(row['label_%d' % l]) for l in labs]))


def stats(x, b, a, coh):
    """原始數值：170 位合起來的 Spearman r、各批 r、原始 x 的上下 1/4 之 Dice（affine → 配準後）。"""
    g = a - b
    rho, p = spearmanr(x, g)
    q1, q3 = np.percentile(x, [25, 75])
    hi, lo = x >= q3, x <= q1
    out = dict(r=float(rho), p=float(p), n=int(len(x)), median=float(np.median(x)),
               dirty=dict(n=int(hi.sum()), b=float(b[hi].mean()), a=float(a[hi].mean()), g=float(g[hi].mean())),
               clean=dict(n=int(lo.sum()), b=float(b[lo].mean()), a=float(a[lo].mean()), g=float(g[lo].mean())),
               mw_p=float(mannwhitneyu(g[hi], g[lo]).pvalue))
    for c in COH:
        k = coh == c
        rr, pp = spearmanr(x[k], g[k])
        out[c] = dict(r=float(rr), p=float(pp), n=int(k.sum()))
    # 批內百分位、批內置中（skullstrip_label_dice_pooled.py 的做法），只拿來核對
    pct, gc = np.zeros(len(x)), np.zeros(len(x))
    for c in COH:
        k = np.where(coh == c)[0]
        xs = x[k]
        pct[k] = [(np.sum(xs < v) + 0.5 * (np.sum(xs == v) - 1)) / (len(xs) - 1) for v in xs]
        gc[k] = g[k] - g[k].mean()
    out['pct_r'], out['pct_p'] = [float(v) for v in spearmanr(pct, gc)]
    out['r_before'], out['p_before'] = [float(v) for v in spearmanr(x, b)]      # 殘留 vs Dice 本身（affine、配準後）
    out['r_after'], out['p_after'] = [float(v) for v in spearmanr(x, a)]
    return out


S = {(r['cohort'], r['subject']): r for r in sv.subjects()}
D = {'model': 'mix_exp3 第 240 輪', 'subject': SUBJECT, 'radii': {}}
for r in RADII:
    V = list(csv.DictReader(open(sv.OUT_CSV(r), encoding='utf-8')))
    coh = np.array([v['cohort'] for v in V])
    x = np.array([float(v['res_total']) for v in V]) / 1000            # cm³
    bright = np.array([float(v['res_bright']) for v in V]) / 1000
    labs = labs_of(r)
    sets = {'near': labs, 'cortex': [3, 42], 'all30': None}
    e = {'labs': labs, 'labs_name': '、'.join(NAME[l] for l in labs), 'bright_share': float(np.median(bright / x)),
         'range': [float(x.min()), float(x.max())], 'cohort_median': {c: float(np.median(x[coh == c])) for c in COH}}
    for k, ls in sets.items():
        b = np.array([avg(S[(v['cohort'], v['subject'])]['before'], ls) for v in V])
        a = np.array([avg(S[(v['cohort'], v['subject'])]['after'], ls) for v in V])
        e[k] = stats(x, b, a, coh)
        if r == MAIN_R and k == 'near':
            PLOT = dict(x=x, g=a - b, b=b, a=a, coh=coh)
    D['radii'][str(r)] = e
# 對照：10/14 第 16 頁的顱頂殘留厚度（mm）vs 大腦皮質 ΔDice（同樣 170 位、原始數值）
T = load_thick()
tx = np.array([float(t['res']['top_vertex_mm']) for t in T])
tb = np.array([avg(t['before'], [3, 42]) for t in T])
ta = np.array([avg(t['after'], [3, 42]) for t in T])
D['thickness'] = stats(tx, tb, ta, np.array([t['cohort'] for t in T]))
json.dump(D, open(os.path.join(HERE, 'deck_data.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('->', os.path.join(HERE, 'deck_data.json'))
for r in RADII:
    e = D['radii'][str(r)]
    print('r=%d mm：中位數 %.0f cm³；旁邊結構 r=%+.2f（p=%.3f；批內百分位 %+.2f）；皮質 %+.2f；30 結構 %+.2f' % (
        r, e['near']['median'], e['near']['r'], e['near']['p'], e['near']['pct_r'], e['cortex']['r'], e['all30']['r']))
t = D['thickness']
print('顱頂厚度（對照）：r=%+.2f（p=%.2g；批內百分位 %+.2f）' % (t['r'], t['p'], t['pct_r']))

# ── 圖 ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, RUST, GRAY = '#0E7C7B', '#A34F1B', '#9AA09B'
CCOL = {'test': '#0E7C7B', 'val': '#5BB8B6', 'MRS': '#A34F1B'}
ptxt = lambda p: 'p < 0.001' if p < 0.001 else 'p = %.3f' % p
sg = lambda v: ('+' if v >= 0 else '−') + '%.2f' % abs(v)      # 減號用 U+2212（DejaVu Sans 補字）


def save(fig, name, dpi=130):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


# 定義：一位受試者三個方向
rec = [x for x in sv.subjects() if x['subject'] == SUBJECT][0]
z = np.load(rec['npz'])
vol, seg = z['vol'], z['seg'].astype(int)
m, _, res, _, ball = sv.measure_volume(vol, seg, MAIN_R)
brain = seg > 0
c = np.argwhere(brain).mean(0).astype(int)
views = [('冠狀面', lambda a: a[:, c[1], :].T), ('矢狀面', lambda a: a[c[0], :, :].T), ('橫切面', lambda a: a[:, :, c[2] + 25].T)]
fig, axes = plt.subplots(1, 3, figsize=(12.13, 3.9), facecolor=PAPER, gridspec_kw={'width_ratios': [192, 224, 192]})
for ax, (t, f) in zip(axes, views):
    ax.imshow(f(vol), cmap='gray', origin='lower', vmin=0, vmax=1)
    R = f(res).astype(float)
    ax.imshow(np.ma.masked_where(R == 0, R), cmap=ListedColormap(['#E74C3C']), origin='lower', alpha=0.6, interpolation='nearest')
    ax.contour(f(brain.astype(float)), levels=[0.5], colors='#F1C40F', linewidths=0.5)
    ax.contour(f(ball.astype(float)), levels=[0.5], colors='#3498DB', linewidths=1.3)
    ax.set_title(t, fontsize=12)
    ax.axis('off')
fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.01, wspace=0.03)
save(fig, '1028_resvol_demo.png', dpi=160)
D['demo_cm3'] = m['res_total'] / 1000


# 公式（同 make_arch.py 的 eq_block：公式置中、編號靠右，下面是「其中」）
def eq_block(name, eqs, lines, h, fs=20):
    W = 12.13
    fig = plt.figure(figsize=(W, h), facecolor=PAPER)
    y = 0.36
    for eq, num in eqs:
        fig.text(0.5, 1 - y / h, eq, ha='center', va='center', fontsize=fs, color=INK)
        fig.text(0.985, 1 - y / h, r'$(%d)$' % num, ha='right', va='center', fontsize=17, color=INK)
        y += 0.62
    y -= 0.62 - 0.46
    for i, ln in enumerate(lines):
        fig.text(0.02, 1 - (y + i * 0.31) / h, ln, ha='left', va='top', fontsize=13, color=INK)
    save(fig, name, dpi=200)


IN, SP = '其中　', '　　　'
eq_block('1028_resvol_eq.png',
         [(r'$B=\mathrm{Fill}\left(\left(S\oplus D_r\right)\ominus D_r\right)$', 1),
          (r'$V_{\mathrm{res}}=\left|\left\{\,x\notin B\ :\ I(x)>0\,\right\}\right|\quad(\mathrm{mm}^3)$', 2)],
         [IN + r'$S$' + '：FreeSurfer aseg 標記之腦組織（label > 0）；' + r'$D_r$' + '：半徑 ' + r'$r$'
          + ' 之球形結構元素（' + r'$r=5$' + ' mm）；' + r'$\oplus,\ \ominus$' + '：dilation、erosion',
          SP + 'Fill：填補封閉孔洞；' + r'$B$' + '：補起腦溝與大腦縱裂後之腦（圖中藍線）；'
          + r'$I$' + '：affine 後之影像（norm.mgz，1 voxel = 1 ' + r'$\mathrm{mm}^3$' + '）',
          SP + '依 10/14 會議意見：' + r'$B$' + ' 外之非零 voxel 皆計為殘留，不設強度門檻、不分區',
          SP + 'closing 補腦溝之作法參考 FreeSurfer local gyrification index 之外殼（Schaer et al., IEEE TMI 2008）'],
         2.8)

# 結果：左 ΔDice、右 Dice 數值，橫軸都是殘留體積；170 位同一個顏色
e = D['radii'][str(MAIN_R)]['near']
labs_txt = '、'.join(dict.fromkeys(NAME[l].lstrip('左右') for l in D['radii'][str(MAIN_R)]['labs']))
x = PLOT['x']
xx = np.array([x.min(), x.max()])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=PAPER)
a1.scatter(x, PLOT['g'], s=24, alpha=0.7, color=RUST, edgecolors='none')
b1, b0 = np.polyfit(x, PLOT['g'], 1)
a1.plot(xx, b0 + b1 * xx, color=INK, lw=2)
a1.set_title('ΔDice（n = %d）\nr = %s，%s' % (e['n'], sg(e['r']), ptxt(e['p'])), fontsize=14, fontweight='bold')
a1.set_ylabel('ΔDice（%s）' % labs_txt, fontsize=12.5)
for y, col, lab, rr, pp in [(PLOT['b'], GRAY, 'affine', e['r_before'], e['p_before']),
                            (PLOT['a'], TEAL, '配準後', e['r_after'], e['p_after'])]:
    a2.scatter(x, y, s=22, alpha=0.75, color=col, edgecolors='none', label='%s：r = %s，%s' % (lab, sg(rr), ptxt(pp)))
    b1, b0 = np.polyfit(x, y, 1)
    a2.plot(xx, b0 + b1 * xx, color=col, lw=2)
a2.set_title('Dice 數值（n = %d）\n灰：affine　綠：配準後' % e['n'], fontsize=14, fontweight='bold')
a2.set_ylabel('Dice（%s）' % labs_txt, fontsize=12.5)
a2.legend(fontsize=11, frameon=False, loc='center left', bbox_to_anchor=(0.0, 0.66))       # 兩群點中間的空白帶
for ax in (a1, a2):
    ax.set_xlabel('殘留體積（cm³）', fontsize=12.5)
    ax.tick_params(labelsize=11.5)
    ax.grid(alpha=.3)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
save(fig, '1028_resvol_result.png')
json.dump(D, open(os.path.join(HERE, 'deck_data.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


# 「球」B 是怎麼來的（2026-10-13 使用者：「想更了解」）：同一位受試者的冠狀面，放大上半部，四步
from scipy import ndimage as ndi
r = MAIN_R
dil = ndi.distance_transform_edt(~brain) <= r                       # S ⊕ D_r：往外長 r mm
clo = (ndi.distance_transform_edt(dil) > r) | brain                  # (S ⊕ D_r) ⊖ D_r：再往內縮 r mm
B = ndi.binary_fill_holes(clo)
out_S = (~brain) & (vol > 0)
out_B = (~B) & (vol > 0)
D['steps_cm3'] = dict(outside_S=float(out_S.sum()) / 1000, outside_B=float(out_B.sum()) / 1000,
                      inside_B_outside_S=float((B & ~brain & (vol > 0)).sum()) / 1000)
sl = lambda a: a[:, c[1], :].T
zs = np.where(sl(brain).any(1))[0]
xs_ = np.where(sl(brain).any(0))[0]
z0, z1 = c[2] - 5, min(zs.max() + 14, vol.shape[2] - 1)
x0, x1 = max(xs_.min() - 12, 0), min(xs_.max() + 12, vol.shape[0] - 1)
crop = lambda a: a[z0:z1, x0:x1]
RED = ListedColormap(['#E74C3C'])
BLUE = ListedColormap(['#3498DB'])
panels = [('① S：FreeSurfer 標記之腦組織（黃線）\n若直接以 S 外計算：紅色（腦溝、縱裂也算）', out_S, None, 'S'),
          ('② S ⊕ D：向外長 %d mm（藍色）\n腦溝與縱裂被填滿，但外緣也變大' % r, None, dil, None),
          ('③ (S ⊕ D) ⊖ D：再向內縮 %d mm（藍色）\n外緣回到原位；寬度 < %d mm 之凹陷保持填滿' % (r, 2 * r), None, clo, None),
          ('④ B ＝ Fill(③)：補起封閉孔洞（藍線）\n殘留＝B 外之非零 voxel（紅色）', out_B, None, 'B')]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.0), facecolor=PAPER)
for ax, (t, red, fill, line) in zip(axes, panels):
    ax.imshow(crop(sl(vol)), cmap='gray', origin='lower', vmin=0, vmax=1)
    if red is not None:
        R = crop(sl(red)).astype(float)
        ax.imshow(np.ma.masked_where(R == 0, R), cmap=RED, origin='lower', alpha=0.6, interpolation='nearest')
    if fill is not None:
        Fm = crop(sl(fill & ~brain)).astype(float)
        ax.imshow(np.ma.masked_where(Fm == 0, Fm), cmap=BLUE, origin='lower', alpha=0.55, interpolation='nearest')
        ax.contour(crop(sl(fill)).astype(float), levels=[0.5], colors='#3498DB', linewidths=1.2)
    ax.contour(crop(sl(brain)).astype(float), levels=[0.5], colors='#F1C40F', linewidths=0.7)
    if line == 'B':
        ax.contour(crop(sl(B)).astype(float), levels=[0.5], colors='#3498DB', linewidths=1.5)
    ax.set_title(t, fontsize=11.5)
    ax.axis('off')
axes[0].text(0.02, 0.02, '全腦：%.0f cm³' % D['steps_cm3']['outside_S'], transform=axes[0].transAxes, color='white', fontsize=11,
             fontweight='bold')
axes[3].text(0.02, 0.02, '全腦：%.0f cm³' % D['steps_cm3']['outside_B'], transform=axes[3].transAxes, color='white', fontsize=11,
             fontweight='bold')
fig.subplots_adjust(left=0.004, right=0.996, top=0.8, bottom=0.01, wspace=0.03)
save(fig, '1028_resvol_steps.png')
json.dump(D, open(os.path.join(HERE, 'deck_data.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('S 外非零 %.0f cm³；B 外非零 %.0f cm³；B 內、S 外（腦溝、縱裂）%.0f cm³' % (
    D['steps_cm3']['outside_S'], D['steps_cm3']['outside_B'], D['steps_cm3']['inside_B_outside_S']))


# ── 不同模型（10/14 紅字 p18：「想看不同 model 做出來的效果」）──────────────────────────────
# 同樣 170 位、同樣的殘留體積（半徑 5 mm）、同樣的鄰近結構，只換配準後的 Dice。val、MRS 的 Dice 在 skullstrip_check\cohorts\
MODELS = [('mix_exp3', '0240', 'mix_exp3（displacement）'), ('mix_exp6', '0190', 'mix_exp6（SVF）'),
          ('mix_wide_vel', '0240', 'mix_wide_vel（2× width）'), ('mix_cascade', '0170', 'mix_cascade'),
          ('mix_pyramid', '0245', 'mix_pyramid（coarse-to-fine）')]
MCOL = ['#9AA09B', '#0E7C7B', '#5BB8B6', '#A34F1B', '#7D3C98']


def after_path(e, ep, cn):
    if cn == 'test':
        return os.path.join(ROOT, 'models', e, 'dice_%s.csv' % ep)
    return os.path.join(sv.SK, 'cohorts', 'dice_%s_%s_%s.csv' % (e, ep, cn))


have = all(os.path.exists(after_path(e, ep, cn)) for e, ep, _ in MODELS for cn in COH)
if have:
    from skullstrip_label_dice_pooled import read as rd
    V = list(csv.DictReader(open(sv.OUT_CSV(MAIN_R), encoding='utf-8')))
    labs = D['radii'][str(MAIN_R)]['labs']
    x = np.array([float(v['res_total']) for v in V]) / 1000
    coh = np.array([v['cohort'] for v in V])
    b = np.array([avg(S[(v['cohort'], v['subject'])]['before'], labs) for v in V])
    D['models'] = {}
    lines = {}
    for e, ep, nm in MODELS:
        A = {cn: rd(after_path(e, ep, cn)) for cn in COH}
        a = np.array([avg(A[v['cohort']][v['subject']], labs) for v in V])
        st = stats(x, b, a, coh)
        st['name'], st['dice_after_mean'] = nm, float(a.mean())
        D['models'][e] = st
        lines[e] = (a, a - b)
        print('%-13s 配準後 %.3f｜ΔDice r=%+.2f p=%.3f｜配準後 Dice r=%+.2f p=%.3f' % (e, a.mean(), st['r'], st['p'], st['r_after'], st['p_after']))
    # 2026-10-14 使用者：「可以寫用哪些結構，然後把受試者點點加上去」→ 每個模型一欄，上排 ΔDice、下排 Dice 數值，每位一點
    xx = np.array([x.min(), x.max()])
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6), facecolor=PAPER, sharex=True, sharey='row')
    for j, ((e, ep, nm), col) in enumerate(zip(MODELS, MCOL)):
        st = D['models'][e]
        a, g = lines[e]
        top, bot = axes[0][j], axes[1][j]
        col = col if e != 'mix_exp3' else '#34495E'          # 跟 affine 的灰點分開
        top.scatter(x, g, s=7, alpha=0.6, color=col, edgecolors='none')
        b1, b0 = np.polyfit(x, g, 1)
        top.plot(xx, b0 + b1 * xx, color=INK, lw=1.6)
        top.set_title(nm.replace('（', '\n（'), fontsize=11, fontweight='bold')
        top.text(0.97, 0.95, 'r = %s\n%s' % (sg(st['r']), ptxt(st['p'])), transform=top.transAxes, ha='right', va='top', fontsize=9.5)
        bot.scatter(x, b, s=6, alpha=0.45, color=GRAY, edgecolors='none')
        bot.scatter(x, a, s=7, alpha=0.6, color=col, edgecolors='none')
        for y, c in [(b, '#6E7572'), (a, INK)]:
            b1, b0 = np.polyfit(x, y, 1)
            bot.plot(xx, b0 + b1 * xx, color=c, lw=1.6)
        bot.text(0.97, 0.62, '配準後 r = %s\n%s' % (sg(st['r_after']), ptxt(st['p_after'])), transform=bot.transAxes,
                 ha='right', va='top', fontsize=9.5)
        for ax in (top, bot):
            ax.tick_params(labelsize=9)
            ax.grid(alpha=.3)
            ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        bot.set_xlabel('殘留體積（cm³）', fontsize=10)
    near = D['radii'][str(MAIN_R)]['near']
    axes[1][0].text(0.03, 0.06, '灰：affine（r = %s）' % sg(near['r_before']), transform=axes[1][0].transAxes, fontsize=9.5,
                    color='#6E7572')
    axes[0][0].set_ylabel('ΔDice\n（大腦皮質、小腦皮質）', fontsize=10.5)
    axes[1][0].set_ylabel('Dice\n（大腦皮質、小腦皮質）', fontsize=10.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, '1028_resvol_models.png')
    json.dump(D, open(os.path.join(HERE, 'deck_data.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
else:
    print('[!] 不同模型：val／MRS 的 Dice 還沒齊（skullstrip_check\cohorts\），先跳過')
