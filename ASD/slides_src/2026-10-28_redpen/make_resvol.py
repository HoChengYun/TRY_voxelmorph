# -*- coding: utf-8 -*-
"""下次 meeting：去顱骨殘留改用「體積」（10/14 老師紅字 p12、p16、p18）的圖與數字。

2026-10-13 使用者：「可以做下次 meeting 報告順便放裡面」「先做這一 part 就好」→ 只做殘留體積這三頁。
殘留體積的量法在 ASD/skullstrip_volume.py（先跑 --scan --analyze --r 5）。只用半徑 5 mm（使用者：「取一個半徑就好」）。

  deck_data.json             投影片上的數字（build.js 讀，不手打）
  1028_resvol_demo.png       定義：一位受試者三個方向，黃線 S、藍線 B、紅色殘留
  1028_resvol_eq.png         公式 (1)(2)
  1028_resvol_result.png     殘留體積 vs Dice 數值（affine、配準後各一組點；170 位；10-22 起不畫 ΔDice）
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
    for c in [c for c in COH if (coh == c).any()]:                 # tigerbx 版沒有 MRS
        k = coh == c
        rr, pp = spearmanr(x[k], g[k])
        out[c] = dict(r=float(rr), p=float(pp), n=int(k.sum()))
    # 批內百分位、批內置中（skullstrip_label_dice_pooled.py 的做法），只拿來核對
    pct, gc = np.zeros(len(x)), np.zeros(len(x))
    for c in [c for c in COH if (coh == c).any()]:
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

# 結果：Dice 數值（灰 affine、綠 配準後），橫軸殘留體積；170 位同一個顏色
# 2026-10-22 使用者：「簡報不用再放 Dice 變化了」→ 拿掉左邊的 ΔDice，只留 Dice 數值
e = D['radii'][str(MAIN_R)]['near']
labs_txt = '、'.join(dict.fromkeys(NAME[l].lstrip('左右') for l in D['radii'][str(MAIN_R)]['labs']))
x = PLOT['x']
xx = np.array([x.min(), x.max()])
fig, ax = plt.subplots(figsize=(8.2, 5.0), facecolor=PAPER)
for y, col, lab, rr, pp in [(PLOT['b'], GRAY, 'affine', e['r_before'], e['p_before']),
                            (PLOT['a'], TEAL, '配準後', e['r_after'], e['p_after'])]:
    ax.scatter(x, y, s=22, alpha=0.75, color=col, edgecolors='none', label='%s：r = %s，%s' % (lab, sg(rr), ptxt(pp)))
    b1, b0 = np.polyfit(x, y, 1)
    ax.plot(xx, b0 + b1 * xx, color=col, lw=2)
ax.set_title('Dice（n = %d）　灰：affine　綠：配準後' % e['n'], fontsize=14, fontweight='bold')
ax.set_ylabel('Dice（%s）' % labs_txt, fontsize=12.5)
ax.legend(fontsize=11, frameon=False, loc='center left', bbox_to_anchor=(0.0, 0.66))       # 兩群點中間的空白帶
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
# 三個架構修改模型讀 deck_models.json（換模型只改那個檔，見 換模型.md）
from deck_models import XC, EC, XP, EP, XM, EM
# 10-28：八欄時實驗名太長、標題會疊在一起 → 改用跟其他圖相同的方法名
MODELS = [('mix_exp3', '0240', 'VoxelMorph（disp.）'), ('mix_exp6', '0190', 'VoxelMorph（SVF）'),
          ('mix_wide_vel', '0240', '2× width（SVF）'), (XC, EC, 'Cascade'),
          (XP, EP, 'Coarse-to-fine'),
          # 10-23 使用者：「p3 好像還漏 cascade + pyramid」、TransMorph 也併進來（val、MRS 的 Dice 在筆電上補算）
          (XM, EM, 'Cascade + C2F'), ('tm_mixv2', '0205', 'TransMorph（disp.）'),
          # 10-28 使用者：「很多一起比較的圖沒有補上 transmorph diff」（val、MRS 的 Dice 用 TransMorph 專案的 eval_mixv2.py 補算）
          ('tm_diff_mixv2', '0075', 'TransMorph-diff')]
MCOL = ['#9AA09B', '#0E7C7B', '#5BB8B6', '#A34F1B', '#7D3C98', '#B03A2E', '#4B5D73', '#8A9BB0']


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
    # 2026-10-14 使用者：「可以寫用哪些結構，然後把受試者點點加上去」→ 每個模型一欄，每位一點
    # 2026-10-22 使用者：「簡報不用再放 Dice 變化了」→ 拿掉上排 ΔDice，只留 Dice 數值（灰 affine、彩色配準後）
    xx = np.array([x.min(), x.max()])
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13.5, 3.7), facecolor=PAPER, sharex=True, sharey=True)
    for j, ((e, ep, nm), col) in enumerate(zip(MODELS, MCOL)):
        st = D['models'][e]
        a, g = lines[e]
        ax = axes[j]
        col = col if e != 'mix_exp3' else '#34495E'          # 跟 affine 的灰點分開
        ax.set_title(nm.replace('（', '\n（'), fontsize=10, fontweight='bold')
        ax.scatter(x, b, s=6, alpha=0.45, color=GRAY, edgecolors='none')
        ax.scatter(x, a, s=7, alpha=0.6, color=col, edgecolors='none')
        for y, c in [(b, '#6E7572'), (a, INK)]:
            b1, b0 = np.polyfit(x, y, 1)
            ax.plot(xx, b0 + b1 * xx, color=c, lw=1.6)
        ax.text(0.97, 0.79, '配準後 r = %s\n%s' % (sg(st['r_after']), ptxt(st['p_after'])), transform=ax.transAxes,
                ha='right', va='top', fontsize=9.5, zorder=6,            # 10-28：移到兩群點之間＋白底（TransMorph-diff 的點較散）
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.3)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel('殘留體積（cm³）', fontsize=10)
    near = D['radii'][str(MAIN_R)]['near']
    axes[0].text(0.03, 0.06, '灰：affine（r = %s）' % sg(near['r_before']), transform=axes[0].transAxes, fontsize=9.5,
                 color='#6E7572')
    axes[0].set_ylabel('Dice\n（大腦皮質、小腦皮質）', fontsize=10.5)
    fig.tight_layout(w_pad=0.4)
    save(fig, '1028_resvol_models.png')
    json.dump(D, open(os.path.join(HERE, 'deck_data.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
else:
    print('[!] 不同模型：val／MRS 的 Dice 還沒齊（skullstrip_check\cohorts\），先跳過')


# ── tigerbx 版（2026-10-22，老師 10/14 紅字 p12「換乾淨的 tbet（ex tigerbx）看 Dice 差多少」）──────────────
# 使用者：「看 tigerbx 那一套跑出來的效果和 freesurfer 差別」→ 同一套殘留體積的定義、同一批人（test、val；MRS 沒有 tigerbx），
# 影像、標籤、模型都換成 tigerbx 那一套（ASD/skullstrip_volume.py --data tigerbx 先跑 --scan --analyze）。
#   1028_resvol_demo_compare.png     同一位受試者：上 FreeSurfer、下 tigerbx（黃線 S、藍線 B、紅色殘留）
#   1028_resvol_fs_vs_tigerbx.png    同一個人兩套的殘留體積
#   1028_resvol_models_tigerbx.png   tigerbx 模型（tiger_exp2、tiger_exp3）：殘留體積 vs ΔDice、Dice
TIG_MODELS = [('tiger_exp2', '0250', 'tiger_exp2（SVF）'), ('tiger_exp3', '0210', 'tiger_exp3（displacement）')]
if os.path.exists(os.path.join(sv.SK, 'residue_volume_tigerbx_r%d_summary.csv' % MAIN_R)):
    FS_V = {(v['cohort'], v['subject']): float(v['res_total']) / 1000
            for v in csv.DictReader(open(sv.OUT_CSV(MAIN_R), encoding='utf-8'))}
    sv.use('tigerbx')                                                     # 以下 sv.* 都是 tigerbx 版
    from skullstrip_label_dice_pooled import read as rd
    ST = {(r['cohort'], r['subject']): r for r in sv.subjects()}
    VT = list(csv.DictReader(open(sv.OUT_CSV(MAIN_R), encoding='utf-8')))
    xt = np.array([float(v['res_total']) for v in VT]) / 1000
    coht = np.array([v['cohort'] for v in VT])
    rows = list(csv.DictReader(open(sv.SUMMARY(MAIN_R), encoding='utf-8')))
    labs_t = [ID[n] for n in rows[0]['labels'].split('、')]
    bt = np.array([avg(ST[(v['cohort'], v['subject'])]['before'], labs_t) for v in VT])
    T = {'labs': labs_t, 'labs_name': '、'.join(NAME[l] for l in labs_t), 'n': int(len(VT)),
         'median': float(np.median(xt)), 'range': [float(xt.min()), float(xt.max())], 'models': {}}
    lines_t = {}
    for e, ep, nm in TIG_MODELS:
        a = np.array([avg(rd(os.path.join(sv.SK, 'cohorts', 'dice_%s_%s_%s.csv' % (e, ep, v['cohort'])))[v['subject']], labs_t)
                      for v in VT])
        st = stats(xt, bt, a, coht)
        st['name'], st['dice_after_mean'], st['dice_before_mean'] = nm, float(a.mean()), float(bt.mean())
        T['models'][e] = st
        lines_t[e] = (a, a - bt)
        print('tigerbx %-11s affine %.3f → %.3f｜ΔDice r=%+.2f p=%.3f｜配準後 r=%+.2f p=%.3f' % (
            e, bt.mean(), a.mean(), st['r'], st['p'], st['r_after'], st['p_after']))
    # 同一個人兩套的殘留
    pair = [(FS_V[(v['cohort'], v['subject'])], float(v['res_total']) / 1000) for v in VT if (v['cohort'], v['subject']) in FS_V]
    pf, pt = np.array(pair).T
    rr, pp = spearmanr(pf, pt)
    T['pair'] = dict(n=int(len(pair)), fs_median=float(np.median(pf)), tg_median=float(np.median(pt)),
                     ratio_median=float(np.median(pt / pf)), r=float(rr), p=float(pp), tg_less=int((pt < pf).sum()))
    print('同一個人：FreeSurfer 中位數 %.0f cm³、tigerbx %.0f cm³（tigerbx／FreeSurfer 中位數 %.2f）；%d/%d 位 tigerbx 較少；r=%+.2f' % (
        T['pair']['fs_median'], T['pair']['tg_median'], T['pair']['ratio_median'], T['pair']['tg_less'], T['pair']['n'], rr))
    D['tigerbx'] = T

    # 圖 1：同一位受試者，上 FreeSurfer、下 tigerbx
    fs_npz = rec['npz']
    tg_npz = [x for x in sv.subjects() if x['subject'] == SUBJECT][0]['npz']
    fig, axes = plt.subplots(2, 3, figsize=(12.13, 7.4), facecolor=PAPER, gridspec_kw={'width_ratios': [192, 224, 192]})
    VOLS = {}
    for row, (lab, npz) in enumerate([('FreeSurfer', fs_npz), ('tigerbx', tg_npz)]):
        zz = np.load(npz)
        v_, s_ = zz['vol'], zz['seg'].astype(int)
        m_, _, res_, _, ball_ = sv.measure_volume(v_, s_, MAIN_R)
        br_ = s_ > 0
        VOLS[lab] = (v_, res_, br_, ball_, m_['res_total'] / 1000)
        c_ = np.argwhere(br_).mean(0).astype(int)
        vw = [('冠狀面', lambda a: a[:, c_[1], :].T), ('矢狀面', lambda a: a[c_[0], :, :].T), ('橫切面', lambda a: a[:, :, c_[2] + 25].T)]
        for ax, (t, f) in zip(axes[row], vw):
            ax.imshow(f(v_), cmap='gray', origin='lower', vmin=0, vmax=1)
            R_ = f(res_).astype(float)
            ax.imshow(np.ma.masked_where(R_ == 0, R_), cmap=ListedColormap(['#E74C3C']), origin='lower', alpha=0.6,
                      interpolation='nearest')
            ax.contour(f(br_.astype(float)), levels=[0.5], colors='#F1C40F', linewidths=0.5)
            ax.contour(f(ball_.astype(float)), levels=[0.5], colors='#3498DB', linewidths=1.3)
            if row == 0:
                ax.set_title(t, fontsize=12)
            ax.axis('off')
        axes[row][0].text(-0.04, 0.5, '%s\n殘留 %.0f cm³' % (lab, m_['res_total'] / 1000), transform=axes[row][0].transAxes,
                          ha='right', va='center', fontsize=12, fontweight='bold', color=INK)
        T['demo_' + lab] = m_['res_total'] / 1000
    fig.subplots_adjust(left=0.12, right=0.995, top=0.95, bottom=0.01, wspace=0.03, hspace=0.04)
    save(fig, '1028_resvol_demo_compare.png', dpi=150)

    # 圖 1b：每個方向切 5 個位置，上 FreeSurfer、下 tigerbx（10-23 使用者：「可以再多不同座標的」）
    #   1028_resvol_slices_{cor,sag,ax}.png，右邊一格標出切面位置
    # 位置用 MNI 座標（affine 後）。v3 atlas 是 MNI152 2009c 在 y 方向前面裁掉 2 層（逐 voxel 比對過，r = 0.9999）：
    #   x = i − 96、y = j − 130、z = k − 78。atlas 的 nii 檔頭寫 y = j − 134（make_atlas.py 裁切時沒照 LPS 方向換算），不用檔頭
    MNI0 = (96, 130, 78)
    CUTS = [('cor', 1, 'y', (40, 10, -20, -50, -80), (0, -2)),      # 位置圖：矢狀面 x = −2
            ('sag', 0, 'x', (-50, -30, -2, 30, 50), (1, -20)),      # x = −2 同 1028_ants_vis_sagittal（中線往左 2 voxels）；位置圖：冠狀面 y = −20
            ('ax', 2, 'z', (-40, -15, 10, 35, 60), (0, -2))]        # 位置圖：矢狀面 x = −2
    # 10-27 使用者：「p17 可以併到 p5 6 7」→ 第三列 SynthStrip（＋SynthSeg 標籤；data/synthseg_preprocessed_v1，跟 tigerbx 一樣 --n4 前處理）。
    # SynthSeg 把腦溝與腦表面外的腦脊髓液也標成 CSF（24），直接用 seg > 0 補球的話殘留幾乎都被算進球裡 → 這一套的 S 不含 24
    SS_NPZ = os.path.join(ROOT, 'data', 'synthseg_preprocessed_v1', 'test', SUBJECT + '.npz')
    ROWS = ['FreeSurfer', 'tigerbx']
    if os.path.exists(SS_NPZ):
        zz = np.load(SS_NPZ)
        v_, s_ = zz['vol'], zz['seg'].astype(int)
        s_[s_ == 24] = 0
        m_, _, res_, _, ball_ = sv.measure_volume(v_, s_, MAIN_R)
        VOLS['SynthStrip'] = (v_, res_, s_ > 0, ball_, m_['res_total'] / 1000)
        T['demo_SynthStrip'] = m_['res_total'] / 1000
        T['demo_SynthStrip_mask'] = float((v_ > 0).sum()) / 1000
        ROWS.append('SynthStrip')
    T['slice_rows'] = ROWS
    fv, _, _, _, _ = VOLS['FreeSurfer']
    head = np.any([VOLS[r][0] > 0 for r in ROWS], axis=0)
    lo = [max(int(np.where(head.any(axis=tuple(k for k in range(3) if k != a)))[0].min()) - 3, 0) for a in range(3)]
    hi = [min(int(np.where(head.any(axis=tuple(k for k in range(3) if k != a)))[0].max()) + 4, head.shape[a]) for a in range(3)]
    # 切面 → 2D（同上面那張：.T、origin lower；冠狀、橫切面影像左＝受試者左）＋裁到頭的範圍
    sl2 = lambda a, ax, idx: np.take(a, idx, axis=ax).T[lo[max(k for k in range(3) if k != ax)]:hi[max(k for k in range(3) if k != ax)],
                                                        lo[min(k for k in range(3) if k != ax)]:hi[min(k for k in range(3) if k != ax)]]
    T['slices'] = {}
    for key, ax_, nm, pos, (lax, lmm) in CUTS:
        cols_ax = min(k for k in range(3) if k != ax_)
        rows_ax = max(k for k in range(3) if k != ax_)
        pw, ph = hi[cols_ax] - lo[cols_ax], hi[rows_ax] - lo[rows_ax]
        lcols = min(k for k in range(3) if k != lax)
        lrows = max(k for k in range(3) if k != lax)
        lw_, lh_ = hi[lcols] - lo[lcols], hi[lrows] - lo[lrows]
        # 版面（吋）：左邊標籤、5 格 × 2 列、右邊位置圖（高度＝KL 列、上下置中）。先解出每 voxel 幾吋
        FW, LM, RM, TOP, BOT, G, GL, KL = 12.13, 1.2, 0.04, 0.36, 0.04, 0.05, 0.22, 1.15
        # 5·pw·s + 4G + GL + lw_·(KL·ph·s)/lh_ = FW − LM − RM
        NR, HMAX = len(ROWS), 5.0          # 三列時改由高度決定比例尺（投影片上留得下觀察那一行）
        s = min((FW - LM - RM - 4 * G - GL) / (5 * pw + lw_ * KL * ph / lh_), (HMAX - TOP - BOT - (NR - 1) * G) / (NR * ph))
        FW = LM + 5 * pw * s + 4 * G + GL + lw_ * KL * ph * s / lh_ + RM
        H = TOP + NR * ph * s + (NR - 1) * G + BOT
        fig = plt.figure(figsize=(FW, H), facecolor=PAPER)
        A = lambda x, y, w, h: fig.add_axes([x / FW, 1 - (y + h) / H, w / FW, h / H])
        idx = [p + MNI0[ax_] for p in pos]
        for row, lab in enumerate(ROWS):
            v_, res_, br_, ball_, cm3 = VOLS[lab]
            for j, (p, i_) in enumerate(zip(pos, idx)):
                ax = A(LM + j * (pw * s + G), TOP + row * (ph * s + G), pw * s, ph * s)
                ax.imshow(sl2(v_, ax_, i_), cmap='gray', origin='lower', vmin=0, vmax=1)
                R_ = sl2(res_, ax_, i_).astype(float)
                ax.imshow(np.ma.masked_where(R_ == 0, R_), cmap=ListedColormap(['#E74C3C']), origin='lower', alpha=0.6,
                          interpolation='nearest')
                ax.contour(sl2(br_, ax_, i_).astype(float), levels=[0.5], colors='#F1C40F', linewidths=0.4)
                ax.contour(sl2(ball_, ax_, i_).astype(float), levels=[0.5], colors='#3498DB', linewidths=1.0)
                ax.axis('off')
                if row == 0:
                    ax.set_title('%s = %s mm' % (nm, sg(p).replace('.00', '')), fontsize=11.5)
            fig.text((LM - 0.1) / FW, 1 - (TOP + row * (ph * s + G) + ph * s / 2) / H, '%s\n殘留 %.0f cm³' % (lab, cm3),
                     ha='right', va='center', fontsize=11.5, fontweight='bold', color=INK)
        # 位置圖（FreeSurfer 影像）：切的位置畫線
        lx = LM + 5 * pw * s + 4 * G + GL
        lh_in = KL * ph * s
        ax = A(lx, TOP + (NR * ph * s + (NR - 1) * G - lh_in) / 2, lw_ * lh_in / lh_, lh_in)
        ax.imshow(sl2(fv, lax, lmm + MNI0[lax]), cmap='gray', origin='lower', vmin=0, vmax=1)
        for p, i_ in zip(pos, idx):
            if ax_ == lrows:                              # 切的軸在位置圖的縱向 → 水平線
                ax.axhline(i_ - lo[ax_], color='#F39C12', lw=1.1)
            else:
                ax.axvline(i_ - lo[ax_], color='#F39C12', lw=1.1)
        ax.set_title('切面位置', fontsize=10.5)
        ax.axis('off')
        save(fig, '1028_resvol_slices_%s.png' % key, dpi=150)
        T['slices'][key] = dict(axis=nm, mm=list(pos), locator=dict(axis='xyz'[lax], mm=lmm))

    # 圖 2：同一個人兩套的殘留
    # 放在投影片右上角約 5 吋寬：figsize 跟著縮小，字才不會太小
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.4), facecolor=PAPER, gridspec_kw={'width_ratios': [1, 1]})
    hi = max(pf.max(), pt.max()) * 1.05
    a1.scatter(pf, pt, s=10, alpha=0.7, color=TEAL, edgecolors='none')
    a1.plot([0, hi], [0, hi], color=MUTED, ls='--', lw=1)
    a1.text(hi * 0.42, hi * 0.47, '兩套相同', ha='center', va='bottom', rotation=45, fontsize=8.5, color=MUTED)
    a1.set_xlim(0, hi)
    a1.set_ylim(0, hi)
    a1.set_aspect('equal')
    a1.set_xlabel('FreeSurfer（cm³）', fontsize=9.5)
    a1.set_ylabel('tigerbx（cm³）', fontsize=9.5)
    a1.set_title('(a) 同一位受試者（n = %d）' % T['pair']['n'], fontsize=10, loc='left')
    bp = a2.boxplot([pf, pt], widths=0.5, patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], [GRAY, TEAL]):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    jit = np.random.default_rng(0)
    for i, arr in enumerate([pf, pt]):
        a2.scatter(i + 1 + jit.uniform(-0.12, 0.12, len(arr)), arr, s=8, alpha=0.6, color=INK, edgecolors='none')
    a2.set_xticks([1, 2])
    a2.set_xticklabels(['FreeSurfer', 'tigerbx'])
    a2.set_ylim(0, hi)
    a2.set_ylabel('殘留體積（cm³）', fontsize=9.5)
    a2.set_title('(b) 中位數 %.0f → %.0f cm³' % (T['pair']['fs_median'], T['pair']['tg_median']), fontsize=10, loc='left')
    for ax in (a1, a2):
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=.3)
        ax.set_axisbelow(True)
        for sp_ in ('top', 'right'):
            ax.spines[sp_].set_visible(False)
    fig.tight_layout()
    save(fig, '1028_resvol_fs_vs_tigerbx.png')

    # 圖 3：tigerbx 模型（同 1028_resvol_models.png 的排法；10-22 起只畫 Dice 數值）
    xx = np.array([xt.min(), xt.max()])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.7), facecolor=PAPER, sharex=True, sharey=True)
    for j, ((e, ep, nm), col) in enumerate(zip(TIG_MODELS, ['#0E7C7B', '#34495E'])):
        st = T['models'][e]
        a, g = lines_t[e]
        ax = axes[j]
        ax.set_title(nm.replace('（', '\n（'), fontsize=11, fontweight='bold')
        ax.scatter(xt, bt, s=7, alpha=0.45, color=GRAY, edgecolors='none')
        ax.scatter(xt, a, s=8, alpha=0.6, color=col, edgecolors='none')
        for y, c in [(bt, '#6E7572'), (a, INK)]:
            b1, b0 = np.polyfit(xt, y, 1)
            ax.plot(xx, b0 + b1 * xx, color=c, lw=1.6)
        ax.text(0.97, 0.62, '配準後 r = %s\n%s' % (sg(st['r_after']), ptxt(st['p_after'])), transform=ax.transAxes,
                ha='right', va='top', fontsize=9.5)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.3)
        ax.set_axisbelow(True)
        for sp_ in ('top', 'right'):
            ax.spines[sp_].set_visible(False)
        ax.set_xlabel('殘留體積（cm³）', fontsize=10)
    axes[0].text(0.03, 0.06, '灰：affine（r = %s）' % sg(T['models']['tiger_exp3']['r_before']), transform=axes[0].transAxes,
                 fontsize=9.5, color='#6E7572')
    axes[0].set_ylabel('Dice（tigerbx 標籤）', fontsize=10)
    fig.tight_layout(w_pad=0.4)
    save(fig, '1028_resvol_models_tigerbx.png')
    # 圖 4：速度場 exp2 對照（使用者 10-22：「FreeSurfer 後面跑的很多實驗 tigerbx 沒跟到，老師比較喜歡速度場，就拿 exp2 來比較」）
    # mix_exp2 vs tiger_exp2：設定相同（SVF、int_steps 7、int_downsize 2、λ 1、250 輪），只差資料。同一批人（test、val）
    #   1028_resvol_exp2_compare.png   左 FreeSurfer、右 tigerbx；橫軸同一個範圍
    FSB = {'test': rd(os.path.join(ROOT, 'models', 'mix_exp2', 'dice_baseline.csv')),
           'val': rd(os.path.join(ROOT, 'models', 'mix_exp2', 'dice_baseline_val.csv'))}
    FSA = {'test': rd(os.path.join(ROOT, 'models', 'mix_exp2', 'dice_0240.csv')),
           'val': rd(os.path.join(sv.SK, 'cohorts', 'dice_mix_exp2_0240_val.csv'))}
    TGA = {c: rd(os.path.join(sv.SK, 'cohorts', 'dice_tiger_exp2_0250_%s.csv' % c)) for c in ('test', 'val')}
    labs_fs = D['radii'][str(MAIN_R)]['labs']
    VTd = {(v['cohort'], v['subject']): float(v['res_total']) / 1000 for v in VT}
    keys = [k for k in VTd if k in FS_V]
    ck = np.array([c for c, _ in keys])
    E2 = {}
    for tag, x_, b_, a_, ls in [
            ('fs', np.array([FS_V[k] for k in keys]), np.array([avg(FSB[c][s], labs_fs) for c, s in keys]),
             np.array([avg(FSA[c][s], labs_fs) for c, s in keys]), labs_fs),
            ('tg', np.array([VTd[k] for k in keys]), np.array([avg(ST[k]['before'], labs_t) for k in keys]),
             np.array([avg(TGA[c][s], labs_t) for c, s in keys]), labs_t)]:
        st = stats(x_, b_, a_, ck)
        st.update(dice_before_mean=float(b_.mean()), dice_after_mean=float(a_.mean()), gain_mean=float((a_ - b_).mean()),
                  labs_name='、'.join(NAME[l] for l in ls),
                  # 斜率（每 100 cm³）與個體差異：配準後 r 顯著但 affine 不顯著時，用來說明是斜率相近、配準後分散較小
                  slope_before=float(np.polyfit(x_, b_, 1)[0] * 100), slope_after=float(np.polyfit(x_, a_, 1)[0] * 100),
                  sd_before=float(b_.std()), sd_after=float(a_.std()), xr=[float(x_.min()), float(x_.max())])
        E2[tag] = (st, x_, b_, a_)
        print('exp2 %s：殘留中位數 %.0f cm³｜affine %.3f → %.3f（%+.3f）｜ΔDice r=%+.2f p=%.3f｜配準後 r=%+.2f p=%.3f（n=%d）' % (
            tag, st['median'], b_.mean(), a_.mean(), (a_ - b_).mean(), st['r'], st['p'], st['r_after'], st['p_after'], st['n']))
    T['exp2'] = {k: v[0] for k, v in E2.items()}
    _br = np.array([(np.load(os.path.join(ROOT, 'data', 'tigerbx_preprocessed_v2', c, s + '.npz'))['seg'] > 0).sum() / 1000 for c, s in keys])
    T['exp2']['tg']['r_res_brain'] = float(spearmanr(E2['tg'][1], _br)[0])          # 殘留 vs 腦體積
    T['exp2']['tg']['r_after_brain'] = float(spearmanr(E2['tg'][3], _br)[0])        # 配準後 Dice vs 腦體積
    xmax = max(E2['fs'][1].max(), E2['tg'][1].max()) * 1.05
    # 10-22 使用者：「簡報不用再放 Dice 變化了」→ 拿掉上排 ΔDice，只留 Dice 數值
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.4), facecolor=PAPER, sharex=True, sharey=True)
    for j, (tag, nm, col) in enumerate([('fs', 'FreeSurfer（mix_exp2）', RUST), ('tg', 'tigerbx（tiger_exp2）', TEAL)]):
        st, x_, b_, a_ = E2[tag]
        ax = axes[j]
        xx = np.array([x_.min(), x_.max()])
        ax.set_title(nm, fontsize=11.5, fontweight='bold', color=col)
        ax.scatter(x_, b_, s=8, alpha=0.45, color=GRAY, edgecolors='none')
        ax.scatter(x_, a_, s=10, alpha=0.65, color=col, edgecolors='none')
        for y, c in [(b_, '#6E7572'), (a_, INK)]:
            b1, b0 = np.polyfit(x_, y, 1)
            ax.plot(xx, b0 + b1 * xx, color=c, lw=1.6)
        ax.axvline(st['median'], color=col, ls=':', lw=1)
        # 放在沒有點的那一側（FreeSurfer 的點在右、tigerbx 的點在左）
        tx, ha = (0.03, 'left') if tag == 'fs' else (0.97, 'right')
        ax.text(tx, 0.62, '配準後 r = %s，%s\naffine r = %s' % (sg(st['r_after']), ptxt(st['p_after']), sg(st['r_before'])),
                transform=ax.transAxes, ha=ha, va='top', fontsize=9.5)
        ax.text(0.03, 0.06, '殘留中位數 %.0f cm³' % st['median'], transform=ax.transAxes, fontsize=9.5, color=col, fontweight='bold')
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.3)
        ax.set_axisbelow(True)
        for sp_ in ('top', 'right'):
            ax.spines[sp_].set_visible(False)
        ax.set_xlabel('殘留體積（cm³）', fontsize=10)
        ax.set_xlim(0, xmax)
    axes[0].set_ylabel('Dice（各自之標籤）\n灰：affine', fontsize=10.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, '1028_resvol_exp2_compare.png')
    json.dump(D, open(os.path.join(HERE, 'deck_data.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
else:
    print('[!] tigerbx 版：先跑 python ASD\\skullstrip_volume.py --data tigerbx --scan --analyze --r 5')
