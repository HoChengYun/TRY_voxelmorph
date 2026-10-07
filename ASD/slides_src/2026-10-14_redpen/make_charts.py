# -*- coding: utf-8 -*-
"""10/14 簡報的圖 -> models/deck_charts/1014_*.png。數字讀 deck_data.json（先跑 gather.py）。
「頭頂殘留怎麼量」那四頁的圖和公式在 make_method.py（2026-10-06 從這裡搬出去）。

  1014_folding_regions.png  ① 擠爆的點落在哪些區域（只留「佔幾 %」那一格、字放大）
  1014_ablation.png         ② 前次結論：一次只改一件事（09-20 那份 ablation.png 的正式用語版）
  1014_lambda.png           ② 平滑權重 2 / 1 / 0.5：速度場 vs 位移場（還沒跑完的點標「訓練中」）
圖上的字 2026-10-07 起一律用正式學術用語（使用者：「正式一點、不要太口語」；術語對照見 README）。
  1014_dilution.png         ③ 30 個結構一起平均 vs 只平均殘留旁邊的結構：頭頂殘留厚度（mm） vs Dice 進步多少（170 人）
  1014_regions.png          ③④ 頭頂／顱底／後腦杓：殘留量 vs 只算殘留旁邊結構的 Dice 進步（三張散佈圖）
  1014_top_example.png      ③ 頭頂放大：殘留多的一位 vs 乾淨的一位（只放加標記的那張、字放大）
  1014_six.png              ③ top_compare.png 那 6 位：皮質 Dice 起點 → 配準後，附 30 個結構一起平均（簡報用精簡版）
  1014_six_back.png         ④ 同上，後腦杓殘留最多／最少各 3 位（只算大腦皮質、小腦皮質）
  1014_wide_struct.png      ⑤ 加寬：每個結構（加寬的效果、加寬後換版本）
  1014_wide_difficulty.png  ⑤ 加寬：越難對的人幫越多（位移場、速度場並排）
  1014_wide_loss.png        ⑤ 加寬：四顆的訓練 loss（影像項、平滑項）
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'ASD'))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
D = json.load(open(os.path.join(HERE, 'deck_data.json'), encoding='utf-8'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#D9D9D2'
TEAL, RUST, RED = '#0E7C7B', '#A34F1B', '#C0392B'


def clean(ax):
    ax.grid(alpha=.3)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


# ── ② 平滑權重 ─────────────────────────────────────────────────────────
M = D['models']
W = [2.0, 1.0, 0.5]                      # 由左到右越放鬆
# 2026-10-07 使用者要求正式用語：圖上的字一律用學術用語（術語對照見 README）
SERIES = [('SVF（full-res.）', TEAL, {2.0: 'mix_exp5', 1.0: 'mix_exp6', 0.5: 'mix_exp7'}),
          ('Displacement field（full-res.）', RUST, {2.0: 'mix_exp4', 1.0: 'mix_exp3'})]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), facecolor=PAPER)
jtxt = lambda y: '0%' if y == 0 else ('< 0.001%' if y < 0.001 else '%.3f%%' % y)   # 速度場權重 1、0.5 只有零星幾點
for ax, key, title in ((axes[0], 'mean', 'Dice（test, n = 51）'), (axes[1], 'jneg', 'Folding ratio（%|J| ≤ 0）')):
    pts = {name: {i: M[exps[w]][key] for i, w in enumerate(W) if w in exps and M[exps[w]]['status'] == 'done'}
           for name, col, exps in SERIES}
    for name, col, exps in SERIES:
        xs = sorted(pts[name])
        ys = [pts[name][i] for i in xs]
        ax.plot(xs, ys, color=col, lw=2.6, marker='o', ms=9, label=name, zorder=3)
        for x, y in zip(xs, ys):
            if key == 'mean':            # 兩條線靠很近：同一個位置比高低，高的數字放上面、低的放下面
                other = [pts[n][x] for n in pts if n != name and x in pts[n]]
                up = not other or y >= max(other)
                ax.annotate('%.3f' % y, (x, y), textcoords='offset points', xytext=(0, 11 if up else -13),
                            ha='center', va='bottom' if up else 'top', fontsize=12, fontweight='bold', color=col)
            else:
                ax.annotate(jtxt(y), (x, y), textcoords='offset points', xytext=(0, 11), ha='center',
                            fontsize=12, fontweight='bold', color=col)
        for i, w in enumerate(W):
            if w in exps and M[exps[w]]['status'] == 'pending':
                ax.annotate('%s\n訓練中' % exps[w], (i, 0), xycoords=('data', 'axes fraction'), xytext=(0, 34),
                            textcoords='offset points', ha='center', fontsize=11, color=col,
                            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=col, ls='--', lw=1.2))
    ax.set_xticks(range(len(W)))
    ax.set_xticklabels(['λ = %g' % w for w in W], fontsize=12)
    ax.set_xlim(-0.5, len(W) - 0.5)
    ax.set_title(title, fontsize=13.5, fontweight='bold')
    ax.annotate('', xy=(0.98, -0.16), xytext=(0.02, -0.16), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.2))
    ax.text(0.5, -0.22, '向右：平滑正則化減弱（允許較大之局部形變）', transform=ax.transAxes, ha='center',
            fontsize=11, color=MUTED)
    clean(ax)
axes[0].set_ylim(0.795, 0.812)
axes[0].legend(fontsize=11.5, frameon=False, loc='upper left')
axes[1].set_ylim(0, 0.42)
axes[1].axhline(0.366, color=RED, ls='--', lw=1.2)
axes[1].text(len(W) - 0.55, 0.373, 'VoxelMorph（TMI 2019, Table I）0.366%', ha='right', va='bottom', fontsize=11, color=RED)
fig.tight_layout(rect=[0, 0.04, 1, 1])
save(fig, '1014_lambda.png')

# ── ② 前次結論：一次只改一件事（09-20 那份 ablation.png 的正式用語版，2026-10-07；09-20 的產生器不動）────────
# mix_exp2 → mix_exp5 只換積分解析度、mix_exp5 → mix_exp4 只換參數化方式、mix_exp4 → mix_exp3 只換 λ（手冊 §20.5）
AB = [('mix_exp2', 'SVF\nhalf-res.\nλ = 2', TEAL), ('mix_exp5', 'SVF\nfull-res.\nλ = 2', '#3A9E9C'),
      ('mix_exp4', 'Displacement\nfull-res.\nλ = 2', '#D9895A'), ('mix_exp3', 'Displacement\nfull-res.\nλ = 1', RUST)]
if all(M[e]['status'] == 'done' for e, *_ in AB):
    vals = [M[e]['mean'] for e, *_ in AB]
    jv = [M[e]['jneg'] for e, *_ in AB]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), facecolor=PAPER, gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]
    for i, ((e, lab, col), v) in enumerate(zip(AB, vals)):
        ax.bar(i, v, color=col, width=.6)
        ax.text(i, v + .0015, '%.3f' % v, ha='center', fontsize=14, fontweight='bold')
    ax.axhline(D['baseline'], ls='--', color=RED, lw=1.3)
    ax.text(-0.42, D['baseline'] + .0018, 'Affine（形變配準前）%.3f' % D['baseline'], ha='left', color=RED,
            fontsize=10, bbox=dict(fc='white', ec='none', pad=1.5))
    TOP = 0.815
    for i, t in enumerate(('積分解析度\n%+.3f' % (vals[1] - vals[0]), '參數化方式\n%+.3f' % (vals[2] - vals[1]),
                           'λ 2 → 1\n%+.3f' % (vals[3] - vals[2]))):
        ax.annotate('', xy=(i + 1, TOP), xytext=(i, TOP), arrowprops=dict(arrowstyle='->', color=INK, lw=1.6))
        ax.text(i + .5, TOP + .0035, t, ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=RED if vals[i + 1] < vals[i] else INK, bbox=dict(fc='white', ec=RULE, pad=2.5))
        for xx in (i, i + 1):
            ax.plot([xx, xx], [vals[xx] + .003, TOP], color=RULE, lw=.9, ls=':')
    ax.set_xticks(range(len(AB)))
    ax.set_xticklabels([a[1] for a in AB], fontsize=10.5)
    ax.set_ylim(0.68, 0.84)
    ax.set_ylabel('Dice', fontsize=11)
    ax.set_title('Dice（test, n = 51）', fontsize=13, fontweight='bold')
    ax = axes[1]
    for i, ((e, lab, col), v) in enumerate(zip(AB, jv)):
        ax.bar(i, v, color=col, width=.6)
        ax.text(i, v + .006, '%.3f%%' % v, ha='center', fontsize=13, fontweight='bold')
    ax.axhline(0.366, ls='--', color=RED, lw=1.3)
    ax.text(len(AB) - .58, 0.375, 'VoxelMorph（TMI 2019, Table I）0.366%', ha='right', color=RED, fontsize=10)
    ax.set_xticks(range(len(AB)))
    ax.set_xticklabels([a[1] for a in AB], fontsize=10)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel('Folding ratio（%）', fontsize=11)
    ax.set_title('Folding ratio（%|J| ≤ 0）', fontsize=13, fontweight='bold')
    for ax in axes:
        clean(ax)
        ax.grid(axis='x', alpha=0)
    fig.suptitle('消融比較：每次僅改變一項因素', fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    save(fig, '1014_ablation.png')

# ── ② 平滑權重變小，哪些結構變好、哪些變差（跟權重 2 比）──────────────────────────
S = D.get('struct', {})
if all(e in S for e in ('mix_exp5', 'mix_exp6', 'mix_exp7')):
    names = sorted(S['mix_exp5'], key=lambda n: S['mix_exp7'][n] - S['mix_exp5'][n])
    d1 = np.array([S['mix_exp6'][n] - S['mix_exp5'][n] for n in names])
    d05 = np.array([S['mix_exp7'][n] - S['mix_exp5'][n] for n in names])
    fig, ax = plt.subplots(figsize=(9.2, 7.0), facecolor=PAPER)    # 簡報上放左半邊，圖小一點、字才不會縮太小
    y = np.arange(len(names))
    ax.barh(y + 0.2, d1, height=0.38, color='#5BB8B6', label='λ = 1（mix_exp6）')
    ax.barh(y - 0.2, d05, height=0.38, color='#0A4F4E', label='λ = 0.5（mix_exp7）')
    for yy, v in zip(y - 0.2, d05):
        ax.text(v + (0.0012 if v >= 0 else -0.0012), yy, '%+.3f' % v, va='center', ha='left' if v >= 0 else 'right',
                fontsize=12, color='#0A4F4E', fontweight='bold')
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=13.5)
    ax.tick_params(axis='x', labelsize=11.5)
    lim = max(abs(d05).max(), abs(d1).max()) + 0.014
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('ΔDice 相對 λ = 2（mix_exp5）（左右平均）', fontsize=13)
    ax.legend(fontsize=13, frameon=False, loc='lower right')
    clean(ax)
    ax.grid(axis='y', alpha=0)
    fig.tight_layout()
    save(fig, '1014_lambda_struct.png')

# ── ③ 30 個結構一起平均 vs 只平均殘留旁邊的結構 ─────────────────────────────
from skullstrip_label_dice_pooled import load
from skullstrip_label_dice import NAME as LNAME
rows = load()
ctx = lambda row: float(np.mean([float(row['label_%d' % l]) for l in (3, 42)]))
MM = D['residue_mm']
ptxt = lambda p: 'p < 0.001' if p < 0.001 else 'p = %.2f' % p


def same_span(axes, ys, pad=1.12):
    """每一格縱軸代表一樣多的 Dice（只是起點不同），格子之間的斜率才能直接比。"""
    w = max(float(y.max() - y.min()) for y in ys) * pad
    for ax, y in zip(axes, ys):
        mid = (float(y.max()) + float(y.min())) / 2
        ax.set_ylim(mid - w / 2, mid + w / 2)


# 2026-10-05 起：橫軸是頭頂殘留厚度（mm）、縱軸是 Dice 進步多少（配準後 - 配準前），都是原始數字、170 人直接合在一起。
# 以前橫軸是「每批各自排名次」、縱軸扣掉同一批的平均；三批的分布差不多，r 幾乎一樣（gather.py 的 residue_mm）
x = np.array([float(r['res']['top_vertex_mm']) for r in rows])
YS = [(np.array([float(r['after']['dice_mean']) - float(r['before']['dice_mean']) for r in rows]),
       '原方法：30 結構平均', MM['top']['all_r'], MM['top']['all_p'], MUTED),
      (np.array([ctx(r['after']) - ctx(r['before']) for r in rows]),
       '會議建議方法：殘留鄰近結構（左右大腦皮質）', MM['top']['r'], MM['top']['p'], RUST)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), facecolor=PAPER)
for ax, (y, title, rr, pp, col) in zip(axes, YS):
    ax.scatter(x, y, s=24, alpha=0.7, color=col, edgecolors='none')
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, b0 + b1 * xx, color=INK, lw=2)
    ax.set_title('%s\nr = %+.2f，%s' % (title, rr, ptxt(pp)), fontsize=15, fontweight='bold')
    ax.set_xlabel('顱頂殘留厚度（mm）', fontsize=13)
    ax.set_ylabel('ΔDice（配準後 − affine）', fontsize=13)
    ax.tick_params(labelsize=12)                       # 投影片上這張縮到約 3.5 吋高，字要放大
    clean(ax)
same_span(axes, [y for y, *_ in YS])
fig.tight_layout()
save(fig, '1014_dilution.png')

# ── ③④ 三個位置：殘留量 vs 只算殘留旁邊結構的 Dice 進步（2026-10-05 起，取代原本 r 的長條）──────────────
LOC = [('top', '顱頂', 'top_vertex_mm', '殘留厚度（mm）'), ('base', '顱底', 'base_blob10', '最大殘留團塊體積（mm³）'),
       ('back', '枕部', 'back_occ_mm', '殘留厚度（mm）')]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), facecolor=PAPER)
ys = []
for ax, (k, nm, metric, xl) in zip(axes, LOC):
    labs = MM[k]['labs']
    av = lambda row: float(np.nanmean([float(row['label_%d' % l]) for l in labs]))
    xk = np.array([float(r['res'][metric]) for r in rows])
    yk = np.array([av(r['after']) - av(r['before']) for r in rows])
    col = RUST if MM[k]['p'] < 0.05 else '#9AA09B'
    ax.scatter(xk, yk, s=20, alpha=0.7, color=col, edgecolors='none')
    b1, b0 = np.polyfit(xk, yk, 1)
    xx = np.array([xk.min(), xk.max()])
    ax.plot(xx, b0 + b1 * xx, color=INK, lw=2)
    struct = '、'.join(dict.fromkeys(LNAME[l].lstrip('左右') for l in labs))     # 左右合併
    ax.set_title('%s（%s）\nr = %+.2f，%s' % (nm, struct, MM[k]['r'], ptxt(MM[k]['p'])),
                 fontsize=13.5, fontweight='bold', color=INK)
    ax.set_xlabel(xl, fontsize=12.5)
    ax.tick_params(labelsize=11.5)
    clean(ax)
    ys.append(yk)
axes[0].set_ylabel('ΔDice（配準後 − affine）', fontsize=12.5)
same_span(axes, ys)
fig.tight_layout()
save(fig, '1014_regions.png')

# ── ① 擠爆的點落在哪些區域（佔幾 %）─────────────────────────────────────────
FOLD = D['folding']
REG = ['大腦皮質', '大腦白質', '腦內、沒有標籤', '腦室・腦脊髓液・脈絡叢', '小腦・腦幹', '深部灰質・海馬・杏仁核', '腦外（背景）']
SHOW = {'大腦皮質': '大腦皮質', '大腦白質': '大腦白質', '腦內、沒有標籤': '腦內未標記區域（多為腦溝）',
        '腦室・腦脊髓液・脈絡叢': '腦室・腦脊髓液', '小腦・腦幹': '小腦・腦幹',
        '深部灰質・海馬・杏仁核': '深部灰質・海馬・杏仁核', '腦外（背景）': '腦外'}
FCOL = {'mix_exp4': '#D9895A', 'mix_exp3': '#A34F1B', 'mix_wide': '#6B3FA0'}
FLAB = {'mix_exp4': 'Displacement・λ = 2', 'mix_exp3': 'Displacement・λ = 1', 'mix_wide': 'Displacement・λ = 1・2× width'}
fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=PAPER)
wbar = 0.26
for j, e in enumerate(('mix_exp4', 'mix_exp3', 'mix_wide')):
    v = [FOLD[e]['share'][r] for r in REG]
    ax.barh(np.arange(len(REG)) + (j - 1) * wbar, v, height=wbar * 0.92, color=FCOL[e], label=FLAB[e])
for i, r in enumerate(REG):
    ax.text(max(FOLD[e]['share'][r] for e in ('mix_exp4', 'mix_exp3', 'mix_wide')) + 0.8, i, '%.1f%%' % FOLD['mix_exp3']['share'][r],
            va='center', fontsize=12.5, fontweight='bold', color=FCOL['mix_exp3'])
ax.set_yticks(range(len(REG)))
ax.set_yticklabels([SHOW[r] for r in REG], fontsize=13)
ax.invert_yaxis()
ax.set_xlabel('Folding voxel 所占比例（%；數值標示為 λ = 1）', fontsize=12)
ax.legend(fontsize=12, frameon=False, loc='lower right')
clean(ax)
ax.grid(axis='y', alpha=0)
fig.tight_layout()
save(fig, '1014_folding_regions.png')

# ── ③ 頭頂放大：殘留多的一位 vs 乾淨的一位 ───────────────────────────────────
from check_top_residue import split_top, panel, GREEN, RED as RRED
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
EX = [('MRS0381-2', '殘留多'), ('T054', '殘留少')]      # 同 check_top_residue.py --example
npz = {r['subject']: r['npz'] for r in rows}
fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), facecolor=PAPER)
for ax, (sid, who) in zip(axes, EX):
    d = np.load(npz[sid])
    v, s = d['vol'], d['seg'].astype(int)
    rr, mm = split_top(v, s)
    panel(ax, v, s, mm, '%s（顱頂殘留 %.1f mm）' % (who, rr['total_mm']), size=(80, 36), overlay=True)
    ax.title.set_fontsize(15)
    ax.title.set_fontweight('bold')
fig.legend(handles=[Patch(facecolor=to_rgba(GREEN, 0.6), label='FreeSurfer 大腦皮質標籤'),
                    Line2D([], [], color=RRED, lw=2.5, label='殘留（本方法判定）')],
           loc='lower center', ncol=2, frameon=False, fontsize=13.5)
fig.tight_layout(rect=[0, 0.1, 1, 1])
save(fig, '1014_top_example.png')

# ── ③④ 殘留最多／最少各 3 位（test，排除 A0131）：頭頂（第 13 頁）、後腦杓（第 15 頁）同一個樣子 ─────────
# 2026-10-05 使用者要「像 top_compare.png 那種圖，但用老師的算法」；同一天又說第 13、15 頁在講同一件事，圖要統一。
# 頭頂的完整版（五個切面）是 check_skullstrip.py --show top --labels 3 42 -> models\skullstrip_check\top_compare_cortex.png；
# 那張太高，放進投影片字會小到看不到，這裡每位只留一個切面，數字放大
from check_skullstrip import pick, find, measure_top, measure_back, measure_base
from scipy import ndimage as ndi
import csv as _csv
SK = os.path.join(ROOT, 'models', 'skullstrip_check')


def rdcsv(p):
    with open(p, encoding='utf-8') as f:
        return {r['file'][:-4]: r for r in _csv.DictReader(f)}


AFT = rdcsv(os.path.join(ROOT, 'models', 'mix_exp3', 'dice_0240.csv'))
BEF = rdcsv(os.path.join(ROOT, 'models', 'mix_exp2', 'dice_baseline.csv'))


def six(metric, fname, prefix='皮質'):
    """metric：'top'、'back' 或 'base'。大字是「起點 → 配準後」的 Dice（只算殘留旁邊的結構，同 gather.py 的 residue_mm），
    下面是進步多少，最下面 30 個結構一起平均的「起點 → 配準後」。prefix：大字前面怎麼稱呼那些結構。"""
    worst, cleanest = pick(os.path.join(SK, 'skullstrip_all520.csv'), metric, 3, 'test', {'A0131'})
    labs = MM[metric]['labs']
    av = lambda row: float(np.nanmean([float(row['label_%d' % l]) for l in labs]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.7), facecolor=PAPER)
    for i, (grp, rr, col) in enumerate((('去顱骨不完全', worst, RUST), ('去顱骨完整', cleanest, TEAL))):
        for j, meta in enumerate(rr):
            s = meta['subject']
            d = np.load(find(os.path.join(ROOT, 'data', 'mixed_preprocessed_v2'), s)[0])
            vol, seg = d['vol'], d['seg']
            H, W = vol.shape[1], vol.shape[2]
            if metric == 'top':                                        # 冠狀中間那片，只留上半（同 top_compare.png 第 2 欄）
                mark = measure_top(vol, seg)[1]
                img, m2 = vol[:, H // 2, :].T, mark[:, H // 2, :].T
                keep = (slice(int(W * 0.56), None), slice(None))
                note = '顱頂殘留 %.2f mm' % float(meta['top_vertex_mm'])
            elif metric == 'base':                                     # 矢狀、穿過最大一坨殘留的中心（2026-10-05 加）
                # 顱底殘留＝離腦 10 mm 以外還亮著的東西，指標是最大一坨的體積；那坨左右位置每人不同，所以切面跟著它走，
                # 上下也以它為中心裁一半高度（大多在腦的前下方）
                mark = measure_base(vol, seg)[1]
                lab = ndi.label(mark)[0]
                pts = np.argwhere(lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1)
                x0, zc = int(np.median(pts[:, 0])), int(np.median(pts[:, 2]))
                img, m2 = vol[x0].T, mark[x0].T
                z0 = min(max(zc - W // 4, 0), W - W // 2)
                keep = (slice(z0, z0 + W // 2), slice(None))
                note = '顱底殘留 %s mm³' % format(int(float(meta['base_blob10'])), ',')
            else:                                                      # 軸狀、腦最後面那一層（同 back_compare.png 第 2 欄）
                mark = measure_back(vol, seg)[1]
                idx = np.argwhere(seg > 0)
                zb = int(np.median(idx[idx[:, 1] <= idx[:, 1].min() + 3][:, 2]))
                img, m2 = vol[:, :, zb].T, mark[:, :, zb].T
                # 只留最後面 42%：back_compare.png 留 55%，但那樣比頭頂那張高，圖下面的字會壓到下一排
                keep = (slice(None, int(H * 0.42)), slice(None))
                note = '枕部殘留 %.2f mm' % float(meta['back_occ_mm'])
            rgb = np.dstack([img] * 3)
            rgb[m2] = [1.0, 0.15, 0.1]
            ax = axes[i, j]
            ax.imshow(rgb[keep], origin='lower')
            ax.axis('off')
            ax.set_title('%s　%s' % (s, note), fontsize=12.5, color=INK)
            # 大字「起點 → 配準後」兩個一樣大：只放大配準後會被起點騙——後腦杓殘留多的 3 位起點就低，
            # 配準後看起來全部比較差，其實進步多少是交錯的（2026-10-05）
            ax.text(0.5, -0.04, '%s %.3f → %.3f' % (prefix, av(BEF[s]), av(AFT[s])), transform=ax.transAxes, ha='center', va='top',
                    fontsize=20, fontweight='bold', color=col)
            ax.text(0.5, -0.31, 'ΔDice %+.3f' % (av(AFT[s]) - av(BEF[s])), transform=ax.transAxes,
                    ha='center', va='top', fontsize=13, fontweight='bold', color=INK)
            ax.text(0.5, -0.50, '30 結構平均 %.3f → %.3f' % (float(BEF[s]['dice_mean']), float(AFT[s]['dice_mean'])),
                    transform=ax.transAxes, ha='center', va='top', fontsize=12, color=MUTED)
        axes[i, 0].text(-0.05, 0.5, grp, transform=axes[i, 0].transAxes, rotation=90, ha='right', va='center',
                        fontsize=17, fontweight='bold', color=col)
    fig.tight_layout(h_pad=7.5, rect=[0, 0.075, 1, 1])     # 圖下面的三行字 tight_layout 算不到，底部自己留空
    save(fig, fname)


six('top', '1014_six.png')
six('back', '1014_six_back.png')
six('base', '1014_six_base.png', prefix='鄰近結構')       # 顱底旁邊的結構有腦幹，不能叫皮質

# ── ⑤ 加寬：每個結構（2026-10-06 使用者要放進簡報）────────────────────────────────────────────
# 左：加寬的效果（速度場 wide_vel - exp6、位移場 wide - exp3）；右：加寬之後換版本（wide_vel - wide）
S = D.get('struct', {})
if all(e in S for e in ('mix_exp3', 'mix_exp6', 'mix_wide', 'mix_wide_vel')):
    names = sorted(S['mix_exp6'], key=lambda n: S['mix_wide_vel'][n] - S['mix_exp6'][n])
    wv = np.array([S['mix_wide_vel'][n] - S['mix_exp6'][n] for n in names])
    wdp = np.array([S['mix_wide'][n] - S['mix_exp3'][n] for n in names])
    ver = np.array([S['mix_wide_vel'][n] - S['mix_wide'][n] for n in names])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=PAPER, sharey=True)
    y = np.arange(len(names))
    a1.barh(y + 0.2, wv, height=0.38, color=TEAL, label='SVF（mix_wide_vel − mix_exp6）')
    a1.barh(y - 0.2, wdp, height=0.38, color='#D9895A', label='Displacement field（mix_wide − mix_exp3）')
    for yy, v in zip(y + 0.2, wv):
        a1.text(v + (0.0008 if v >= 0 else -0.0008), yy, '%+.3f' % v, va='center', ha='left' if v >= 0 else 'right',
                fontsize=10.5, color=TEAL, fontweight='bold')
    a2.barh(y, ver, height=0.55, color=['#2F7FD0' if v >= 0 else '#9AA09B' for v in ver])
    for yy, v in zip(y, ver):
        a2.text(v + (0.0005 if v >= 0 else -0.0005), yy, '%+.3f' % v, va='center', ha='left' if v >= 0 else 'right',
                fontsize=10.5, color=INK)
    lim1 = max(abs(wv).max(), abs(wdp).max()) + 0.009
    a1.set_xlim(-lim1, lim1)
    lim2 = abs(ver).max() + 0.006
    a2.set_xlim(-lim2, lim2)
    for ax in (a1, a2):
        ax.axvline(0, color=INK, lw=1)
        clean(ax)
        ax.grid(axis='y', alpha=0)
        ax.tick_params(axis='x', labelsize=11)
    a1.set_yticks(y)
    a1.set_yticklabels(names, fontsize=12.5)
    a1.set_xlabel('ΔDice（2× width − default width，左右平均）', fontsize=12.5)
    a1.set_title('加寬之效果：多數結構上升', fontsize=14, fontweight='bold')
    a1.legend(fontsize=11, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)   # 放圖外面，不壓到長條
    a2.set_xlabel('ΔDice（SVF − displacement field，皆為 2× width）', fontsize=12.5)
    a2.set_title('2× width 下兩種參數化之差異：各結構 %s 以內' % ('±%.3f' % abs(ver).max()), fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, '1014_wide_struct.png')

# ── ⑤ 加寬：越難對的人幫越多（兩個版本並排；09-20 那份第 25 頁只有位移場）─────────────────────────
WDF = D.get('wide_diff', {})
if 'vel' in WDF and 'disp' in WDF:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), facecolor=PAPER, sharey=True)
    for ax, key, title, col in ((axes[0], 'disp', 'Displacement field 加寬（mix_wide − mix_exp3）', '#D9895A'),
                                (axes[1], 'vel', 'SVF 加寬（mix_wide_vel − mix_exp6）', TEAL)):
        w = WDF[key]
        xb, yg = np.array(w['base']), np.array(w['gain'])
        ax.scatter(xb, yg, s=30, color=col, alpha=0.85, edgecolors='none')
        b1, b0 = np.polyfit(xb, yg, 1)
        xx = np.array([xb.min(), xb.max()])
        ax.plot(xx, b0 + b1 * xx, color=INK, lw=2, ls='--')
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title('%s\nr = %.2f（p %s）' % (title, w['r'], '< 0.001' if w['p'] < 0.001 else '= %.2f' % w['p']),
                     fontsize=13.5, fontweight='bold')
        ax.text(0.02, 0.97, 'Affine 最低 10 位：%+.4f' % w['hard10'], transform=ax.transAxes, ha='left', va='top',
                fontsize=12, fontweight='bold', color=RUST)
        ax.text(0.98, 0.97, 'Affine 最高 10 位：%+.4f' % w['easy10'], transform=ax.transAxes, ha='right', va='top',
                fontsize=12, fontweight='bold', color=TEAL)
        ax.set_xlabel('Affine Dice（形變配準前）　← 越左越難配準', fontsize=12)
        lo_, hi_ = min(min(WDF[k]['gain']) for k in WDF), max(max(WDF[k]['gain']) for k in WDF)
        ax.set_ylim(lo_ - 0.002, hi_ + 0.0065)              # 上面留空給「起點最差／最好 10 位」那兩行字
        ax.tick_params(labelsize=11)
        clean(ax)
    axes[0].set_ylabel('加寬之 ΔDice', fontsize=12.5)
    fig.tight_layout()
    save(fig, '1014_wide_difficulty.png')

# ── ⑤ 訓練 loss：四顆（2026-10-06 使用者要放進簡報）──────────────────────────────────────────
# 讀法同 ASD/plot_loss_curve.py（那支 import 時就解析命令列參數，正規表示式照抄）
import re as _re
_LINE = _re.compile(r'epoch:\s*(\d+)\s+step:\s*(\d+)/(\d+).*?loss:\s*(-?[\d.eE+-]+)\s+'
                    r'\((-?[\d.eE+-]+),\s*(-?[\d.eE+-]+)(?:,\s*(-?[\d.eE+-]+))?\)')


def loss_curve(e):
    raw = open(os.path.join(ROOT, 'log', e + '.txt'), 'rb').read()
    for enc in ('utf-16', 'utf-8', 'cp950'):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if '\ufffd' not in t and 'epoch' in t:
            break
    acc = {}
    for mm in _LINE.finditer(t):
        acc.setdefault(int(mm.group(1)), []).append((float(mm.group(5)), float(mm.group(6))))
    eps = sorted(acc)
    return np.array(eps), np.array([np.mean(acc[k], axis=0) for k in eps])


LOSS4 = [('mix_exp3', 'Displacement・default width', '#D9895A', '--'), ('mix_wide', 'Displacement・2× width', RUST, '-'),
         ('mix_exp6', 'SVF・default width', '#5BB8B6', '--'), ('mix_wide_vel', 'SVF・2× width', TEAL, '-')]
if all(os.path.exists(os.path.join(ROOT, 'log', e + '.txt')) for e, *_ in LOSS4):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.7), facecolor=PAPER)
    for e, lab, col, ls in LOSS4:
        ep, mv = loss_curve(e)
        a1.plot(ep, mv[:, 0], color=col, ls=ls, lw=2, label=lab)
        a2.plot(ep, mv[:, 1], color=col, ls=ls, lw=2, label=lab)
    a1.set_title('相似度項（−NCC；越低表示越相似）', fontsize=13.5, fontweight='bold')
    a2.set_title('平滑項（越低表示形變越平滑）', fontsize=13.5, fontweight='bold')
    for ax in (a1, a2):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.tick_params(labelsize=11)
        clean(ax)
    a1.set_ylim(-0.26, -0.15)
    a1.legend(fontsize=11.5, frameon=False, loc='upper right')
    fig.tight_layout()
    save(fig, '1014_wide_loss.png')

# ── ⑥ 架構修改（紅字以外；2026-10-06 加入簡報，10-07 依使用者要求改為正式用語。架構圖與公式在 make_arch.py）──────────────────────
from matplotlib.patches import FancyBboxPatch

TEAL_L = '#5BB8B6'
# 文獻依據（數字照論文抄，寫死；出處見 文獻/對位模型文獻筆記.md §1、§6）
# 左：Jian et al., "Mamba? Catch The Hype Or Rethink What Really Helps for Image Registration", WBIR 2024, Table 2 的 LPBA 欄
#     （訓練用 OASIS／ADNI／IXI，LPBA 40 人沒看過、隨機 200 對；DSC %）。2026-10-06 對過原文 HTML：
#     VXM 67.0、Mam-VXM 67.5、TM 67.3、DWP（兩張分開抽特徵＋每層先搬過去＋金字塔＝我們的第 2 步）70.4、DWCPI（四種全加）71.3
#     （只有兩張分開抽特徵 Dual 是 66.4，比原本還低：功勞在由粗到細，不在分開抽特徵）
LIT_MAMBA = [('VoxelMorph（baseline）', 67.0, '#9AA09B'), ('Mamba backbone', 67.5, '#7FA7D0'),
             ('Transformer backbone', 67.3, '#7FA7D0'), ('Coarse-to-fine（DWP）', 70.4, TEAL_L),
             ('All four designs（DWCPI）', 71.3, TEAL)]
LIT_LUMIR = [('SITReg（1st）', 0.785, TEAL, 'coarse-to-fine'), ('VFA', 0.777, TEAL, 'coarse-to-fine'),
             ('TransMorph', 0.762, '#7FA7D0', 'Transformer'), ('uniGradICON', 0.742, '#9AA09B', ''),
             ('SynthMorph', 0.722, '#9AA09B', ''), ('VoxelMorph', 0.714, RUST, 'baseline（本研究）'),
             ('ANTs SyN', 0.703, '#9AA09B', 'classical')]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), facecolor=PAPER, gridspec_kw={'width_ratios': [1, 1.15]})
y = np.arange(len(LIT_MAMBA))[::-1]
a1.barh(y, [v - 60 for _, v, _ in LIT_MAMBA], left=60, height=0.6, color=[c for *_, c in LIT_MAMBA])
for yy, (n, v, c) in zip(y, LIT_MAMBA):
    a1.text(v + 0.15, yy, '%.1f' % v, va='center', fontsize=13, fontweight='bold', color=TEAL if c in (TEAL, TEAL_L) else INK)
a1.text(67.75, 2.5, 'backbone 替換：< 1%', va='center', fontsize=12, color='#2F7FD0')
a1.set_yticks(y)
a1.set_yticklabels([n for n, *_ in LIT_MAMBA], fontsize=12.5)
a1.set_xlim(60, 73.5)
a1.set_xlabel('DSC (%)（LPBA，zero-shot；橫軸起點 60）', fontsize=11.5)
a1.set_title('Jian et al., WBIR 2024', fontsize=13.5, fontweight='bold')
y = np.arange(len(LIT_LUMIR))[::-1]
a2.barh(y, [v - 0.68 for _, v, _, _ in LIT_LUMIR], left=0.68, height=0.6, color=[c for _, _, c, _ in LIT_LUMIR])
for yy, (n, v, c, note) in zip(y, LIT_LUMIR):
    a2.text(v + 0.0015, yy, '%.3f%s' % (v, '　' + note if note else ''), va='center', fontsize=12,
            fontweight='bold' if note in ('coarse-to-fine', 'baseline（本研究）') else 'normal', color=c if c in (TEAL, RUST) else INK)
a2.set_yticks(y)
a2.set_yticklabels([n for n, *_ in LIT_LUMIR], fontsize=12.5)
a2.set_xlim(0.68, 0.825)
a2.set_xlabel('DSC（test, n = 590；橫軸起點 0.68）', fontsize=11.5)
a2.set_title('LUMIR 2024 test leaderboard', fontsize=13.5, fontweight='bold')
for ax in (a1, a2):
    clean(ax)
    ax.grid(axis='y', alpha=0)
    ax.tick_params(axis='x', labelsize=11)
fig.tight_layout(w_pad=3)
save(fig, '1014_arch_lit.png')

# Step 0：test-time recursion，同一模型遞迴 1、2、3 次（gather.py 的 multipass）
MPD = D.get('multipass', {})
GROUPS = [('mix_exp6', 'SVF'), ('mix_exp3', 'Displacement field'), ('mix_wide_vel', 'SVF, 2× width')]
if all(e in MPD for e, _ in GROUPS):
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=PAPER)
    PC = [('#B4B2A9', 1.0), (TEAL, 1.0), (TEAL, 0.4)]
    lo = 0.800
    pts = lambda v: '{:,.0f}'.format(v)
    for gi, (e, lab) in enumerate(GROUPS):
        ps = MPD[e]['passes']
        for k, p in enumerate(ps):
            x = gi * 4 + k
            col, al = PC[k]
            ax.bar(x, p['mean'] - lo, bottom=lo, width=0.85, color=col, alpha=al)
            ax.text(x, p['mean'] + 0.0003, '%.4f' % p['mean'], ha='center', va='bottom', fontsize=11,
                    fontweight='bold' if k == 1 else 'normal', color=TEAL if k == 1 else INK)
        tr = ax.get_xaxis_transform()                  # x 用資料座標、y 用軸的比例：字固定在軸下面
        ax.text(gi * 4 + 1, -0.04, '%s\n%s' % (e, lab), transform=tr, ha='center', va='top', fontsize=12, fontweight='bold')
        ax.text(gi * 4 + 1, -0.205, 'Folding（voxels / subject）\n' + ' → '.join(pts(p['points']) for p in ps),
                transform=tr, ha='center', va='top', fontsize=10.5, color=MUTED, linespacing=1.3)
    ax.set_xticks([])
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(lo, 0.8215)
    ax.set_ylabel('Test DSC（n = 51）', fontsize=11.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col, alpha=al, label='%d pass%s' % (k + 1, '（原模型）' if k == 0 else 'es'))
                       for k, (col, al) in enumerate(PC)], fontsize=11.5, frameon=False, loc='upper left', ncol=3)
    clean(ax)
    ax.grid(axis='x', alpha=0)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.97, bottom=0.25)
    save(fig, '1014_arch_step0.png')
