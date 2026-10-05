# -*- coding: utf-8 -*-
"""10/14 簡報的圖 -> models/deck_charts/1014_*.png。數字讀 deck_data.json（先跑 gather.py）。

  1014_folding_regions.png  ① 擠爆的點落在哪些區域（只留「佔幾 %」那一格、字放大）
  1014_lambda.png           ② 平滑權重 2 / 1 / 0.5：速度場 vs 位移場（還沒跑完的點標「跑中」）
  1014_dilution.png         ③ 30 個結構一起平均 vs 只平均殘留旁邊的結構：頭頂殘留厚度（mm） vs Dice 進步多少（170 人）
  1014_regions.png          ③④ 頭頂／顱底／後腦杓：殘留量 vs 只算殘留旁邊結構的 Dice 進步（三張散佈圖）
  1014_top_example.png      ③ 頭頂放大：殘留多的一位 vs 乾淨的一位（只放加標記的那張、字放大）
  1014_six.png              ③ top_compare.png 那 6 位：皮質 Dice 起點 → 配準後，附 30 個結構一起平均（簡報用精簡版）
  1014_six_back.png         ④ 同上，後腦杓殘留最多／最少各 3 位（只算大腦皮質、小腦皮質）
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
SERIES = [('速度場（全尺寸）', TEAL, {2.0: 'mix_exp5', 1.0: 'mix_exp6', 0.5: 'mix_exp7'}),
          ('位移場（全尺寸）', RUST, {2.0: 'mix_exp4', 1.0: 'mix_exp3'})]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), facecolor=PAPER)
jtxt = lambda y: '0%' if y == 0 else ('< 0.001%' if y < 0.001 else '%.3f%%' % y)   # 速度場權重 1、0.5 只有零星幾點
for ax, key, title in ((axes[0], 'mean', '對得多準（test Dice）'), (axes[1], 'jneg', '有沒有擠爆（擠爆點的比例 %）')):
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
                ax.annotate('%s\n跑中' % exps[w], (i, 0), xycoords=('data', 'axes fraction'), xytext=(0, 34),
                            textcoords='offset points', ha='center', fontsize=11, color=col,
                            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=col, ls='--', lw=1.2))
    ax.set_xticks(range(len(W)))
    ax.set_xticklabels(['平滑權重 %g' % w for w in W], fontsize=12)
    ax.set_xlim(-0.5, len(W) - 0.5)
    ax.set_title(title, fontsize=13.5, fontweight='bold')
    ax.annotate('', xy=(0.98, -0.16), xytext=(0.02, -0.16), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.2))
    ax.text(0.5, -0.22, '越往右，平滑的限制越鬆（形變可以捏得越用力）', transform=ax.transAxes, ha='center',
            fontsize=11, color=MUTED)
    clean(ax)
axes[0].set_ylim(0.795, 0.812)
axes[0].legend(fontsize=11.5, frameon=False, loc='upper left')
axes[1].set_ylim(0, 0.42)
axes[1].axhline(0.366, color=RED, ls='--', lw=1.2)
axes[1].text(len(W) - 0.55, 0.373, '論文的位移場版 0.366%', ha='right', va='bottom', fontsize=11, color=RED)
fig.tight_layout(rect=[0, 0.04, 1, 1])
save(fig, '1014_lambda.png')

# ── ② 平滑權重變小，哪些結構變好、哪些變差（跟權重 2 比）──────────────────────────
S = D.get('struct', {})
if all(e in S for e in ('mix_exp5', 'mix_exp6', 'mix_exp7')):
    names = sorted(S['mix_exp5'], key=lambda n: S['mix_exp7'][n] - S['mix_exp5'][n])
    d1 = np.array([S['mix_exp6'][n] - S['mix_exp5'][n] for n in names])
    d05 = np.array([S['mix_exp7'][n] - S['mix_exp5'][n] for n in names])
    fig, ax = plt.subplots(figsize=(9.2, 7.0), facecolor=PAPER)    # 簡報上放左半邊，圖小一點、字才不會縮太小
    y = np.arange(len(names))
    ax.barh(y + 0.2, d1, height=0.38, color='#5BB8B6', label='權重 1（mix_exp6）')
    ax.barh(y - 0.2, d05, height=0.38, color='#0A4F4E', label='權重 0.5（mix_exp7）')
    for yy, v in zip(y - 0.2, d05):
        ax.text(v + (0.0012 if v >= 0 else -0.0012), yy, '%+.3f' % v, va='center', ha='left' if v >= 0 else 'right',
                fontsize=12, color='#0A4F4E', fontweight='bold')
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=13.5)
    ax.tick_params(axis='x', labelsize=11.5)
    lim = max(abs(d05).max(), abs(d1).max()) + 0.014
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('跟平滑權重 2（mix_exp5）比，Dice 變多少（左右平均）', fontsize=13)
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
       '以前的算法：30 個結構一起平均', MM['top']['all_r'], MM['top']['all_p'], MUTED),
      (np.array([ctx(r['after']) - ctx(r['before']) for r in rows]),
       '老師的算法：只平均殘留旁邊的結構（左右大腦皮質）', MM['top']['r'], MM['top']['p'], RUST)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), facecolor=PAPER)
for ax, (y, title, rr, pp, col) in zip(axes, YS):
    ax.scatter(x, y, s=24, alpha=0.7, color=col, edgecolors='none')
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, b0 + b1 * xx, color=INK, lw=2)
    ax.set_title('%s\nr = %+.2f，%s' % (title, rr, ptxt(pp)), fontsize=15, fontweight='bold')
    ax.set_xlabel('頭頂殘留厚度（mm）', fontsize=13)
    ax.set_ylabel('Dice 進步多少（配準後 - 配準前）', fontsize=13)
    ax.tick_params(labelsize=12)                       # 投影片上這張縮到約 3.5 吋高，字要放大
    clean(ax)
same_span(axes, [y for y, *_ in YS])
fig.tight_layout()
save(fig, '1014_dilution.png')

# ── ③④ 三個位置：殘留量 vs 只算殘留旁邊結構的 Dice 進步（2026-10-05 起，取代原本 r 的長條）──────────────
LOC = [('top', '頭頂', 'top_vertex_mm', '殘留厚度（mm）'), ('base', '顱底', 'base_blob10', '最大一坨殘留的體積（mm³）'),
       ('back', '後腦杓', 'back_occ_mm', '殘留厚度（mm）')]
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
    ax.set_title('%s（只算%s）\nr = %+.2f，%s' % (nm, struct, MM[k]['r'], ptxt(MM[k]['p'])),
                 fontsize=13.5, fontweight='bold', color=INK)
    ax.set_xlabel(xl, fontsize=12.5)
    ax.tick_params(labelsize=11.5)
    clean(ax)
    ys.append(yk)
axes[0].set_ylabel('Dice 進步多少（配準後 - 配準前）', fontsize=12.5)
same_span(axes, ys)
fig.tight_layout()
save(fig, '1014_regions.png')

# ── ① 擠爆的點落在哪些區域（佔幾 %）─────────────────────────────────────────
FOLD = D['folding']
REG = ['大腦皮質', '大腦白質', '腦內、沒有標籤', '腦室・腦脊髓液・脈絡叢', '小腦・腦幹', '深部灰質・海馬・杏仁核', '腦外（背景）']
SHOW = {'大腦皮質': '大腦皮質', '大腦白質': '大腦白質', '腦內、沒有標籤': '腦內沒有標籤（多是腦溝空隙）',
        '腦室・腦脊髓液・脈絡叢': '腦室・腦脊髓液', '小腦・腦幹': '小腦・腦幹',
        '深部灰質・海馬・杏仁核': '深部灰質・海馬・杏仁核', '腦外（背景）': '腦外'}
FCOL = {'mix_exp4': '#D9895A', 'mix_exp3': '#A34F1B', 'mix_wide': '#6B3FA0'}
FLAB = {'mix_exp4': '位移場・平滑權重 2', 'mix_exp3': '位移場・平滑權重 1', 'mix_wide': '位移場・平滑權重 1・加寬'}
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
ax.set_xlabel('擠爆的點有幾 % 落在這一區（數字是平滑權重 1 那顆）', fontsize=12)
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
EX = [('MRS0381-2', '殘留多的人'), ('T054', '乾淨的人')]      # 同 check_top_residue.py --example
npz = {r['subject']: r['npz'] for r in rows}
fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), facecolor=PAPER)
for ax, (sid, who) in zip(axes, EX):
    d = np.load(npz[sid])
    v, s = d['vol'], d['seg'].astype(int)
    rr, mm = split_top(v, s)
    panel(ax, v, s, mm, '%s（頭頂殘留 %.1f mm）' % (who, rr['total_mm']), size=(80, 36), overlay=True)
    ax.title.set_fontsize(15)
    ax.title.set_fontweight('bold')
fig.legend(handles=[Patch(facecolor=to_rgba(GREEN, 0.6), label='FreeSurfer 畫成皮質'),
                    Line2D([], [], color=RRED, lw=2.5, label='程式算成「殘留」')],
           loc='lower center', ncol=2, frameon=False, fontsize=13.5)
fig.tight_layout(rect=[0, 0.1, 1, 1])
save(fig, '1014_top_example.png')

# ── ③④ 殘留最多／最少各 3 位（test，排除 A0131）：頭頂（第 13 頁）、後腦杓（第 15 頁）同一個樣子 ─────────
# 2026-10-05 使用者要「像 top_compare.png 那種圖，但用老師的算法」；同一天又說第 13、15 頁在講同一件事，圖要統一。
# 頭頂的完整版（五個切面）是 check_skullstrip.py --show top --labels 3 42 -> models\skullstrip_check\top_compare_cortex.png；
# 那張太高，放進投影片字會小到看不到，這裡每位只留一個切面，數字放大
from check_skullstrip import pick, find, measure_top, measure_back
import csv as _csv
SK = os.path.join(ROOT, 'models', 'skullstrip_check')


def rdcsv(p):
    with open(p, encoding='utf-8') as f:
        return {r['file'][:-4]: r for r in _csv.DictReader(f)}


AFT = rdcsv(os.path.join(ROOT, 'models', 'mix_exp3', 'dice_0240.csv'))
BEF = rdcsv(os.path.join(ROOT, 'models', 'mix_exp2', 'dice_baseline.csv'))


def six(metric, fname):
    """metric：'top' 或 'back'。大字是「起點 → 配準後」的 Dice（只算殘留旁邊的結構，同 gather.py 的 residue_mm），
    下面是進步多少，最下面 30 個結構一起平均的「起點 → 配準後」。"""
    worst, cleanest = pick(os.path.join(SK, 'skullstrip_all520.csv'), metric, 3, 'test', {'A0131'})
    labs = MM[metric]['labs']
    av = lambda row: float(np.nanmean([float(row['label_%d' % l]) for l in labs]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.7), facecolor=PAPER)
    for i, (grp, rr, col) in enumerate((('沒去乾淨', worst, RUST), ('去得乾淨', cleanest, TEAL))):
        for j, meta in enumerate(rr):
            s = meta['subject']
            d = np.load(find(os.path.join(ROOT, 'data', 'mixed_preprocessed_v2'), s)[0])
            vol, seg = d['vol'], d['seg']
            H, W = vol.shape[1], vol.shape[2]
            if metric == 'top':                                        # 冠狀中間那片，只留上半（同 top_compare.png 第 2 欄）
                mark = measure_top(vol, seg)[1]
                img, m2 = vol[:, H // 2, :].T, mark[:, H // 2, :].T
                keep = (slice(int(W * 0.56), None), slice(None))
                note = '頭頂殘留 %.2f mm' % float(meta['top_vertex_mm'])
            else:                                                      # 軸狀、腦最後面那一層（同 back_compare.png 第 2 欄）
                mark = measure_back(vol, seg)[1]
                idx = np.argwhere(seg > 0)
                zb = int(np.median(idx[idx[:, 1] <= idx[:, 1].min() + 3][:, 2]))
                img, m2 = vol[:, :, zb].T, mark[:, :, zb].T
                # 只留最後面 42%：back_compare.png 留 55%，但那樣比頭頂那張高，圖下面的字會壓到下一排
                keep = (slice(None, int(H * 0.42)), slice(None))
                note = '後腦杓殘留 %.2f mm' % float(meta['back_occ_mm'])
            rgb = np.dstack([img] * 3)
            rgb[m2] = [1.0, 0.15, 0.1]
            ax = axes[i, j]
            ax.imshow(rgb[keep], origin='lower')
            ax.axis('off')
            ax.set_title('%s　%s' % (s, note), fontsize=12.5, color=INK)
            # 大字「起點 → 配準後」兩個一樣大：只放大配準後會被起點騙——後腦杓殘留多的 3 位起點就低，
            # 配準後看起來全部比較差，其實進步多少是交錯的（2026-10-05）
            ax.text(0.5, -0.04, '皮質 %.3f → %.3f' % (av(BEF[s]), av(AFT[s])), transform=ax.transAxes, ha='center', va='top',
                    fontsize=20, fontweight='bold', color=col)
            ax.text(0.5, -0.31, '進步 %+.3f' % (av(AFT[s]) - av(BEF[s])), transform=ax.transAxes,
                    ha='center', va='top', fontsize=13, fontweight='bold', color=INK)
            ax.text(0.5, -0.50, '30 個結構一起平均 %.3f → %.3f' % (float(BEF[s]['dice_mean']), float(AFT[s]['dice_mean'])),
                    transform=ax.transAxes, ha='center', va='top', fontsize=12, color=MUTED)
        axes[i, 0].text(-0.05, 0.5, grp, transform=axes[i, 0].transAxes, rotation=90, ha='right', va='center',
                        fontsize=17, fontweight='bold', color=col)
    fig.tight_layout(h_pad=7.5, rect=[0, 0.075, 1, 1])     # 圖下面的三行字 tight_layout 算不到，底部自己留空
    save(fig, fname)


six('top', '1014_six.png')
six('back', '1014_six_back.png')
