# -*- coding: utf-8 -*-
"""10/28 簡報：⑥ 架構修改之結果（2 × 2 完成、延長訓練）、⑧ 與傳統方法（ANTs SyN）之比較、附錄（影像邊界之 folding）。
2026-10-21 使用者：「開做」（AI 帶回 mix_cascade_pyramid、mix_pyramid_500；ANTs SyN 51 位跑完）。

數字 → deck_data_arch.json（build.js 讀；投影片上的數字一律從這裡來，不手打）
圖   → models\\deck_charts\\1028_*.png
  1028_arch_res.png     五個模型 × 五項指標（Dice、未參與訓練之結構、HD95、SDlogJ、腦內 folding）
  1028_arch_2x2.png     每位受試者相對 mix_exp6 之 ΔDice（2× width、cascade、coarse-to-fine、兩者合併）
  1028_p500.png         coarse-to-fine 延長至 500 epochs：val Dice、訓練之相似度項
  1028_ants_paper.png   VoxelMorph 論文 Table I 與本研究：affine、ANTs SyN、VoxelMorph
  1028_bgfold.png       附錄：影像邊界 folding 之成因（一維示意）與只改推論之 padding 後之結果
ANTs 與 VoxelMorph 並排之受試者圖另見 make_ants_vis.py（要等 models\\ants_syn\\vis 跑完）。

資料來源
  test Dice           models\\<實驗>\\dice_<epoch>.csv（AI 上 test_dice.py 的正式結果）；affine = models\\mix_exp2\\dice_baseline.csv
  HD95、SDlogJ、folding  models\\<實驗>\\surface_<epoch>.csv（筆電 test_dice.py --surface；腦內 folding = jneg_fg_pct × atlas 非背景 voxel 數）
  未參與訓練之結構、NCC  models\\<實驗>\\semisup_check_<epoch>.csv（ASD/check_semisup.py）
  ANTs SyN            models\\ants_syn\\dice_syn_test.csv（ASD/ants_syn_baseline.py，論文 §V-A-3 之參數）
  影像邊界 folding     models\\folding_check\\bg\\bg_border_test.csv、bg_stats.csv、bg_levels.csv（ASD/check_bg_folding.py，5 位）
  val 曲線、訓練 loss  models\\mix_pyramid(_500)\\dice_curve_val.csv、log\\mix_pyramid(_500).txt
"""
import os
import re
import csv
import json

import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
J = lambda *p: os.path.join(ROOT, *p)
OUT = J('models', 'deck_charts')
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#D9D9D2'
TEAL, TEAL_L, BLUE, AMBER, RED, GRAY, PURPLE = '#0E7C7B', '#5BB8B6', '#2F7FD0', '#D08C2B', '#B03A2E', '#9A9A93', '#7A4FA0'
W = 12.13
FG_VOX = 1867705                     # atlas 非背景 voxel 數（jneg_fg_pct 之分母）
ALL_VOX = 192 * 224 * 192

# 論文 Table I（Balakrishnan et al., IEEE TMI 2019）：Dice、folding voxel 數、CPU／GPU 秒數
TAB1 = {'affine': 0.584, 'ants': 0.749, 'vxm': 0.753, 'ants_fold': 9662, 'vxm_fold': 19077, 'ants_sec': 9059, 'vxm_gpu_sec': 0.45}

MODELS = [('mix_exp6', '0190'), ('mix_wide_vel', '0240'), ('mix_cascade', '0170'), ('mix_pyramid', '0245'),
          ('mix_cascade_pyramid', '0140'), ('mix_pyramid_500', '0335'), ('mix_exp3', '0240')]
EP = dict(MODELS)


def rd(p):
    with open(p, encoding='utf-8') as fh:
        return {r['file']: r for r in csv.DictReader(fh)}


base = rd(J('models', 'mix_exp2', 'dice_baseline.csv'))
FILES = sorted(base)
col = lambda tab, k, scale=1.0: np.array([float(tab[f][k]) * scale for f in FILES])

V = {'affine': {'dice': col(base, 'dice_mean'), 'held': col(rd(J('models', 'semisup_check_affine.csv')), 'heldout_mean'),
                'ncc': col(rd(J('models', 'semisup_check_affine.csv')), 'ncc'),
                'hd95': col(rd(J('models', 'mix_exp2', 'surface_baseline.csv')), 'hd95_mean')}}
for e, ep in MODELS:
    d, s = rd(J('models', e, 'dice_%s.csv' % ep)), rd(J('models', e, 'surface_%s.csv' % ep))
    v = {'dice': col(d, 'dice_mean'), 'hd95': col(s, 'hd95_mean'), 'sdlogj': col(s, 'sdlogj'),
         'fold': col(s, 'jneg_fg_pct', FG_VOX / 100), 'fold_all': col(s, 'jneg_pct', ALL_VOX / 100)}
    p = J('models', e, 'semisup_check_%s.csv' % ep)
    if os.path.exists(p):
        q = rd(p)
        v['held'], v['ncc'] = col(q, 'heldout_mean'), col(q, 'ncc')
    V[e] = v
syn = rd(J('models', 'ants_syn', 'dice_syn_test.csv'))
assert set(syn) == set(FILES), 'ANTs SyN 的受試者跟 test 對不上'
V['ants'] = {'dice': col(syn, 'dice_mean'), 'fold': col(syn, 'jneg_fg_pct', FG_VOX / 100), 'sec': col(syn, 'sec')}


def params(e):
    import torch
    ck = torch.load(J('models', e, '%s.pt' % EP[e]), map_location='cpu', weights_only=False)
    return int(sum(t.numel() for k, t in ck['model_state'].items() if not k.endswith('.grid')))


def paired(a, b, k='dice', lower_better=False):
    x, y = V[a][k], V[b][k]
    d = x - y
    win = int((d < 0).sum() if lower_better else (d > 0).sum())
    return {'mean': float(d.mean()), 'win': win, 'n': len(d), 'p': float(wilcoxon(x, y).pvalue)}


OUT_D = {'n': len(FILES), 'paper': TAB1}
OUT_D['m'] = {e: {k: float(v.mean()) for k, v in V[e].items()} for e in V}
OUT_D['params'] = {e: params(e) for e in ['mix_exp6', 'mix_wide_vel', 'mix_cascade', 'mix_pyramid', 'mix_cascade_pyramid']}
P = {}
for e in ['mix_wide_vel', 'mix_cascade', 'mix_pyramid', 'mix_cascade_pyramid']:
    P[e + '_vs_exp6'] = {k: paired(e, 'mix_exp6', k, k in ('hd95', 'sdlogj')) for k in ('dice', 'held', 'ncc', 'hd95')}
P['cp_vs_cascade'] = paired('mix_cascade_pyramid', 'mix_cascade')
P['cp_vs_pyramid'] = paired('mix_cascade_pyramid', 'mix_pyramid')
P['pyramid_vs_cascade'] = paired('mix_pyramid', 'mix_cascade')
P['p500_vs_pyramid'] = paired('mix_pyramid_500', 'mix_pyramid')
P['cp_vs_p500'] = paired('mix_cascade_pyramid', 'mix_pyramid_500')
P['cp_vs_p500_held'] = paired('mix_cascade_pyramid', 'mix_pyramid_500', 'held')
for e in ['mix_exp3', 'mix_exp6', 'mix_cascade', 'mix_pyramid', 'mix_pyramid_500', 'mix_cascade_pyramid']:
    P[e + '_vs_ants'] = paired(e, 'ants')
OUT_D['p'] = P
g1, g2, g12 = (P[e + '_vs_exp6']['dice']['mean'] for e in ('mix_cascade', 'mix_pyramid', 'mix_cascade_pyramid'))
OUT_D['additive'] = {'cascade': g1, 'pyramid': g2, 'both': g12, 'ratio': g12 / (g1 + g2)}
order = np.argsort(V['affine']['dice'])
OUT_D['hard'] = {e: {'hard10': float((V[e]['dice'] - V['mix_exp6']['dice'])[order[:10]].mean()),
                     'easy10': float((V[e]['dice'] - V['mix_exp6']['dice'])[order[-10:]].mean())}
                 for e in ['mix_cascade', 'mix_pyramid', 'mix_cascade_pyramid']}

# ── 延長訓練：val 曲線（0～500）、訓練之相似度項 ─────────────────────────────
def val_curve(p):
    with open(p, encoding='utf-8') as fh:
        return {int(r['epoch']): float(r['dice_mean']) for r in csv.DictReader(fh)}


VC = val_curve(J('models', 'mix_pyramid', 'dice_curve_val.csv'))
VC.update({k: v for k, v in val_curve(J('models', 'mix_pyramid_500', 'dice_curve_val.csv')).items() if k > 250})
VCP = val_curve(J('models', 'mix_cascade_pyramid', 'dice_curve_val.csv'))
win = lambda C, a, b: float(np.mean([v for k, v in C.items() if a <= k <= b]))
OUT_D['p500'] = {'win': {'%d-%d' % (a, b): win(VC, a, b) for a, b in [(150, 200), (200, 250), (255, 300), (300, 400), (400, 500)]},
                 'best250': max(((k, v) for k, v in VC.items() if k <= 250), key=lambda t: t[1]),
                 'best500': max(VC.items(), key=lambda t: t[1]),
                 'cp_win': {'%d-%d' % (a, b): win(VCP, a, b) for a, b in [(150, 200), (200, 250)]}}

PAT = re.compile(r'epoch: (\d+)\s+step: \d+/\d+\s+time: ([\d.]+) sec\s+loss: (-?[\d.]+)\s+\((-?[\d.]+), (-?[\d.]+)\)')


def loss_by_epoch(name):
    raw = open(J('log', name + '.txt'), 'rb').read()
    t = None
    for enc in ('utf-8', 'utf-16', 'cp950'):
        try:
            c = raw.decode(enc)
        except Exception:
            continue
        if '�' not in c and 'epoch' in c:
            t = c
            break
    E = {}
    for m in PAT.finditer(t):
        E.setdefault(int(m.group(1)), []).append((float(m.group(2)), float(m.group(4))))
    return {e: (float(np.mean([x[1] for x in v])), float(np.sum([x[0] for x in v]))) for e, v in E.items()}


LP = loss_by_epoch('mix_pyramid')
LP.update(loss_by_epoch('mix_pyramid_500'))
LCP = loss_by_epoch('mix_cascade_pyramid')
OUT_D['p500']['img_loss'] = {'250': LP[250][0], '500': LP[500][0]}
OUT_D['hours'] = {'mix_pyramid_500': sum(LP[e][1] for e in LP if e > 250) / 3600,
                  'mix_cascade_pyramid': sum(v[1] for v in LCP.values()) / 3600}

# ── ANTs SyN ───────────────────────────────────────────────────────────
OUT_D['ants'] = {'sec_mean': float(V['ants']['sec'].mean()), 'sec_min': float(V['ants']['sec'].min()),
                 'sec_max': float(V['ants']['sec'].max()), 'threads': 20}

# ── 附錄：影像邊界之 folding ────────────────────────────────────────────
BT = list(csv.DictReader(open(J('models', 'folding_check', 'bg', 'bg_border_test.csv'), encoding='utf-8')))
BS = list(csv.DictReader(open(J('models', 'folding_check', 'bg', 'bg_stats.csv'), encoding='utf-8')))
bg = {}
for e in ['mix_pyramid', 'mix_pyramid_500', 'mix_cascade_pyramid']:
    b = [r for r in BT if r['model'] == e]
    s = [r for r in BS if r['model'] == e]
    near = sum(int(r['bg_border_0_0']) + int(r['bg_border_1_1']) + int(r['bg_border_2_4']) for r in s)
    bg[e] = {'n': len(b), 'bg_zeros': float(np.mean([float(r['bg_zeros']) for r in b])),
             'bg_border': float(np.mean([float(r['bg_border']) for r in b])),
             'fg_zeros': float(np.mean([float(r['fg_zeros']) for r in b])),
             'fg_border': float(np.mean([float(r['fg_border']) for r in b])),
             'dice_maxdiff': float(max(abs(float(r['dice_border']) - float(r['dice_zeros'])) for r in b)),
             'within4': near / max(1, sum(int(r['fold_bg']) for r in s)),
             'disp_border': float(np.mean([float(r['disp_border_mean']) for r in s])),
             'bg_test51': float((V[e]['fold_all'] - V[e]['fold']).mean())}
s6 = [r for r in BS if r['model'] == 'mix_exp6']
bg['mix_exp6'] = {'disp_border': float(np.mean([float(r['disp_border_mean']) for r in s6]))}
OUT_D['bgfold'] = bg

with open(os.path.join(HERE, 'deck_data_arch.json'), 'w', encoding='utf-8') as fh:
    json.dump(OUT_D, fh, ensure_ascii=False, indent=1)
print('->', os.path.join(HERE, 'deck_data_arch.json'))


# ══ 圖 ═══════════════════════════════════════════════════════════════
def save(fig, name, dpi=130):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def clean(ax, axis='x'):
    ax.grid(alpha=.3, axis=axis)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def dot_chart(name, rows, panels, h):
    """列 = 模型、欄 = 指標（同 2026-10-14_redpen/make_semisup.py 的 dot_chart）。"""
    fig, axes = plt.subplots(1, len(panels), figsize=(W, h), facecolor=PAPER, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=1 - 0.62 / h, bottom=0.3 / h, wspace=0.14)
    yv = -np.arange(len(rows), dtype=float)
    for ax, (title, note, k, kind, xl, fmt) in zip(axes, panels):
        span = xl[1] - xl[0]
        for yy, (lab, cfg, c, val) in zip(yv, rows):
            v = val.get(k)
            if v is None:
                ax.text(xl[0] + 0.02 * span, yy, '—', va='center', fontsize=10, color=MUTED)
                continue
            if kind == 'dot':
                ax.plot([xl[0], v], [yy, yy], color=c, lw=1, alpha=0.25)
                ax.plot(v, yy, 'o', color=c, ms=8)
            else:
                ax.barh(yy, v, height=0.6, color=c)
            inside = kind == 'bar' and v > 0.62 * span
            ax.text(v + (-0.015 if inside else (0.045 if kind == 'dot' else 0.02)) * span, yy, fmt(v), va='center',
                    ha='right' if inside else 'left', fontsize=10.5, color='white' if inside else INK)
        ax.set_xlim(*xl)
        ax.text(0, 1.0 + 0.36 / h, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=11.5, fontweight='bold', color=INK)
        if note:
            ax.text(0, 1.0 + 0.06 / h, note, transform=ax.transAxes, ha='left', va='bottom', fontsize=9.5, color=MUTED)
        ax.tick_params(axis='x', labelsize=9.5)
        ax.tick_params(axis='y', length=0)
        clean(ax)
    tr = axes[0].get_yaxis_transform()
    for yy, (lab, cfg, c, _) in zip(yv, rows):
        axes[0].text(-0.04, yy + 0.17, lab, transform=tr, ha='right', va='center', fontsize=11, fontweight='bold', color=INK)
        axes[0].text(-0.04, yy - 0.2, cfg, transform=tr, ha='right', va='center', fontsize=9.5, color=c)
    axes[0].set_yticks([])
    axes[0].set_ylim(yv[-1] - 0.75, yv[0] + 0.55)
    save(fig, name)


f2, f3 = (lambda v: '%.2f' % v), (lambda v: '%.3f' % v)
fN = lambda v: '0' if v == 0 else ('%.1f' % v if v < 10 else '{:,.0f}'.format(v))
M = OUT_D['m']

# ── 1. 五個模型 × 五項指標 ─────────────────────────────────────────────
ROWS5 = [('mix_exp6', 'VoxelMorph（SVF）', TEAL), ('mix_wide_vel', '2× width', TEAL_L), ('mix_cascade', 'Cascade（2 stages）', BLUE),
         ('mix_pyramid', 'Coarse-to-fine', AMBER), ('mix_cascade_pyramid', 'Cascade + coarse-to-fine', RED)]
rows = [(e, cfg, c, {k: M[e][k] for k in ('dice', 'held', 'hd95', 'sdlogj', 'fold')}) for e, cfg, c in ROWS5]
lo = lambda k, pad: min(r[3][k] for r in rows) - pad
hi = lambda k, pad: max(r[3][k] for r in rows) + pad
dot_chart('1028_arch_res.png', rows, [
    ('(a) Dice ↑', '30 結構；affine %.3f' % M['affine']['dice'], 'dice', 'dot', (lo('dice', 0.004), hi('dice', 0.006)), f3),
    ('(b) Dice ↑', '未參與訓練之結構；affine %.3f' % M['affine']['held'], 'held', 'dot', (lo('held', 0.012), hi('held', 0.014)), f3),
    ('(c) HD95（mm）↓', '30 結構；affine %.2f' % M['affine']['hd95'], 'hd95', 'dot', (lo('hd95', 0.03), hi('hd95', 0.05)), f2),
    ('(d) SDlogJ ↓', 'log|J| 之標準差（腦內）', 'sdlogj', 'bar', (0, hi('sdlogj', 0) * 1.45), f2),
    ('(e) Folding voxels ↓', '腦內，每位平均', 'fold', 'bar', (0, hi('fold', 0) * 1.45), fN),
], 3.3)

# ── 2. 每位受試者之 ΔDice（相對 mix_exp6）─────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.3), facecolor=PAPER)
fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.2)
rng = np.random.default_rng(0)
labels = []
for i, (e, cfg, c) in enumerate(ROWS5[1:]):
    d = V[e]['dice'] - V['mix_exp6']['dice']
    x = i + rng.uniform(-0.17, 0.17, len(d))
    ax.scatter(x, d, s=14, color=c, alpha=0.65, edgecolors='none')
    ax.plot([i - 0.28, i + 0.28], [d.mean()] * 2, color=INK, lw=2)
    ax.text(i + 0.31, d.mean(), '%+.4f' % d.mean(), va='center', fontsize=9.5, color=INK)
    labels.append(cfg.replace(' + ', '\n+ ').replace('（', '\n（'))
add = OUT_D['additive']
ax.plot([3 - 0.3, 3 + 0.3], [add['cascade'] + add['pyramid']] * 2, color=MUTED, lw=1.4, ls='--')
ax.text(3 - 0.32, add['cascade'] + add['pyramid'], '兩者增益之和\n%+.4f' % (add['cascade'] + add['pyramid']),
        ha='right', va='center', fontsize=8.5, color=MUTED)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_xlim(-0.5, 3.75)
ax.set_ylabel('ΔDice（相對 mix_exp6）', fontsize=10)
ax.set_title('每位受試者之 ΔDice（test %d 位；橫線 = 平均）' % len(FILES), fontsize=10.5, loc='left')
clean(ax, 'y')
save(fig, '1028_arch_2x2.png')

# ── 3. 延長訓練：val Dice 與訓練之相似度項 ────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.9), facecolor=PAPER)
fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.15, wspace=0.2)
ks = sorted(k for k in VC if k > 0)
a1.plot([k for k in ks if k <= 250], [VC[k] for k in ks if k <= 250], 'o-', ms=3, lw=1.2, color=AMBER, label='mix_pyramid（1～250）')
a1.plot([k for k in ks if k >= 250], [VC[k] for k in ks if k >= 250], 'o-', ms=3, lw=1.2, color=RED,
        label='mix_pyramid_500（自 epoch 250 接續）')
for (a, b) in [(150, 200), (200, 250), (300, 400), (400, 500)]:
    w_ = win(VC, a, b)
    a1.plot([a, b], [w_, w_], color=INK, lw=2.2)
    a1.text(b, w_ + 0.0004, '%.4f' % w_, ha='right', va='bottom', fontsize=8.5, color=INK, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85), zorder=6)
for k, c in [(OUT_D['p500']['best250'][0], AMBER), (OUT_D['p500']['best500'][0], RED)]:
    a1.plot(k, VC[k], '*', ms=15, color=c, markeredgecolor=INK, zorder=5)
a1.axvline(250, color=MUTED, ls=':', lw=1)
a1.set_ylim(0.800, 0.8195)
a1.set_xlabel('epoch'); a1.set_ylabel('val Dice（30 結構）')
a1.set_title('(a) val Dice（每 5 epochs；粗線 = 區間平均；★ = 選定之 epoch）', fontsize=10.5, loc='left')
a1.legend(fontsize=8.5, loc='lower right')
clean(a1, 'both')
le = sorted(LP)
a2.plot(le, [LP[e][0] for e in le], color=AMBER, lw=1.2, label='mix_pyramid／mix_pyramid_500')
a2.plot(sorted(LCP), [LCP[e][0] for e in sorted(LCP)], color=PURPLE, lw=1.2, label='mix_cascade_pyramid（250 epochs）')
a2.axvline(250, color=MUTED, ls=':', lw=1)
a2.set_ylim(-0.275, -0.235)
a2.set_xlabel('epoch'); a2.set_ylabel('相似度項（−NCC，訓練集）')
a2.set_title('(b) 訓練損失之相似度項（每 epoch 平均；越低越好）', fontsize=10.5, loc='left')
a2.legend(fontsize=8.5, loc='upper right')
clean(a2, 'both')
save(fig, '1028_p500.png')

# ── 4. 論文 Table I 與本研究 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor=PAPER)
fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.2)
groups = [('VoxelMorph 論文（Table I）\n8 個資料集，test 250 位', [('Affine', TAB1['affine'], GRAY), ('ANTs SyN', TAB1['ants'], PURPLE),
                                                         ('VoxelMorph', TAB1['vxm'], TEAL)]),
          ('本研究（mixed_v2）\ntest %d 位' % len(FILES), [('Affine', M['affine']['dice'], GRAY), ('ANTs SyN', M['ants']['dice'], PURPLE),
                                                     ('VoxelMorph', M['mix_exp3']['dice'], TEAL),
                                                     ('Cascade +\ncoarse-to-fine', M['mix_cascade_pyramid']['dice'], RED)])]
x0, ticks, tl = 0, [], []
fig.subplots_adjust(bottom=0.3)
for gname, bars in groups:
    for j, (lab, v, c) in enumerate(bars):
        ax.bar(x0 + j, v, width=0.78, color=c)
        ax.text(x0 + j, v + 0.005, '%.3f' % v, ha='center', va='bottom', fontsize=9.5, color=INK)
        ticks.append(x0 + j); tl.append(lab)
    ax.text(x0 + (len(bars) - 1) / 2, -0.2, gname, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9.5,
            color=INK, fontweight='bold')
    x0 += len(bars) + 0.8
ax.set_ylim(0.5, 0.86)
ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=8.5)
ax.set_ylabel('Dice')
ax.set_title('Affine、ANTs SyN、VoxelMorph（displacement field，λ = 1）之 Dice', fontsize=10.5, loc='left')
clean(ax, 'y')
save(fig, '1028_ants_paper.png')

# ── 5. 附錄：影像邊界之 folding ─────────────────────────────────────────
N, Uv, uv = 192, 2.1, 0.6
x = np.arange(N - 7, N)


def sample(U, p, mode):
    q = np.clip(p, 0, N - 1) if mode == 'border' else p
    i0 = np.floor(q).astype(int)
    w = q - i0
    get = lambda i: np.where((i >= 0) & (i <= N - 1), U[np.clip(i, 0, N - 1)], 0.0)
    return (1 - w) * get(i0) + w * get(i0 + 1)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.6), facecolor=PAPER, gridspec_kw={'width_ratios': [1, 1.15]})
fig.subplots_adjust(left=0.06, right=0.985, top=0.85, bottom=0.17, wspace=0.28)
for mode, c, lab in [('zeros', RED, "padding_mode = 'zeros'（訓練與評估之設定）"), ('border', BLUE, "padding_mode = 'border'")]:
    U = uv + sample(np.full(N, Uv), x + uv, mode)
    phi = x + U
    a1.plot(x, phi, 'o-', color=c, label=lab, ms=5)
    Jd = np.gradient(phi)
    for xi, pi, ji in zip(x, phi, Jd):
        if ji <= 0:
            a1.annotate(r'$\partial\phi/\partial x = %.2f$' % ji, (xi, pi), textcoords='offset points', xytext=(-12, -20), ha='right',
                        fontsize=9.5, color=c)
a1.axvline(N - 1, color=MUTED, ls=':', lw=1)
a1.text(N - 1 - 0.1, a1.get_ylim()[0] + 0.3, '影像邊界', ha='right', fontsize=9, color=MUTED)
a1.set_xlabel('x（voxel）'); a1.set_ylabel(r'$\phi(x) = x + U(x)$')
a1.set_title(r'(a) 一維示意：$U(x) = u(x) + U_{\uparrow}(x + u(x))$，$U_{\uparrow} = %.1f$、$u = %.1f$' % (Uv, uv),
             fontsize=10.5, loc='left')
a1.legend(fontsize=8.5, loc='upper left')
clean(a1, 'both')
NM = {'mix_pyramid': 'Coarse-to-fine', 'mix_pyramid_500': 'Coarse-to-fine\n（500 epochs）', 'mix_cascade_pyramid': 'Cascade +\ncoarse-to-fine'}
xs = np.arange(3)
z = [bg[e]['bg_zeros'] for e in NM]
b = [bg[e]['bg_border'] for e in NM]
a2.bar(xs - 0.2, z, 0.38, color=RED, label="'zeros'")
a2.bar(xs + 0.2, b, 0.38, color=BLUE, label="'border'（僅改推論）")
for i, (zz, bb) in enumerate(zip(z, b)):
    a2.text(i - 0.2, zz, '{:,.0f}'.format(zz), ha='center', va='bottom', fontsize=9.5)
    a2.text(i + 0.2, bb, '{:,.0f}'.format(bb), ha='center', va='bottom', fontsize=9.5)
a2.set_xticks(xs); a2.set_xticklabels(list(NM.values()), fontsize=9.5)
a2.set_ylabel('腦外 folding voxels（每位平均）')
a2.set_title('(b) 推論時改變 padding 之結果（%d 位；Dice 差異 ≤ %.5f）'
             % (bg['mix_cascade_pyramid']['n'], max(bg[e]['dice_maxdiff'] for e in NM)), fontsize=10.5, loc='left')
a2.legend(fontsize=8.5)
clean(a2, 'y')
save(fig, '1028_bgfold.png')
