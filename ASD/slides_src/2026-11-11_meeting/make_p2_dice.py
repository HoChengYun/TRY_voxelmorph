# -*- coding: utf-8 -*-
"""Part 2：總表數字（含參數量，10/29 紅字 p15）與 Dice 兩張圖
→ models\\deck_charts\\1111_p2_metrics.json、1111_dice_subjects.png、1111_dice_struct_box.png

test 51 位、論文 30 個結構；folding 用 surface CSV 的 jneg_fg_pct（分母：MNI152 atlas 非背景）換成 voxel 數。
配對檢定一律 Wilcoxon signed-rank（同 10/28 簡報 make_arch_res.py 的 paired）。
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from style import J, OUT, MODELS, NCC, PARAMS, FG_VOX, INK, MUTED, GRAY, RULE, W, save, clean

KEYS = ('vxm', 'rcn', 'dwp')
LBL = [int(x) for x in np.load(J('voxelmorph-code', 'data', 'labels.npz'))['labels']]
STRUCT = [('大腦白質', (2, 41)), ('大腦皮質', (3, 42)), ('側腦室', (4, 43)), ('小腦白質', (7, 46)), ('小腦皮質', (8, 47)), ('視丘', (10, 49)),
          ('尾狀核', (11, 50)), ('殼核', (12, 51)), ('蒼白球', (13, 52)), ('海馬迴', (17, 53)), ('杏仁核', (18, 54)), ('腹側間腦', (28, 60)),
          ('脈絡叢', (31, 63)), ('第三腦室', (14,)), ('第四腦室', (15,)), ('腦幹', (16,)), ('腦脊髓液', (24,))]


def load(path):
    c = pd.read_csv(path)
    c['s'] = c.file.str[:-4]
    return c.set_index('s').sort_index()


aff = load(J('models', 'mix_exp3', 'dice_baseline.csv'))
T = {k: load(J('models', MODELS[k][3], MODELS[k][4])) for k in KEYS}
S = {k: load(J('models', MODELS[k][3], MODELS[k][5])) for k in KEYS}
for k in KEYS:
    assert list(T[k].index) == list(aff.index), k

out = {'affine': {'dice': float(aff.dice_mean.mean()), 'dice_sd': float(aff.dice_mean.std())}}
for k in KEYS:
    d = T[k].dice_mean
    o = dict(dice=float(d.mean()), dice_sd=float(d.std()), hd95=float(S[k].hd95_mean.mean()), sdlogj=float(S[k].sdlogj.mean()),
             fold_vox=float((S[k].jneg_fg_pct / 100 * FG_VOX).mean()), params=PARAMS[k],
             n_gt_affine=int((d > aff.dice_mean).sum()))
    if k != 'vxm':
        v = T['vxm'].dice_mean
        o['vs_vxm_n'] = int((d > v).sum())
        o['vs_vxm_p'] = float(wilcoxon(d, v).pvalue)
    out[k] = o
# 修改一（RCN＋local NCC）：只進總表，不進下面兩張圖
tn, sn = load(J('models', NCC[3], NCC[4])), load(J('models', NCC[3], NCC[5]))
assert list(tn.index) == list(aff.index)
dn = tn.dice_mean
out['ncc'] = dict(dice=float(dn.mean()), dice_sd=float(dn.std()), hd95=float(sn.hd95_mean.mean()), sdlogj=float(sn.sdlogj.mean()),
                  fold_vox=float((sn.jneg_fg_pct / 100 * FG_VOX).mean()), params=PARAMS['ncc'], n_gt_affine=int((dn > aff.dice_mean).sum()),
                  vs_vxm_n=int((dn > T['vxm'].dice_mean).sum()), vs_vxm_p=float(wilcoxon(dn, T['vxm'].dice_mean).pvalue),
                  vs_rcn_n=int((dn > T['rcn'].dice_mean).sum()), vs_rcn_p=float(wilcoxon(dn, T['rcn'].dice_mean).pvalue))
json.dump(out, open(J('models', 'deck_charts', '1111_p2_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, o in out.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in o.items()})

# ---- 每位受試者的 Dice：affine → 三個模型（細線連接同一位）----
fig, ax = plt.subplots(figsize=(W * 0.62, 4.6))
fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.12)
cols = [('affine', aff.dice_mean, GRAY)] + [(MODELS[k][0], T[k].dice_mean, MODELS[k][2]) for k in KEYS]
xs = np.arange(len(cols))
for i in range(len(aff)):
    ax.plot(xs, [c[1].iloc[i] for c in cols], color=GRAY, lw=0.45, alpha=0.45, zorder=1)
rng = np.random.RandomState(0)
for x, (n, v, c) in zip(xs, cols):
    ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=14, color=c, alpha=0.8, edgecolor='none', zorder=2)
    ax.plot([x - 0.22, x + 0.22], [v.mean()] * 2, color=INK, lw=2, zorder=3)
    ax.text(x + 0.26, v.mean(), '%.4f' % v.mean(), va='center', fontsize=9.5, color=INK)
ax.set_xticks(xs, [c[0] for c in cols], fontsize=10)
ax.set_xlim(-0.4, len(cols) - 0.3)
ax.set_ylabel('Dice（30 結構平均）', fontsize=10)
clean(ax)
ax.set_title('每位受試者之 Dice（test 51 位；黑線 = 平均，細線連接同一位）', fontsize=10, loc='left', color=INK)
save(fig, '1111_dice_subjects.png')

# ---- 各結構 Dice 箱形圖（左右合併；VoxelMorph TMI 2019 Fig. 5 之畫法），依 VoxelMorph 平均排序 ----


def merged(c, labs):
    return c[['label_%d' % l for l in labs]].mean(axis=1)


order = sorted(STRUCT, key=lambda s: merged(T['vxm'], s[1]).mean())
fig, ax = plt.subplots(figsize=(W, 4.4))
fig.subplots_adjust(left=0.06, right=0.995, top=0.86, bottom=0.14)
series = [('affine', aff, GRAY)] + [(MODELS[k][0], T[k], MODELS[k][2]) for k in KEYS]
n = len(series)
wd = 0.8 / n
for i, (nm, labs) in enumerate(order):
    for j, (mn, c, colr) in enumerate(series):
        v = merged(c, labs).values
        pos = i - 0.4 + wd * (j + 0.5)
        bp = ax.boxplot(v, positions=[pos], widths=wd * 0.85, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', ms=2, mfc=colr, mec='none', alpha=0.6),
                        medianprops=dict(color='white', lw=1.2), whiskerprops=dict(color=colr, lw=0.9),
                        capprops=dict(color=colr, lw=0.9), boxprops=dict(facecolor=colr, edgecolor=colr))
ax.set_xticks(np.arange(len(order)), [o[0] for o in order], fontsize=9.5)
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_ylabel('Dice', fontsize=10)
clean(ax)
for j, (mn, _, colr) in enumerate(series):
    ax.scatter([], [], marker='s', s=60, color=colr, label=mn)
ax.legend(loc='lower right', ncol=4, frameon=False, fontsize=9.5)
ax.set_title('各結構之 Dice（test 51 位，左右合併；依 VoxelMorph 平均由低至高排序）', fontsize=10, loc='left', color=INK)
save(fig, '1111_dice_struct_box.png')
