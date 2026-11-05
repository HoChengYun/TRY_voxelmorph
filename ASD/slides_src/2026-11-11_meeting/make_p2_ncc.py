# -*- coding: utf-8 -*-
"""Part 2：修改一（RCN＋local NCC，11-04 帶回）之結果
→ models\\deck_charts\\1111_ncc_result.png

  (a) 主要結構之 Dice（test 51 位，左右合併）：affine｜RCN（原作）｜RCN＋local NCC｜VoxelMorph（mix_exp3）
  (b) 受試者小腦白質相對亮度 vs 形變後小腦白質體積：RCN（原作）vs RCN＋local NCC（atlas 皆為 MNI152；同 1111_rcn_mech.png (b) 之畫法）
資料：test CSV（dice_*.csv）；體積：models\\orig_rcn_vtn3\\vis\\per_subject_stats.csv（原作）、models\\rcn_vtn3_ncc\\vis\\group_stats.csv（第 3 級 = 最終輸出）；
      相對亮度：models\\orig_rcn_vtn3_subj\\vis\\intensity.csv（同 make_p2_rcn.py）。
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import J, MODELS, NCC, INK, MUTED, GRAY, W, save, clean

GROUPS = [('小腦白質', (7, 46)), ('小腦皮質', (8, 47)), ('大腦白質', (2, 41)), ('大腦皮質', (3, 42))]


def load(d, f):
    c = pd.read_csv(J('models', d, f))
    c['s'] = c.file.str[:-4]
    return c.set_index('s').sort_index()


SER = [('Affine', load('mix_exp3', 'dice_baseline.csv'), GRAY),
       (MODELS['rcn'][0], load(MODELS['rcn'][3], MODELS['rcn'][4]), MODELS['rcn'][2]),
       (NCC[0], load(NCC[3], NCC[4]), NCC[2]),
       ('VoxelMorph', load(MODELS['vxm'][3], MODELS['vxm'][4]), MODELS['vxm'][2])]

fig, ax = plt.subplots(1, 2, figsize=(W, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.15, wspace=0.2)

# ---- (a) 主要結構之 Dice ----
a = ax[0]
cols = GROUPS + [('30 結構平均', None)]
n = len(SER)
wd = 0.8 / n
for i, (nm, labs) in enumerate(cols):
    for j, (sn, c, colr) in enumerate(SER):
        v = (c.dice_mean if labs is None else c[['label_%d' % l for l in labs]].mean(axis=1)).values
        pos = i - 0.4 + wd * (j + 0.5)
        a.boxplot(v, positions=[pos], widths=wd * 0.85, patch_artist=True, showfliers=True,
                  flierprops=dict(marker='o', ms=2, mfc=colr, mec='none', alpha=0.6),
                  medianprops=dict(color='white', lw=1.2), whiskerprops=dict(color=colr, lw=0.9),
                  capprops=dict(color=colr, lw=0.9), boxprops=dict(facecolor=colr, edgecolor=colr))
a.axvline(len(GROUPS) - 0.5, color=GRAY, lw=0.8, ls=':')
a.set_xticks(np.arange(len(cols)), [c[0] for c in cols], fontsize=10)
a.set_xlim(-0.55, len(cols) - 0.45)
a.set_ylim(0.38, 0.97)
a.set_ylabel('Dice（左右合併）', fontsize=10)
for sn, _, colr in SER:
    a.scatter([], [], marker='s', s=55, color=colr, label=sn)
a.legend(loc='lower center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handletextpad=0.3)
a.set_title('(a) 主要結構之 Dice（test 51 位）', fontsize=10.5, loc='left', color=INK)
clean(a)

# ---- (b) 相對亮度 vs 形變後體積 ----
a = ax[1]
V1 = J('models', 'orig_rcn_vtn3_subj', 'vis')
it = pd.read_csv(os.path.join(V1, 'intensity.csv')).pivot(index='subject', columns='group', values='mean_intensity')
rel = it['cb_wm'] / it['wm']
subj = rel.drop(index=['sub-0036 (atlas)', 'MNI152 (atlas)'])
ps = pd.read_csv(J('models', 'orig_rcn_vtn3', 'vis', 'per_subject_stats.csv')).set_index('subject')
gs = pd.read_csv(J('models', NCC[3], 'vis', 'group_stats.csv'))
g3 = gs[(gs.stage == 3) & (gs.group == 'cb_wm')].set_index('subject')
for vol, col, lab in ((ps.rcn_vol_cb_wm, MODELS['rcn'][2], MODELS['rcn'][0]), (g3.vol_ratio, NCC[2], NCC[0])):
    y = vol.reindex(subj.index)
    assert y.notna().all(), lab
    rs = subj.corr(y, method='spearman')
    a.scatter(subj, y, s=22, color=col, edgecolor='white', linewidth=0.7, zorder=3,
              label='%s（中位數 %.2f；Spearman ρ = %s）' % (lab, y.median(), ('%.2f' % rs).replace('-', '−')))
xa = rel['MNI152 (atlas)']
a.axvline(xa, color=GRAY, lw=1.1, ls=(0, (4, 3)), zorder=2)
a.text(xa - 0.004, 2.12, 'MNI152 atlas\n%.3f' % xa, ha='right', va='top', fontsize=8.5, color=INK)
a.axhline(1, color=GRAY, lw=0.9)
a.set_xlim(0.69, 1.0)
a.set_ylim(0.8, 3.45)
a.set_xlabel('受試者之小腦白質相對亮度（小腦白質 ÷ 大腦白質）', fontsize=10)
a.set_ylabel('形變後小腦白質體積（atlas = 1）', fontsize=10)
a.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='none', fontsize=8.8, bbox_to_anchor=(0.965, 1.0))
a.set_title('(b) 相對亮度與形變後體積（每點一位；atlas 皆為 MNI152）', fontsize=10.5, loc='left', color=INK)
clean(a, grid='both')
save(fig, '1111_ncc_result.png')

# ---- 數字給 build.js（不手打）----
KEY = {'Affine': 'affine', MODELS['rcn'][0]: 'rcn', NCC[0]: 'ncc', 'VoxelMorph': 'vxm'}
grp = lambda c, labs: c[['label_%d' % l for l in labs]].mean(axis=1)     # noqa: E731
out = {'dice': {KEY[sn]: {nm: float(grp(c, labs).mean()) for nm, labs in GROUPS} for sn, c, _ in SER}}
cb = {KEY[sn]: grp(c, (7, 46)) for sn, c, _ in SER}
out['cb_wm_below_affine'] = {k: int((cb[k] < cb['affine']).sum()) for k in ('rcn', 'ncc', 'vxm')}
out['ctx_ncc_gt_vxm'] = int((grp(SER[2][1], (3, 42)) > grp(SER[3][1], (3, 42))).sum())
out['wm_ncc_gt_vxm'] = int((grp(SER[2][1], (2, 41)) > grp(SER[3][1], (2, 41))).sum())
out['cb_wm_vol_median'] = {'rcn': float(ps.rcn_vol_cb_wm.reindex(subj.index).median()), 'ncc': float(g3.vol_ratio.reindex(subj.index).median())}
out['cb_wm_vol_range_ncc'] = [float(g3.vol_ratio.min()), float(g3.vol_ratio.max())]
v = pd.read_csv(J('models', NCC[3], 'dice_curve_val.csv')).set_index('epoch').dice_mean
out['val_best'] = {'step': int(v.idxmax()), 'dice': float(v.max())}
out['val_min'] = {'step': int(v.idxmin()), 'dice': float(v.min())}
json.dump(out, open(J('models', 'deck_charts', '1111_ncc.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
