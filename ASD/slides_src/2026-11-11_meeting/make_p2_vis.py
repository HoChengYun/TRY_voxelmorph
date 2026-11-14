# -*- coding: utf-8 -*-
"""Part 2（11-11 使用者：p11 用圖比字好、各結構之 Dice 與體積拿回來）
→ models\\deck_charts\\1111_p2_box.png、1111_p2_struct.png、1111_p2_vis.json（真人 atlas 重現之數字）

1111_p2_box.png：每位受試者之 Dice（箱形圖＋每點一位），右側附 Dice、HD95、folding、參數量（p15 紅字）。
  模型：affine、VoxelMorph（mix_exp3）、RCN（原作重現）、DWP（原作重現，第 32 輪）、RCN＋local NCC（修改一）。
  每位 Dice 從 test CSV 讀；彙總數字從 1111_p2_metrics.json 讀（make_p2_dice.py 產生），不手打。
1111_p2_struct.png：17 個結構（左右合併）之 Dice 與形變後體積（atlas = 1）。
  資料：models\\orig_dwp\\vis_0032\\per_structure_summary.csv（Dice = 51 位平均、體積 = 51 位中位數；DWP 為第 32 輪）。
  做法同 models\\orig_dwp\\vis_0032\\03_per_structure_dice_volume.png，改為正式用語、配色同本簡報。
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import J, OUT, MODELS, NCC, INK, MUTED, GRAY, RULE, W, save, clean, heat_t, heat_cell

MET = json.load(open(os.path.join(OUT, '1111_p2_metrics.json'), encoding='utf-8'))


def load(d, f):
    c = pd.read_csv(J('models', d, f))
    c['s'] = c.file.str[:-4]
    return c.set_index('s').sort_index().dice_mean


# ════ 每位受試者之 Dice ════
ROWS = [('affine', 'Affine（起點）', load('mix_exp3', 'dice_baseline.csv'), GRAY),
        ('rcn', 'RCN（原作重現）', load(MODELS['rcn'][3], MODELS['rcn'][4]), MODELS['rcn'][2]),
        ('ncc', 'RCN＋local NCC（修改一）', load(NCC[3], NCC[4]), NCC[2]),
        ('vxm', 'VoxelMorph（mix_exp3）', load(MODELS['vxm'][3], MODELS['vxm'][4]), MODELS['vxm'][2]),
        ('dwp', 'DWP（原作重現）', load(MODELS['dwp'][3], MODELS['dwp'][4]), MODELS['dwp'][2])]
for k, _, v, _ in ROWS:
    assert abs(v.mean() - MET[k]['dice']) < 1e-6, k      # 每位 Dice 與彙總表一致
    assert len(v) == 51, k

fig = plt.figure(figsize=(W, 4.2))
ax = fig.add_axes([0.19, 0.13, 0.47, 0.8])
rng = np.random.RandomState(0)
for i, (k, lab, v, c) in enumerate(ROWS):
    ax.boxplot(v.values, positions=[i], vert=False, widths=0.52, patch_artist=True, showfliers=False,
               medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=c, lw=1.1), capprops=dict(color=c, lw=1.1),
               boxprops=dict(facecolor=c + '33', edgecolor=c, lw=1.2), zorder=2)
    ax.scatter(v.values, i + rng.uniform(-0.17, 0.17, len(v)), s=10, color=c, alpha=0.75, edgecolor='none', zorder=3)
ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS], fontsize=10.5)
for t, r in zip(ax.get_yticklabels(), ROWS):
    t.set_color(INK)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlim(0.62, 0.86)
ax.set_xlabel('Dice（test 51 位；論文之 30 個結構平均；每點一位）', fontsize=10, color=MUTED)
clean(ax, grid='x')
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)

# 右側數字欄（與每一列對齊）
# 每欄：(x, 欄名, 格式, 取值, 較佳方向, 是否取 log)；底色只比配準後之模型（affine 不上色），參數量不分好壞、不上色
COLS = [(0.705, 'Dice', lambda o: '%.4f' % o['dice'], 'dice', 'high', False), (0.785, 'HD95（mm）', lambda o: '%.2f' % o['hd95'], 'hd95', 'low', False),
        (0.875, 'Folding\n（voxel／位）', lambda o: format(round(o['fold_vox']), ',') if o['fold_vox'] >= 10 else '%.1f' % o['fold_vox'], 'fold_vox', 'low', True),
        (0.955, '參數量', lambda o: '%.2f M' % (o['params'] / 1e6), 'params', None, False)]
REG = [k for k, *_ in ROWS if k != 'affine']
for x, head, fn, key, better, lg in COLS:
    fig.text(x, 0.955, head, ha='center', va='top', fontsize=9.5, color=MUTED, fontweight='bold', linespacing=1.15)
    tt = dict(zip(REG, heat_t([MET[k][key] for k in REG], better, lg))) if better else {}
    for i, (k, lab, v, c) in enumerate(ROWS):
        y = ax.transData.transform((0, i))[1]
        y = fig.transFigure.inverted().transform((0, y))[1]
        txt = '—' if k == 'affine' and head != 'Dice' else fn(MET[k])
        if k in tt:
            heat_cell(fig, x, y, txt, tt[k])
        else:
            fig.text(x, y, txt, ha='center', va='center', fontsize=10.5, color=INK)
fig.text(0.83, 0.02, '底色：同一欄內比較，深 = 較佳；粗體 = 該欄最佳', ha='center', fontsize=8.5, color=MUTED)
save(fig, '1111_p2_box.png')

# ════ 各結構之 Dice 與形變後體積 ════
S = pd.read_csv(J('models', 'orig_dwp', 'vis_0032', 'per_structure_summary.csv'), encoding='utf-8-sig').set_index('結構')
assert len(S) == 17
S = S.loc[(S.D_rcn - S.D_dwp).sort_values().index]          # RCN 落後 DWP 越多越上面
SER = [('affine', 'Affine（起點）', GRAY, 's'), ('exp3', 'VoxelMorph', MODELS['vxm'][2], 'D'),
       ('rcn', 'RCN（原作重現）', MODELS['rcn'][2], 'o'), ('dwp', 'DWP（原作重現）', MODELS['dwp'][2], '^')]
fig, axs = plt.subplots(1, 2, figsize=(W, 5.0), sharey=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(left=0.085, right=0.975, top=0.86, bottom=0.1, wspace=0.06)
y = np.arange(len(S))[::-1]
for a, pre, lim, ttl in ((axs[0], 'D_', (0.25, 0.95), 'Dice（51 位平均）'), (axs[1], 'V_', (0.75, 1.85), '形變後體積（atlas 之同一結構 = 1；51 位中位數）')):
    vals = S[[pre + k for k, *_ in SER]].values
    a.hlines(y, vals.min(1), vals.max(1), color=RULE, lw=2.2, zorder=1)
    for k, lab, c, m in SER:
        a.scatter(S[pre + k], y, marker=m, s=46, color=c, edgecolor='white', linewidth=0.8, zorder=3, label=lab)
    a.set_xlim(*lim)
    a.set_title(ttl, fontsize=10.5, loc='left', color=INK)
    clean(a, grid='x')
    a.tick_params(axis='y', length=0)
axs[1].axvline(1, color=MUTED, lw=1, zorder=0)
r_cb = S.loc['小腦白質', 'V_rcn']
axs[1].text(r_cb + 0.025, y[list(S.index).index('小腦白質')], '%.2f 倍' % r_cb, va='center', fontsize=9.5, color=INK)
axs[0].set_yticks(y, list(S.index), fontsize=10)
axs[0].spines['left'].set_visible(False)
axs[1].spines['left'].set_visible(False)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
save(fig, '1111_p2_struct.png')

# ════ 真人 atlas（sub-0036）重現之數字（給 build.js，不手打）════
def per_label(d, f):
    c = pd.read_csv(J('models', d, f))
    c['s'] = c.file.str[:-4]
    return c.set_index('s').sort_index()


sb = per_label('orig_rcn_vtn3_subj', 'dice_baseline.csv')
sr = per_label('orig_rcn_vtn3_subj', 'dice_023000.csv')
assert list(sb.index) == list(sr.index) and len(sb) == 51
labs = [c for c in sb.columns if c.startswith('label_')]
SUBJ = dict(affine=float(sb.dice_mean.mean()), rcn=float(sr.dice_mean.mean()), n_gt=int((sr.dice_mean > sb.dice_mean).sum()),
            n_struct_below_affine=int((sr[labs].mean() < sb[labs].mean()).sum()), n_struct=len(labs))
gs = pd.read_csv(J('models', 'orig_rcn_vtn3_subj', 'vis', 'group_stats.csv'))
SUBJ['cb_wm_vol_median'] = float(gs[(gs.stage == 'stage3') & (gs.group == 'cb_wm')].vol_ratio.median())   # 同 make_p2_rcn.py
json.dump({'subj': SUBJ}, open(os.path.join(OUT, '1111_p2_vis.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[OK]', os.path.join(OUT, '1111_p2_vis.json'), SUBJ)
