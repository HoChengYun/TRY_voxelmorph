# -*- coding: utf-8 -*-
"""Part 2：修改一（RCN＋local NCC）之訓練過程，與原作 RCN 並列
→ models\\deck_charts\\1111_train_ncc.png

兩欄（原作 RCN｜RCN＋local NCC），各欄縱軸獨立（相似度定義不同：1 − r vs −NCC）：
  上：相似度項、中：正則化項（三個 VTN 合計）、第三列：affine stem 之 det／ortho 損失（log 刻度）、下：val Dice（★ = 選定之 checkpoint）
細線 = 每一步，粗線 = 250 步移動平均；虛線 = lr 減半（1.5 萬、2 萬步）。
"""
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import J, INK, MUTED, GRAY, W, save, clean

PAT = re.compile(r'epoch:\s*(\d+)\s+step:\s*(\d+)/(\d+).*?loss:\s*(\S+)\s+\((\S+),\s*(\S+),\s*(\S+)\)')
COLS = [('orig_rcn_vtn3', 'RCN（原作）', 'global correlation', '#2F7FD0', 23000, '相似度項 1 − r'),
        ('rcn_vtn3_ncc', 'RCN＋local NCC（修改一）', '僅更換相似度項', '#D08C2B', 12000, '相似度項 −NCC')]
VAL_AFF = pd.read_csv(J('models', 'mix_exp3', 'dice_baseline_val.csv')).dice_mean.mean()


def read_log(d):
    t = open(J('models', d, 'train_log.txt'), encoding='utf-8', errors='replace').read()
    rows = [((int(m[1]) - 1) * int(m[3]) + int(m[2]), float(m[5]), float(m[6]), float(m[7].rstrip(')'))) for m in PAT.finditer(t)]
    return pd.DataFrame(rows, columns=['step', 'sim', 'reg', 'aff'])


fig, ax = plt.subplots(4, 2, figsize=(W, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 0.75, 0.75, 1.15]))
fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.07, hspace=0.12, wspace=0.2)
for j, (d, name, sub, col, best, simlab) in enumerate(COLS):
    L = read_log(d)
    x = L.step.values / 1000
    for r, (comp, lab, logy) in enumerate((('sim', simlab, False), ('reg', '正則化項', False), ('aff', 'affine 之 det／ortho', True))):
        a = ax[r, j]
        y = L[comp].values
        a.plot(x, y, color=col, lw=0.35, alpha=0.25)
        sm = pd.Series(y).rolling(250, min_periods=1, center=True).mean().values
        a.plot(x, sm, color=col, lw=1.6)
        if logy:
            a.set_yscale('log')
            a.set_ylim(5e-5, 0.3)
            a.set_yticks([1e-4, 1e-3, 1e-2, 1e-1], ['0.0001', '0.001', '0.01', '0.1'])     # 預設 10^-n 之負號字型缺字
            a.minorticks_off()
        else:
            lo, hi = np.percentile(y[300:], [0.2, 99.9])
            a.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        a.set_ylabel(lab, fontsize=9.5)
        clean(a)
    v = pd.read_csv(J('models', d, 'dice_curve_val.csv'))
    a = ax[3, j]
    a.plot(v.epoch / 1000, v.dice_mean, color=col, lw=1.8, marker='o', ms=3.5)
    b = v[v.epoch == best].iloc[0]
    a.scatter([best / 1000], [b.dice_mean], marker='*', s=170, color=col, edgecolor='white', linewidth=0.8, zorder=5)
    a.annotate('%.4f' % b.dice_mean, (best / 1000, b.dice_mean), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9, color=INK)
    lo = v.loc[v.dice_mean.idxmin()]
    if lo.dice_mean < 0.6:
        a.annotate('%.3f（第 {:,} 步）'.format(int(lo.epoch)) % lo.dice_mean, (lo.epoch / 1000, 0.6), xytext=(lo.epoch / 1000 + 1.2, 0.606), textcoords='data', fontsize=8.5, color=INK)
        a.annotate('', xy=(lo.epoch / 1000, 0.6), xytext=(lo.epoch / 1000, 0.635), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.2))
    a.axhline(VAL_AFF, color=GRAY, lw=1, ls=(0, (4, 3)))
    a.text(25.3, VAL_AFF + 0.003, 'affine %.4f' % VAL_AFF, ha='right', va='bottom', fontsize=8.5, color=MUTED)
    a.set_ylim(0.6, 0.81)
    a.set_ylabel('val Dice（30 結構）', fontsize=9.5)
    a.set_xlabel('訓練步數（× 1,000）', fontsize=9.5)
    a.set_xlim(0, 25.5)
    clean(a)
    for r in range(4):
        for xx in (15, 20):
            ax[r, j].axvline(xx, color=GRAY, lw=0.8, ls=':')
    ax[0, j].set_title('%s\n%s' % (name, sub), fontsize=10.5, color=col, fontweight='bold', loc='left')
fig.suptitle('訓練過程（縱軸各自獨立；細線 = 每一步，粗線 = 250 步移動平均；點線 = lr 減半；★ = 以 val Dice 選定之 checkpoint）',
             fontsize=10, color=MUTED, x=0.075, ha='left', y=0.985)
save(fig, '1111_train_ncc.png')
