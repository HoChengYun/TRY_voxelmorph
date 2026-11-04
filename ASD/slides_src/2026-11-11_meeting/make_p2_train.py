# -*- coding: utf-8 -*-
"""Part 2：三個模型的訓練過程，各畫各的（11-03 使用者：「loss 圖可以各放各的，看他訓練過程的狀態」）
→ models\\deck_charts\\1111_train_curves.png

每一欄一個模型；上：相似度項、中：正則化項（訓練 log 每一步的值，250 步移動平均）、下：val Dice（★ = 選定之 checkpoint）。
三個模型的損失定義不同（VoxelMorph、DWP：−local NCC；RCN：1 − global correlation），縱軸各自獨立，不跨欄比較。
  VoxelMorph：log\\mix_exp3.txt 的 loss: 總計 (相似度, λ·平滑)
  RCN       ：models\\orig_rcn_vtn3\\train_log.txt 的 loss: 總計 (1 − r, 平滑, det/ortho)；lr 在 1.5 萬、2 萬步各減半
  DWP       ：models\\orig_dwp\\train_log.txt 的 loss: 總計 (最細層相似度, 最細層平滑)；總計另含粗層與反方向，不畫
"""
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import J, MODELS, INK, MUTED, GRAY, W, save, clean

PAT = re.compile(r'epoch:\s*(\d+)\s+step:\s*(\d+)/(\d+).*?loss:\s*([-\d.]+)\s+\(([-\d.]+),\s*([-\d.]+)')


def read_log(p):
    raw = open(p, 'rb').read()
    for enc in ('utf-8', 'utf-16', 'cp950'):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if '�' not in t and 'epoch' in t:
            break
    rows = []
    for line in t.splitlines():
        m = PAT.search(line)
        if m:
            ep, st, per = int(m.group(1)), int(m.group(2)), int(m.group(3))
            rows.append(((ep - 1) * per + st, float(m.group(5)), float(m.group(6))))
    return pd.DataFrame(rows, columns=['step', 'sim', 'reg']).sort_values('step')


LOGS = {'vxm': J('log', 'mix_exp3.txt'), 'rcn': J('models', 'orig_rcn_vtn3', 'train_log.txt'), 'dwp': J('models', 'orig_dwp', 'train_log.txt')}
SIM_LABEL = {'vxm': '相似度項 −NCC', 'rcn': '相似度項 1 − r（global）', 'dwp': '相似度項 −NCC（最細層）'}
REG_LABEL = {'vxm': r'正則化項 $\lambda\,\|\nabla u\|^2$', 'rcn': '正則化項（三個 VTN 合計）', 'dwp': '正則化項（最細層）'}
STEP_PER_X = {'vxm': 100, 'rcn': 1, 'dwp': 418}          # val 曲線的 x 欄換成步數
VAL_AFFINE = pd.read_csv(J('models', 'mix_exp3', 'dice_baseline_val.csv')).dice_mean.mean()

fig, ax = plt.subplots(3, 3, figsize=(W, 7.4), sharex='col', gridspec_kw=dict(height_ratios=[1, 0.8, 1.15]))
fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.08, hspace=0.12, wspace=0.28)
summary = {}
for j, k in enumerate(('vxm', 'rcn', 'dwp')):
    name, sub, col, d, _, _, vcsv, best = MODELS[k]
    L = read_log(LOGS[k])
    x = L.step.values / 1000
    for r, comp, lab in ((0, 'sim', SIM_LABEL[k]), (1, 'reg', REG_LABEL[k])):
        a = ax[r, j]
        y = L[comp].values
        a.plot(x, y, color=col, lw=0.4, alpha=0.18)
        sm = pd.Series(y).rolling(250, min_periods=1, center=True).mean().values
        a.plot(x, sm, color=col, lw=1.8)
        a.set_ylabel(lab, fontsize=9.5)
        lo, hi = np.percentile(sm[200:], [0.5, 99.5])
        pad = (hi - lo) * 0.12
        a.set_ylim(lo - pad, hi + pad)
        clean(a)
    v = pd.read_csv(J('models', d, vcsv))
    v['step'] = v.epoch * STEP_PER_X[k]
    a = ax[2, j]
    a.plot(v.step / 1000, v.dice_mean, color=col, lw=1.8, marker='o' if len(v) <= 30 else None, ms=3.5)
    b = v.loc[(v.step - best).abs().idxmin()]
    a.scatter([b.step / 1000], [b.dice_mean], marker='*', s=170, color=col, edgecolor='white', linewidth=0.8, zorder=5)
    a.annotate('%.4f' % b.dice_mean, (b.step / 1000, b.dice_mean), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9, color=INK)
    a.axhline(VAL_AFFINE, color=GRAY, lw=1, ls=(0, (4, 3)))
    a.text(25, VAL_AFFINE + 0.003, 'affine %.4f' % VAL_AFFINE, ha='right', va='bottom', fontsize=8.5, color=MUTED)
    a.set_ylim(0.66, 0.83)
    a.set_ylabel('val Dice（30 結構）', fontsize=9.5)
    a.set_xlabel('訓練步數（× 1,000）', fontsize=9.5)
    a.set_xlim(0, 25.5)
    clean(a)
    if k == 'rcn':
        for xx in (15, 20):
            for r in range(3):
                ax[r, j].axvline(xx, color=GRAY, lw=0.8, ls=':')
        ax[0, j].text(15.2, ax[0, j].get_ylim()[1], 'lr × 0.5', fontsize=8, color=MUTED, va='top')
        ax[0, j].text(20.2, ax[0, j].get_ylim()[1], 'lr × 0.5', fontsize=8, color=MUTED, va='top')
    ax[0, j].set_title('%s\n%s' % (name, sub), fontsize=10.5, color=col, fontweight='bold', loc='left')
    summary[k] = dict(best_step=int(b.step), best_val=float(b.dice_mean), n_val=len(v),
                      sim_first=float(L.sim.iloc[:250].mean()), sim_last=float(L.sim.iloc[-250:].mean()),
                      reg_last=float(L.reg.iloc[-250:].mean()), steps=int(L.step.max()))
fig.suptitle('訓練過程（三個模型之損失定義不同，縱軸各自獨立；細線 = 每一步，粗線 = 250 步移動平均；★ = 以 val Dice 選定之 checkpoint）',
             fontsize=10, color=MUTED, x=0.075, ha='left', y=0.985)
save(fig, '1111_train_curves.png')
for k, s in summary.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in s.items()})
