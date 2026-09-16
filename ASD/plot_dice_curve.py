# -*- coding: utf-8 -*-
"""把 dice_curve.csv 畫成兩格圖：Dice vs epoch、折疊率 vs epoch。

原本 models/asd_exp1/dice_curve_analysis.png 是某次臨時腳本畫的，沒留下來，
所以後面的實驗都沒有這張圖。這支把它固定下來。

輸入（都在 --model-dir 底下，由 test_dice.py 產生）
    dice_curve.csv        必要。欄位 epoch, dice_mean, jneg_pct
    dice_curve_val.csv    有的話一起畫（val 挑 epoch 的實驗才有）
    dice_baseline.csv     有的話拿來畫「只有 affine」的起點
    dice_baseline_val.csv 同上，val 的起點

輸出
    <model-dir>/dice_curve_analysis.png

⚠️ 舊圖上那條「±2 SEM」藍帶已經拿掉。那條用的是基準線的標準誤，
   不能拿來說「帶子裡的 epoch 分不出高下」—— 各 epoch 評的是同一批受試者，
   要比就得用逐人配對差值，而 dice_curve.csv 只存了平均值，算不出來。
   現在畫的是「前 10 名 epoch 的範圍」，那是描述事實，不是統計檢定。

用法
    python ASD\\plot_dice_curve.py --model-dir models\\mix_exp1
    python ASD\\plot_dice_curve.py --model-dir models\\asd_exp1 --baseline 0.6760
"""
import os
import sys
import csv
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 論文 Table I（Balakrishnan et al., IEEE TMI 2019）的折疊率
PAPER_VXM = 0.366
PAPER_SYN = 0.185

ap = argparse.ArgumentParser()
ap.add_argument('--model-dir', required=True, help='例如 models\\mix_exp1')
ap.add_argument('--out', default=None, help='預設 <model-dir>\\dice_curve_analysis.png')
ap.add_argument('--title', default=None, help='預設用資料夾名')
ap.add_argument('--baseline', type=float, default=None,
                help='只有 affine 的起點；預設從 dice_baseline.csv 算')
ap.add_argument('--label', default=None, help='圖例上的說明，例如 "ncc, λ=1.0"')
args = ap.parse_args()

MD = os.path.abspath(args.model_dir)
name = os.path.basename(MD)


def read_curve(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['epoch']), float(r['dice_mean']), float(r['jneg_pct'])))
    rows.sort()
    return rows


def read_baseline(path):
    if not os.path.exists(path):
        return None
    vals = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            vals.append(float(r['dice_mean']))
    return sum(vals) / len(vals) if vals else None


test = read_curve(os.path.join(MD, 'dice_curve.csv'))
val = read_curve(os.path.join(MD, 'dice_curve_val.csv'))
if not test and not val:
    sys.exit('[X] %s 底下沒有 dice_curve.csv（也沒有 dice_curve_val.csv）' % MD)

base = args.baseline
if base is None:
    base = read_baseline(os.path.join(MD, 'dice_baseline.csv'))
base_val = read_baseline(os.path.join(MD, 'dice_baseline_val.csv'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# ── 上：Dice ─────────────────────────────────────────────────────────
series = []
if test:
    series.append(('test', test, '#1f77b4', 'o', '-'))
if val:
    series.append(('val', val, '#2ca02c', 's', '--'))

for tag, rows, color, marker, ls in series:
    ep = [r[0] for r in rows]
    dc = [r[1] for r in rows]
    lbl = '%s (%s)' % (name, args.label) if args.label else name
    ax1.plot(ep, dc, ls, color=color, marker=marker, ms=4, lw=1.8,
             label='%s — %s' % (lbl, tag))
    best = max(rows, key=lambda r: r[1])
    ax1.plot(best[0], best[1], '*', color='#ff7f0e', ms=20,
             markeredgecolor='#8a4500', zorder=5,
             label='best on %s: epoch %d = %.4f' % (tag, best[0], best[1]))
    # 前 10 名 epoch 的範圍（描述用，不是統計檢定）
    top = sorted((r[1] for r in rows), reverse=True)[:10]
    if len(top) >= 2:
        ax1.axhspan(top[-1], top[0], color=color, alpha=0.08,
                    label='top-10 epochs on %s: %.4f–%.4f' % (tag, top[-1], top[0]))

if base is not None:
    ax1.axhline(base, ls='--', color='#d62728', lw=1.4,
                label='affine only (test) %.4f' % base)
if base_val is not None:
    ax1.axhline(base_val, ls=':', color='#2ca02c', lw=1.4,
                label='affine only (val) %.4f' % base_val)

ax1.set_ylabel('Dice (30 structures)')
ax1.set_title('%s — Dice vs epoch' % name, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend(loc='lower right', fontsize=8)

# ── 下：折疊率 ───────────────────────────────────────────────────────
allzero = True
for tag, rows, color, marker, ls in series:
    ep = [r[0] for r in rows]
    jn = [r[2] for r in rows]
    if max(jn) > 0:
        allzero = False
    ax2.plot(ep, jn, ls, color=color, marker=marker, ms=4, lw=1.8,
             label='%s — %s%s' % (name, tag, '  (all zero)' if max(jn) == 0 else ''))

ax2.axhline(PAPER_VXM, ls='--', color='#d62728', lw=1.4,
            label='paper VoxelMorph(CC) %.3f%%' % PAPER_VXM)
ax2.axhline(PAPER_SYN, ls=':', color='#9467bd', lw=1.4,
            label='paper ANTs SyN %.3f%%' % PAPER_SYN)
ax2.set_xlabel('epoch')
ax2.set_ylabel('% |J| <= 0  (folding)')
ax2.set_title('Folding budget completely unused' if allzero else 'Folding rate vs epoch',
              fontweight='bold')
ax2.grid(alpha=0.3)
ax2.legend(loc='upper right', fontsize=8)
if allzero:
    ax2.set_ylim(-0.02, max(PAPER_VXM * 1.25, 0.05))

fig.tight_layout()
out = args.out or os.path.join(MD, 'dice_curve_analysis.png')
fig.savefig(out, dpi=130)
print('  -> %s' % out)
for tag, rows, _, _, _ in series:
    best = max(rows, key=lambda r: r[1])
    print('  %-5s 最佳 epoch %d = %.4f；折疊率 %.4f%%（最大 %.4f%%）'
          % (tag, best[0], best[1], best[2], max(r[2] for r in rows)))
