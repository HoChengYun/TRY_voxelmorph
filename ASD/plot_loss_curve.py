# -*- coding: utf-8 -*-
"""把訓練 log 畫成 loss 曲線（每個 epoch 取 100 步的平均）。

train.py 每一步印一行：
    epoch: 0250  step: 100/100   time: 1.48 sec  loss: -0.211664  (-0.235144, 0.023480)
                                                        總 loss    影像項（NCC）  平滑項
總 loss = 影像項 + 平滑項（平滑項已經乘過 λ 和 loss_mult）。

畫三格：總 loss、影像項、平滑項。給多份 log 就畫在同一張圖上比較。

⚠️ 平滑項跨實驗不能直接比大小：train.py 是 Grad('l2', loss_mult=int_downsize)，
   --int-downsize 2 的實驗平滑項會被乘 2，--int-downsize 1 的不會。

⚠️ 這是「訓練集」上的 loss，跟 val / test 的 Dice 不是同一件事。
   loss 一直降不代表 Dice 一直升；挑 epoch 還是看 dice_curve_val.csv。

用法
    python ASD\\plot_loss_curve.py --logs log\\mix_exp2.txt
    python ASD\\plot_loss_curve.py --logs log\\mix_exp2.txt log\\mix_exp3.txt --out models\\mix_exp3\\loss_exp2_vs_exp3.png
"""
import os
import re
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--logs', nargs='+', required=True, help='訓練 stdout，例如 log\\mix_exp2.txt')
ap.add_argument('--labels', nargs='*', default=None, help='圖例名稱，預設用檔名')
ap.add_argument('--out', default=None,
                help='預設：只給一份 log 時存到 models\\<實驗名>\\loss_curve.png')
args = ap.parse_args()

LINE = re.compile(r'epoch:\s*(\d+)\s+step:\s*(\d+)/(\d+).*?loss:\s*(-?[\d.eE+-]+)\s+'
                  r'\((-?[\d.eE+-]+),\s*(-?[\d.eE+-]+)\)')


def read_text(path):
    # log/ 混用兩種編碼（CLAUDE.md「編碼陷阱」）：早期是 UTF-16LE，後來是 UTF-8
    raw = open(path, 'rb').read()
    for enc in ('utf-16', 'utf-8', 'cp950'):
        try:
            t = raw.decode(enc)
            if '�' not in t and 'epoch' in t:
                return t
        except Exception:
            pass
    return raw.decode('utf-8', errors='replace')


def per_epoch(path):
    acc = {}
    for m in LINE.finditer(read_text(path)):
        ep = int(m.group(1))
        acc.setdefault(ep, []).append([float(m.group(4)), float(m.group(5)), float(m.group(6))])
    if not acc:
        sys.exit('[X] %s 裡找不到 loss 紀錄' % path)
    eps = sorted(acc)
    mean = np.array([np.mean(acc[e], axis=0) for e in eps])
    return np.array(eps), mean, {e: len(acc[e]) for e in eps}


labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.logs]
if len(labels) != len(args.logs):
    sys.exit('[X] --labels 數量要跟 --logs 一樣')

colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
titles = ['Total loss (image + smoothness)', 'Image term (NCC, lower is better)',
          'Smoothness term (already x lambda x loss_mult)']

for k, (path, lab) in enumerate(zip(args.logs, labels)):
    eps, mean, counts = per_epoch(path)
    short = [e for e, c in counts.items() if c < max(counts.values())]
    c = colors[k % len(colors)]
    for i, ax in enumerate(axes):
        ax.plot(eps, mean[:, i], color=c, lw=1.6, label=lab)
    print('%-12s %d 個 epoch；最後一個 epoch 平均：總 %.4f、影像 %.4f、平滑 %.4f（平滑佔 %.1f%%）'
          % (lab, len(eps), mean[-1, 0], mean[-1, 1], mean[-1, 2],
             100 * abs(mean[-1, 2]) / (abs(mean[-1, 1]) + abs(mean[-1, 2]))))
    if short:
        print('    ⚠️ 這些 epoch 步數不足（log 可能被截斷）：%s' % short[:10])

for ax, t in zip(axes, titles):
    ax.set_title(t, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[-1].set_xlabel('epoch')
fig.suptitle('Training loss per epoch (mean of steps)', fontweight='bold')
fig.tight_layout()

out = args.out
if out is None:
    if len(args.logs) == 1:
        out = os.path.join(ROOT, 'models', labels[0], 'loss_curve.png')
    else:
        sys.exit('[X] 多份 log 要用 --out 指定輸出位置')
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
fig.savefig(out, dpi=130)
print('  -> %s' % out)
