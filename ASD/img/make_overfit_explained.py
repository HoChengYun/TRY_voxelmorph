# -*- coding: utf-8 -*-
"""手冊 §22 的教學圖：overfit 長什麼樣，以及 mix_exp3 屬於哪一種。

讀 log/mix_exp3.txt（訓練 loss）與 models/mix_exp3/dice_curve_val.csv（驗證 Dice），
輸出到同資料夾的 overfit_explained.png。
"""
import os, re, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # ASD/img -> ASD -> 專案根目錄
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, RUST = '#0E7C7B', '#A34F1B'


def read_log(p):
    raw = open(p, 'rb').read()
    t = None
    for enc in ('utf-16', 'utf-8', 'cp950'):
        try:
            cand = raw.decode(enc)
        except Exception:
            continue
        # ⚠️ 只看有沒有 U+FFFD 不夠 —— UTF-8 的 log 用 UTF-16 解會變成看似合法的亂碼。
        #    再要求解出來必須含 epoch 這個字（沿用 ASD/plot_loss_curve.py 的做法）。
        if chr(0xFFFD) not in cand and 'epoch' in cand:
            t = cand
            break
    if t is None:
        t = raw.decode('utf-8', errors='replace')
    per = {}
    for m in re.finditer(r'epoch:\s*(\d+)\s+step:\s*\d+/\d+.*?loss:\s*(-?[\d.eE+-]+)', t):
        per.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    e = sorted(per)
    return np.array(e), np.array([np.mean(per[k]) for k in e])


fig = plt.figure(figsize=(15, 5.6), facecolor=PAPER)

# ── 上排：三種情況的示意 ──────────────────────────────────────────
x = np.linspace(0, 100, 300)
titles = ['訓練不夠（underfit）', '剛剛好', '過擬合（overfit）']
for k in range(3):
    ax = fig.add_subplot(2, 3, k + 1)
    tr = 1.0 * np.exp(-x / [90, 35, 18][k]) + [0.35, 0.08, 0.02][k]
    if k == 0:
        va = tr + 0.05
    elif k == 1:
        va = tr + 0.07 + 0.0002 * x
    else:
        va = 0.9 * np.exp(-x / 14) + 0.12 + 0.0055 * np.clip(x - 25, 0, None)
    ax.plot(x, tr, color=TEAL, lw=2.4, label='訓練資料')
    ax.plot(x, va, color=RUST, lw=2.4, label='沒看過的資料')
    if k == 2:
        i = va.argmin()
        ax.axvline(x[i], color=INK, ls=':', lw=1.6)
        ax.annotate('從這裡開始壞掉', (x[i], va[i]), (x[i] + 8, va[i] + 0.35),
                    fontsize=11, arrowprops=dict(arrowstyle='->', color=INK))
    ax.set_title(titles[k], fontsize=13.5, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.25)
    ax.set_xlabel('訓練輪數 →', fontsize=10.5, color=MUTED)
    if k == 0:
        ax.set_ylabel('錯誤（越低越好）', fontsize=10.5, color=MUTED)
    ax.legend(fontsize=10, loc='upper right')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    note = ['兩條都還很高\n再train下去會更好', '訓練持續變好\n沒看過的也持平或微升',
            '訓練還在變好\n沒看過的卻往上翹'][k]
    ax.text(.03, .05, note, transform=ax.transAxes, fontsize=10.5, color=MUTED, va='bottom')

# ── 下排：mix_exp3 實際的曲線 ────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 4)
e, l = read_log(os.path.join(ROOT, 'log', 'mix_exp3.txt'))
ax1.plot(e, l, color=TEAL, lw=1.6)
ax1.set_title('mix_exp3：訓練 loss', fontsize=13, fontweight='bold', color=TEAL)
ax1.set_xlabel('訓練輪數', fontsize=10.5)
ax1.grid(alpha=.3)
ax1.set_axisbelow(True)
ax1.text(.5, .12, '一路往下 —— 這條不能拿來判斷 overfit', transform=ax1.transAxes,
         ha='center', fontsize=11, color=MUTED)
for sp in ('top', 'right'):
    ax1.spines[sp].set_visible(False)

ax2 = fig.add_subplot(2, 3, 5)
with open(os.path.join(ROOT, 'models', 'mix_exp3', 'dice_curve_val.csv'), encoding='utf-8') as f:
    r = sorted([(int(x['epoch']), float(x['dice_mean'])) for x in csv.DictReader(f)])
ve = np.array([a for a, _ in r])
vd = np.array([b for _, b in r])
ax2.plot(ve, vd, 'o-', color=RUST, lw=1.8, ms=4)
m = ve >= 150
ax2.plot(ve[m], np.polyval(np.polyfit(ve[m], vd[m], 1), ve[m]), color=INK, ls='--', lw=1.8)
ax2.set_title('mix_exp3：驗證集 Dice（沒看過的 51 位）', fontsize=13, fontweight='bold', color=RUST)
ax2.set_xlabel('訓練輪數', fontsize=10.5)
ax2.grid(alpha=.3)
ax2.set_axisbelow(True)
ax2.set_ylim(0.67, 0.815)
ax2.text(.5, .12, '沒有往下掉 → 到 250 輪為止沒有 overfit', transform=ax2.transAxes,
         ha='center', fontsize=11, color=MUTED)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)

ax3 = fig.add_subplot(2, 3, 6)
ax3.axis('off')
ax3.text(0, 1.0, '判讀順序', fontsize=14, fontweight='bold', va='top')
steps = [
    ('① 先看「沒看過的資料」那條', '訓練那條一直降是正常的，不是證據'),
    ('② 它有沒有轉頭往上（或 Dice 往下）', '有轉折 = overfit，轉折點就是該停的地方'),
    ('③ 跌幅有沒有大過抖動', 'mix_exp3 的抖動是 ±0.003，小於這個就是雜訊'),
    ('④ 交叉印證：形變有沒有變誇張', '折疊率一路 0.233% 沒變 = 沒在硬凹'),
]
y = 0.84
for a, b in steps:
    ax3.text(0, y, a, fontsize=12, fontweight='bold', va='top', color=INK)
    ax3.text(0.02, y - 0.10, b, fontsize=11, va='top', color=MUTED)
    y -= 0.24
fig.suptitle('overfit 怎麼看', fontsize=17, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, 'overfit_explained.png')
fig.savefig(out, dpi=125, facecolor=PAPER)
print('->', out)
