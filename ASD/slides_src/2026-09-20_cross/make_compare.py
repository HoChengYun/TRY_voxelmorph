# -*- coding: utf-8 -*-
"""把 exp2 / exp4 / exp3 的圖排成三版對照，給簡報用。

輸出到 models/deck_charts/：
    curve_exp234.png      三顆的 val Dice 曲線 + 擠爆比例（從 dice_curve_val.csv 重畫）
    jacobian_exp234.png   三顆的 Jacobian 圖疊成一張（讀既有的 vis_T054 輸出）
    grid_exp234.png       三顆的形變網格疊成一張

⚠️ 後兩張是把 visualize 出來的 PNG 直接堆起來，不是重新計算。
   所以 vis_T054 那些圖要先存在（見手冊 §13 的視覺化速查）。
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
os.makedirs(OUT, exist_ok=True)

INK, MUTED, RULE, PAPER = '#141A1D', '#5F6A6B', '#D9D9D2', '#FAFAF8'
TEAL, RUST_L, RUST = '#0E7C7B', '#D9895A', '#A34F1B'

# (實驗, 最佳 epoch, 標籤, 顏色)
EXPS = [
    ('mix_exp2', '0240', '速度場版・平滑權重 2', TEAL),
    ('mix_exp4', '0230', '位移場版・平滑權重 2', RUST_L),
    ('mix_exp3', '0240', '位移場版・平滑權重 1', RUST),
]


def curve(exp):
    p = os.path.join(ROOT, 'models', exp, 'dice_curve_val.csv')
    with open(p, encoding='utf-8') as f:
        r = sorted([(int(x['epoch']), float(x['dice_mean']), float(x['jneg_pct']))
                    for x in csv.DictReader(f)])
    return np.array(r)


# ── 1. 三顆的訓練曲線 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                         gridspec_kw={'height_ratios': [1.35, 1]})
for exp, ep, lab, col in EXPS:
    r = curve(exp)
    axes[0].plot(r[:, 0], r[:, 1], color=col, lw=2, marker='o', ms=4, label=lab)
    axes[1].plot(r[:, 0], r[:, 2], color=col, lw=2, marker='o', ms=4, label=lab)
    b = r[np.argmax(r[:, 1])]
    axes[0].plot(b[0], b[1], '*', color=col, ms=18, markeredgecolor=INK, zorder=5)

axes[0].set_ylabel('Dice（驗證集 51 位）', fontsize=11)
axes[0].set_title('對得多準：星號是選中的那一輪', fontsize=12.5, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=11)
axes[0].set_ylim(0.67, 0.815)

axes[1].axhline(0.366, ls='--', color='#C0392B', lw=1.2)
axes[1].text(248, 0.375, '論文的同版本 0.366%', ha='right', color='#C0392B', fontsize=9.5)
axes[1].set_ylabel('擠爆的比例', fontsize=11)
axes[1].set_xlabel('訓練輪數', fontsize=11)
axes[1].set_title('有沒有擠爆', fontsize=12.5, fontweight='bold')
axes[1].set_ylim(-0.02, 0.45)

for ax in axes:
    ax.grid(alpha=.3)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'curve_exp234.png'), dpi=130)
plt.close(fig)
print('->', os.path.join(OUT, 'curve_exp234.png'))


# ── 2 & 3. 把既有的視覺化 PNG 堆成三版對照 ───────────────────────────
def stack(kind, subject, out_name, xcrop=(0.035, 0.295), ycrop=(0.145, 1.0)):
    """三顆橫著排，只取軸狀切面那一格（原圖是三個切面並排，太寬塞不進投影片）。"""
    fig = plt.figure(figsize=(13, 5.2), facecolor=PAPER)
    for k, (exp, ep, lab, col) in enumerate(EXPS):
        p = os.path.join(ROOT, 'models', exp, 'vis_' + subject, '%s_%s_%s.png' % (kind, subject, ep))
        if not os.path.exists(p):
            raise SystemExit('[X] 找不到 %s' % p)
        im = mpimg.imread(p)
        h, w = im.shape[:2]
        im = im[int(h * ycrop[0]):int(h * ycrop[1]), int(w * xcrop[0]):int(w * xcrop[1])]
        ax = fig.add_axes([0.02 + k * 0.327, 0.12, 0.31, 0.80])
        ax.imshow(im)
        ax.axis('off')
        ax.text(0.5, -0.045, lab, transform=ax.transAxes, ha='center', va='top',
                fontsize=14, fontweight='bold', color=col)
    fig.savefig(os.path.join(OUT, out_name), dpi=130, facecolor=PAPER)
    plt.close(fig)
    print('->', os.path.join(OUT, out_name))


stack('jacobian', 'T054', 'jacobian_exp234.png')
stack('grid', 'T054', 'grid_exp234.png')
