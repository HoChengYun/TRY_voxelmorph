# -*- coding: utf-8 -*-
"""紅筆第 7 項要用的三張圖（去頭骨乾不乾淨）。

輸出到 models/deck_charts/：
    ss_method.png    兩種算法各兩格：關鍵步驟 -> 結果（文字說明放在投影片上，圖裡不塞）
    ss_examples.png  上緣、顱底各挑一位沒去乾淨 vs 一位去得乾淨，紅色是殘留
    ss_effect.png    50 位散布圖：殘留越多，模型的進步幅度有沒有變小

⚠️ 受試者是從 deck_data.json 的 skullstrip.worst / .best 讀的，不在這裡手挑。
   殘留怎麼算的見 ASD/check_skullstrip.py；那裡的 how_it_works.png 是完整版（四格），
   這裡是投影片用的精簡版。
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'ASD'))
from check_skullstrip import measure_top, measure_base, find, VERY_FAR  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

OUT = os.path.join(ROOT, 'models', 'deck_charts')
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(HERE, 'deck_data.json'), encoding='utf-8'))
SS = D['skullstrip']
DATA = os.path.join(ROOT, 'data', 'mixed_preprocessed_v2')

INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, RUST = '#0E7C7B', '#A34F1B'


def take(a, ax, i):
    return [a[i], a[:, i], a[:, :, i]][ax].T


def bbox(img, pad=4):
    """把四周的黑邊切掉，投影片上才不會一堆空白。"""
    ys, xs = np.where(img > 0.02)
    if not len(ys):
        return slice(None), slice(None)
    return (slice(max(0, ys.min() - pad), ys.max() + pad),
            slice(max(0, xs.min() - pad), xs.max() + pad))


# ── 1. 算法示意（投影片精簡版）────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(10, 7.4), facecolor=PAPER)

p, _ = find(DATA, SS['top']['worst'][0]['s'])
d = np.load(p)
vol, seg = d['vol'], d['seg']
_, tissue, _, _ = measure_top(vol, seg)
D3, H3, W3 = vol.shape
i = H3 // 2
crop = int(W3 * 0.58)
g = take(vol, 1, i)[crop:]
brain = take(seg > 0, 1, i)[crop:]
z = np.arange(W3)[None, :]
ztop = np.where((seg > 0).any(axis=2), ((seg > 0) * z[None]).max(axis=2), -1)

rgb = np.dstack([g] * 3)
rgb[brain] = 0.65 * rgb[brain] + 0.35 * np.array([0.15, 0.45, 0.95])
ax[0][0].imshow(rgb, origin='lower')
ax[0][0].plot(np.arange(D3), ztop[:, i] - crop, color='#FFD400', lw=2.4)
ax[0][0].set_xlim(0, D3 - 1)
ax[0][0].set_ylim(0, W3 - crop - 1)
ax[0][0].set_title('黃線＝每一欄最高的腦組織', fontsize=14, fontweight='bold')

rgb = np.dstack([g] * 3)
rgb[take(tissue, 1, i)[crop:]] = [1.0, 0.15, 0.1]
ax[0][1].imshow(rgb, origin='lower')
ax[0][1].set_title('紅＝黃線以上還亮著的組織', fontsize=14, fontweight='bold')

p, _ = find(DATA, 'sub-0038')
d = np.load(p)
vol, seg = d['vol'], d['seg']
dist = ndi.distance_transform_edt(seg == 0)
very = (vol > 0.02) & (dist > VERY_FAR)
lab, n = ndi.label(very)
sz = np.bincount(lab.ravel())
sz[0] = 0
big = lab == sz.argmax()
i = vol.shape[0] // 2
g = take(vol, 0, i)
sy, sx = bbox(g)
im = ax[1][0].imshow(np.clip(take(dist, 0, i), 0, 20)[sy, sx], cmap='viridis', origin='lower')
ax[1][0].set_title('顏色＝這個點離腦多遠（mm）', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax[1][0], fraction=0.046)
rgb = np.dstack([g] * 3)
rgb[take(very, 0, i)] = [1.0, 0.15, 0.1]
rgb[take(big, 0, i)] = [1.0, 0.85, 0.0]
ax[1][1].imshow(rgb[sy, sx], origin='lower')
ax[1][1].set_title('黃＝離腦 10mm 以外，最大的一坨', fontsize=14, fontweight='bold')

for r, lab_ in [(0, '上緣'), (1, '顱底')]:
    ax[r][0].text(-0.06, 0.5, lab_, transform=ax[r][0].transAxes, rotation=90,
                  va='center', ha='center', fontsize=17, fontweight='bold', color=INK)
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
fig.subplots_adjust(hspace=0.16, wspace=0.05, left=0.055, right=0.98, top=0.95, bottom=0.02)
fig.savefig(os.path.join(OUT, 'ss_method.png'), dpi=125, facecolor=PAPER)
plt.close(fig)
print('->', os.path.join(OUT, 'ss_method.png'))


# ── 2. 四個例子 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(15.5, 7.9), facecolor=PAPER)
plan = [
    ('top', 0, SS['top']['worst'][0], '沒去乾淨', RUST),
    ('top', 1, SS['top']['best'][-1], '去得乾淨', TEAL),
    ('base', 0, SS['base']['worst'][0], '沒去乾淨', RUST),
    ('base', 1, SS['base']['best'][-1], '去得乾淨', TEAL),
]
for metric, col, who, tag, color in plan:
    r = 0 if metric == 'top' else 1
    a = ax[r][col]
    p, _ = find(DATA, who['s'])
    d = np.load(p)
    vol, seg = d['vol'], d['seg']
    if metric == 'top':
        _, mark, _, _ = measure_top(vol, seg)
        sl, idx = 1, vol.shape[1] // 2                 # 冠狀
        g = take(vol, sl, idx)[int(vol.shape[2] * 0.56):]
        mk = take(mark, sl, idx)[int(vol.shape[2] * 0.56):]
        unit = '%.2f mm' % who['v']
    else:
        _, mark = measure_base(vol, seg)
        sl, idx = 0, vol.shape[0] // 2                 # 矢狀
        g, mk = take(vol, sl, idx), take(mark, sl, idx)
        sy, sx = bbox(g)
        g, mk = g[sy, sx], mk[sy, sx]
        unit = '%d 顆' % int(who['v'])
    rgb = np.dstack([g] * 3)
    rgb[mk] = [1.0, 0.15, 0.1]
    a.imshow(rgb, origin='lower')
    a.axis('off')
    a.set_title('%s　%s　殘留 %s' % (tag, who['s'], unit),
                fontsize=16, fontweight='bold', color=color, pad=8)
    a.text(0.5, -0.045, 'Dice %.3f → %.3f' % (who['before'], who['after']),
           transform=a.transAxes, ha='center', va='top',
           fontsize=18, fontweight='bold', color=color)
ax[0][0].text(-0.03, 0.5, '上緣', transform=ax[0][0].transAxes, rotation=90,
              va='center', ha='center', fontsize=19, fontweight='bold', color=INK)
ax[1][0].text(-0.03, 0.5, '顱底', transform=ax[1][0].transAxes, rotation=90,
              va='center', ha='center', fontsize=19, fontweight='bold', color=INK)
fig.subplots_adjust(hspace=0.40, wspace=0.02, left=0.04, right=0.99, top=0.945, bottom=0.055)
fig.savefig(os.path.join(OUT, 'ss_examples.png'), dpi=120, facecolor=PAPER)
plt.close(fig)
print('->', os.path.join(OUT, 'ss_examples.png'))


# ── 3. 殘留 vs 模型進步幅度 ─────────────────────────────────────────
ss = {r['subject']: r for r in
      csv.DictReader(open(os.path.join(ROOT, 'models', 'skullstrip_check',
                                       'skullstrip_all520.csv'), encoding='utf-8'))
      if r['split'] == 'test' and r['subject'] not in SS['excluded']}


def dice_csv(p):
    with open(os.path.join(ROOT, p), encoding='utf-8') as f:
        return {r['file'][:-4]: float(r['dice_mean']) for r in csv.DictReader(f)}


after = dice_csv('models/mix_exp3/dice_%s.csv' % D['models']['mix_exp3']['epoch'])
before = dice_csv('models/mix_exp2/dice_baseline.csv')
K = sorted(ss)
gain = np.array([after[k] - before[k] for k in K])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.0), facecolor=PAPER)
for a, (key, g, name, unit) in zip(axes, [
        ('top_vertex_mm', SS['top'], '上緣：皮質上方留了幾 mm', 'mm'),
        ('base_blob10', SS['base'], '顱底：最大一坨有幾顆體素', '顆')]):
    x = np.array([float(ss[k][key]) for k in K])
    a.scatter(x, gain, s=46, color=TEAL, alpha=.72, edgecolor='white', linewidth=.8, zorder=3)
    z = np.polyfit(x, gain, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    a.plot(xs, np.polyval(z, xs), color=RUST, lw=2.2, ls='--', zorder=4)
    a.set_title(name, fontsize=14, fontweight='bold')
    a.set_xlabel('殘留（%s）' % unit, fontsize=12)
    a.set_ylabel('模型讓 Dice 進步多少', fontsize=12)
    a.grid(alpha=.3)
    a.set_axisbelow(True)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    a.text(.97, .06, '相關 %+.2f' % g['r_gain'], transform=a.transAxes,
           ha='right', fontsize=15, fontweight='bold', color=RUST)
fig.suptitle('殘留越多，模型就拉不動嗎？　—　測試集 %d 位（每個點是一位受試者）' % SS['n'],
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(os.path.join(OUT, 'ss_effect.png'), dpi=130, facecolor=PAPER)
plt.close(fig)
print('->', os.path.join(OUT, 'ss_effect.png'))
