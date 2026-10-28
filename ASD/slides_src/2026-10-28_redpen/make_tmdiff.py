# -*- coding: utf-8 -*-
"""10/28 簡報：TransMorph-diff（官方訓練方式）一頁的圖與數字。2026-10-28。

使用者：「transmorph diff 加上」。TransMorph-diff（TransMorph 專案 mixv2/train_mixv2_diff.py，照官方：機率式 SVF、MSE/σ² ＋ KL、
Adamax、lr 遞減、三向翻轉；只有訓練量改成 2.5 萬步）在 AI 上練完，val 挑第 75 輪，test Dice 只有 0.714（affine 0.688）。
小腦反而比 affine 差 → 檢查 atlas 與受試者的亮度：受試者小腦比 atlas 暗很多，MSE 會硬把亮度對上（NCC 是局部正規化，不受影響）。

  1028_tmdiff.png        上：T054 經過左側小腦之矢狀面（atlas、affine、TransMorph、TransMorph-diff），線＝atlas 之小腦輪廓
                         下：各結構之 Dice（左右合併；affine、VoxelMorph（SVF）、TransMorph、TransMorph-diff）
  deck_data_tmdiff.json  投影片上的數字：各結構 Dice、atlas 與受試者（test 51 位）各結構之平均亮度

資料
  Dice      models\\mix_exp2\\dice_baseline.csv（affine）、mix_exp6\\dice_0190.csv、tm_mixv2\\dice_0205.csv、tm_diff_mixv2\\dice_0075.csv
            （tm_diff_mixv2 的 CSV 是 TransMorph 專案 runs\\tm_diff_mixv2\\ 複製過來的）
  形變後影像 models\\deck_charts\\1028_vis_cache_T054_tm_mixv2.npz、…_tm_diff_mixv2.npz（TransMorph 專案 mixv2\\export_vis.py）
"""
import os
import csv
import glob
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
J = lambda *p: os.path.join(ROOT, *p)
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
GRAY, TEAL, SLATE, RED = '#9A9A93', '#0E7C7B', '#4B5D73', '#B03A2E'
S, TD = 'T054', ('tm_diff_mixv2', '0075')

STRUCT = [('Cerebral-WM', (2, 41)), ('Cerebral-Cortex', (3, 42)), ('Lateral-Vent', (4, 43)), ('Cerebellum-WM', (7, 46)),
          ('Cerebellum-Cortex', (8, 47)), ('Thalamus', (10, 49)), ('Caudate', (11, 50)), ('Putamen', (12, 51)),
          ('Pallidum', (13, 52)), ('Hippocampus', (17, 53)), ('Amygdala', (18, 54)), ('VentralDC', (28, 60)),
          ('Brain-Stem', (16,)), ('3rd-Ventricle', (14,)), ('4th-Ventricle', (15,))]
METH = [('affine', 'Affine', GRAY, ('mix_exp2', 'dice_baseline.csv')), ('mix_exp6', 'VoxelMorph（SVF，NCC）', TEAL, ('mix_exp6', 'dice_0190.csv')),
        ('tm_mixv2', 'TransMorph（disp.，NCC）', SLATE, ('tm_mixv2', 'dice_0205.csv')),
        ('tm_diff', 'TransMorph-diff（SVF，MSE，官方）', RED, (TD[0], 'dice_%s.csv' % TD[1]))]


def rd(p):
    with open(p, encoding='utf-8') as fh:
        return {r['file']: r for r in csv.DictReader(fh)}


T = {'struct': {}, 'mean': {}}
tabs = {k: rd(J('models', *src)) for k, _, _, src in METH}
files = sorted(tabs['affine'])
for k in tabs:
    assert sorted(tabs[k]) == files, k
    T['mean'][k] = float(np.mean([float(tabs[k][f]['dice_mean']) for f in files]))
for n, ls in STRUCT:
    T['struct'][n] = {k: float(np.mean([np.nanmean([float(tabs[k][f]['label_%d' % l]) for l in ls]) for f in files])) for k in tabs}

# atlas 與受試者（test 51 位）各結構之平均亮度（影像值 0～1）
atlas = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol']
aseg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg']
INT = [('Cerebral-WM', (2, 41)), ('Cerebral-Cortex', (3, 42)), ('Cerebellum-WM', (7, 46)), ('Cerebellum-Cortex', (8, 47)),
       ('Brain-Stem', (16,)), ('Lateral-Vent', (4, 43))]
sub = {n: [] for n, _ in INT}
for f in sorted(glob.glob(J('data', 'mixed_preprocessed_v2', 'test', '*.npz'))):
    z = np.load(f)
    for n, ls in INT:
        sub[n].append(float(z['vol'][np.isin(z['seg'], ls)].mean()))
T['intensity'] = {n: {'atlas': float(atlas[np.isin(aseg, ls)].mean()), 'sub': float(np.mean(sub[n])), 'sub_sd': float(np.std(sub[n]))}
                  for n, ls in INT}
with open(os.path.join(HERE, 'deck_data_tmdiff.json'), 'w', encoding='utf-8') as fh:
    json.dump(T, fh, ensure_ascii=False, indent=1)

# ── 圖 ────────────────────────────────────────────────────────────────
vol = np.load(J('data', 'mixed_preprocessed_v2', 'test', S + '.npz'))['vol']
W_ = {'atlas': atlas, 'affine': vol,
      'tm_mixv2': np.load(J('models', 'deck_charts', '1028_vis_cache_%s_tm_mixv2.npz' % S))['warped'],
      'tm_diff': np.load(J('models', 'deck_charts', '1028_vis_cache_%s_tm_diff_mixv2.npz' % S))['warped']}
i0 = 96 - 20                                       # MNI x = −20 mm（左側小腦半球）；x = i − 96
cer = np.isin(aseg, (7, 8, 46, 47))
fg = atlas > 0
ys, zs = np.where(fg[i0])
box = (max(ys.min() - 4, 0), ys.max() + 5, max(zs.min() - 4, 0), zs.max() + 5)
sl = lambda a: a[i0, box[0]:box[1], box[2]:box[3]].T          # 列 = z（origin lower → 上＝上）、欄 = y（右＝前）
PAN = [('atlas', 'Atlas（fixed）'), ('affine', 'Affine'), ('tm_mixv2', 'TransMorph'), ('tm_diff', 'TransMorph-diff')]

FH, TOP = 5.55, 0.46                                   # 圖高、上排影像頂端離圖頂（吋；留給說明文字與子圖標題）
fig = plt.figure(figsize=(7.4, FH), facecolor=PAPER)
h_, w_ = sl(atlas).shape
pw = 7.4 / 4 - 0.06
ph = pw * h_ / w_
for j, (k, t) in enumerate(PAN):
    ax = fig.add_axes([(j * (pw + 0.06) + 0.02) / 7.4, 1 - (TOP + ph) / FH, pw / 7.4, ph / FH])
    ax.imshow(sl(W_[k]), cmap='gray', origin='lower', vmin=0, vmax=1)
    ax.contour(sl(cer).astype(float), levels=[0.5], colors='#F1C40F', linewidths=0.8)
    ax.set_title(t, fontsize=9.5, color=INK, pad=3)
    ax.axis('off')
ax = fig.add_axes([0.09, 0.15, 0.89, 1 - (TOP + ph + 0.55) / FH - 0.15])
names = [n for n, _ in STRUCT]
x = np.arange(len(names))
for j, (k, lab, c, _) in enumerate(METH):
    ax.plot(x + (j - 1.5) * 0.17, [T['struct'][n][k] for n in names], 'o', ms=4.5, color=c, label='%s %.3f' % (lab, T['mean'][k]))
for n in ('Cerebellum-WM', 'Cerebellum-Cortex'):
    ax.axvspan(names.index(n) - 0.45, names.index(n) + 0.45, color=RED, alpha=0.08, lw=0)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7.8)
ax.set_xlim(-0.6, len(names) - 0.4)
ax.set_ylabel('Dice（左右合併）', fontsize=9)
ax.tick_params(axis='y', labelsize=8)
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.0))
ax.grid(axis='y', alpha=.3)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.text(0.01, 1 - 0.12 / FH, '%s，矢狀面 x = −20 mm（左側小腦半球）；黃線＝atlas 之小腦' % S, fontsize=8.5, color=MUTED, va='center')
p = J('models', 'deck_charts', '1028_tmdiff.png')
fig.savefig(p, dpi=150, facecolor=PAPER)
print('->', p)
print({k: round(v, 4) for k, v in T['mean'].items()})
for n in ('Cerebellum-WM', 'Cerebellum-Cortex', 'Cerebral-Cortex', 'Pallidum'):
    print(n, {k: round(v, 3) for k, v in T['struct'][n].items()})
print({n: (round(v['atlas'], 3), round(v['sub'], 3)) for n, v in T['intensity'].items()})
