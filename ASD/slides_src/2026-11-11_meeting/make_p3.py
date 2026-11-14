# -*- coding: utf-8 -*-
"""Part 3（11-11 使用者：「告訴老師我預計要怎麼一步一步從 VoxelMorph 改到最後 VXM 線架構版」）：
速度場的 2 × 2（手冊 §31.14）＋合併版變低之查核（消融第 1 顆 rep2）。
→ models\\deck_charts\\1111_p3.json、1111_p3_route.png、1111_p3_combo.png、1111_p3_box.png（各項指標）、1111_p3_struct.png（各結構 Dice 熱圖）

數字全部從 CSV、log、.pt 算，不手打：
- test Dice、folding：dice_<epoch>.csv（folding = jneg_fg_pct × MNI152 腦內 voxel 數）
- 參數量：arch.load_model() 讀 .pt 後 sum(p.numel())（p15 紅字「用程式數，不用估的」）
- val 第 150～250 輪平均：dice_curve_val.csv
- 訓練影像項：log\\<實驗名>.txt 每一輪 100 步的平均（括號裡第一個數；--dwp 時 ms: 另計，括號裡仍只有最後一層）
"""
import csv
import json
import os
import re
import statistics as st
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import wilcoxon

from style import J, OUT, INK, MUTED, PAPER, RULE, W, FG_VOX, save, clean, heat_t, heat_cell

sys.path.insert(0, J('ASD'))
sys.path.insert(0, J('voxelmorph-code'))
os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
os.environ.setdefault('VXM_BACKEND', 'pytorch')

# key：(models 子資料夾, 選定的 epoch)
RUNS = {
    'vxm': ('mix_exp3', '0240'),                         # ⓪ VoxelMorph 原論文設定（位移場）
    'svf': ('mix_exp6', '0190'),                         # ① 改為 SVF
    'casc': ('mix_cascade_rcn', '0195'),                 # ② + cascade（RCN）
    'pyr': ('mix_pyramid_dwp', '0205'),                  # ② + coarse-to-fine（DWP）
    'comb': ('mix_cascade_pyramid_rcn_dwp', '0155'),     # ③ 合併（加原作細節）
    'comb_old': ('mix_cascade_pyramid', '0140'),         # 合併（未加原作細節）
    'rep2': ('mix_cascade_pyramid_rep2', '0250'),        # 同上，重新訓練（消融第 1 顆）
    'rcn_only': ('mix_cascade_pyramid_rcn', '0200'),     # 合併、只開 --rcn（消融第 2 顆，11-11 帶回）
    'casc_old': ('mix_cascade', '0170'),                 # cascade、未加 RCN 細節
    'pyr_old': ('mix_pyramid', '0245'),                  # C2F、未加 DWP 細節
}


def read_csv(p):
    return list(csv.DictReader(open(p, encoding='utf-8')))


def test_rows(k):
    n, ep = RUNS[k]
    return {r['file']: r for r in read_csv(J('models', n, 'dice_%s.csv' % ep))}


def n_params(k):
    import torch
    from arch import load_model
    n, ep = RUNS[k]
    m = load_model(J('models', n, '%s.pt' % ep), 'cpu')
    return int(sum(p.numel() for p in m.parameters()))


def read_log(name):
    raw = open(J('log', name + '.txt'), 'rb').read()
    for enc in ('utf-8', 'utf-16', 'cp950'):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if '�' not in t and 'epoch' in t:
            return t
    raise RuntimeError('讀不出 ' + name)


PAT = re.compile(r'epoch:\s*(\d+)\s+step:.*?loss:\s*-?[\d.]+\s*\((-?[\d.]+),')


def train_img(k):
    ep = {}
    for m in PAT.finditer(read_log(RUNS[k][0])):
        ep.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    assert all(len(v) == 100 for v in ep.values()), k
    return {e: float(np.mean(v)) for e, v in sorted(ep.items())}


T = {k: test_rows(k) for k in RUNS}
files = sorted(T['vxm'])
assert all(sorted(T[k]) == files for k in RUNS), '各模型的 test 受試者不一致'
D = {k: np.array([float(T[k][f]['dice_mean']) for f in files]) for k in RUNS}

out = {}
for k, (n, ep) in RUNS.items():
    src = T[k]
    if 'jneg_fg_pct' not in src[files[0]]:          # mix_exp3、mix_exp6 的 dice CSV 是加這一欄之前跑的，改讀同一個 epoch 的 surface CSV
        src = {r['file']: r for r in read_csv(J('models', n, 'surface_%s.csv' % ep))}
        assert all(abs(float(src[f]['dice_mean']) - float(T[k][f]['dice_mean'])) < 5e-4 for f in files), k   # 兩次推論在 GPU 上差 ≤ 1.4e-4
    fold = np.array([float(src[f]['jneg_fg_pct']) for f in files]) / 100 * FG_VOX
    o = dict(name=n, epoch=int(ep), dice=float(D[k].mean()), fold_vox=float(fold.mean()), params=n_params(k))
    sp = J('models', n, 'surface_%s.csv' % ep)
    if os.path.exists(sp):                           # HD95、SDlogJ（test_dice.py --surface）；rep2 沒跑 --surface
        sr = {r['file']: r for r in read_csv(sp)}
        o.update(hd95=float(np.mean([float(sr[f]['hd95_mean']) for f in files])), sdlogj=float(np.mean([float(sr[f]['sdlogj']) for f in files])))
    vp = J('models', n, 'dice_curve_val.csv')
    if k not in ('vxm', 'svf'):                     # cascade／C2F 系列都比 val 後段平均（都是每 5 輪一點）
        vc = {int(r['epoch']): float(r['dice_mean']) for r in read_csv(vp)}
        late = [v for e, v in vc.items() if e >= 150]
        assert len(late) == 21, (k, len(late))
        o.update(val_late=float(np.mean(late)), val_best=max(vc.values()), val_curve=vc)
    out[k] = o


def paired(a, b):
    return dict(n=int((D[a] > D[b]).sum()), p=float(wilcoxon(D[a], D[b]).pvalue))


out['pairs'] = {'casc_svf': paired('casc', 'svf'), 'pyr_svf': paired('pyr', 'svf'), 'svf_vxm': paired('svf', 'vxm'), 'casc_pyr': paired('casc', 'pyr'),
                'casc_vxm': paired('casc', 'vxm'), 'pyr_vxm': paired('pyr', 'vxm'), 'comb_vxm': paired('comb', 'vxm'),
                'casc_comb': paired('casc', 'comb'), 'pyr_comb': paired('pyr', 'comb'),
                'rep2_old': paired('rep2', 'comb_old'), 'rep2_comb': paired('rep2', 'comb'), 'old_comb': paired('comb_old', 'comb'),
                'rcnonly_rep2': paired('rcn_only', 'rep2'), 'rcnonly_comb': paired('rcn_only', 'comb'),
                'combold_cascold': paired('comb_old', 'casc_old'), 'combold_pyrold': paired('comb_old', 'pyr_old')}
TR = {k: train_img(k) for k in ('comb_old', 'rep2', 'comb')}
out['train_img'] = {k: {str(e): v for e, v in TR[k].items()} for k in TR}
for k in TR:
    out[k]['train_img_ep1_50'] = float(np.mean([v for e, v in TR[k].items() if e <= 50]))
json.dump(out, open(os.path.join(OUT, '1111_p3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[OK]', os.path.join(OUT, '1111_p3.json'))
for k in RUNS:
    o = out[k]
    print('%-9s Dice %.4f  folding %8.1f  params %9d  val_late %s' % (k, o['dice'], o['fold_vox'], o['params'], '%.4f' % o['val_late'] if 'val_late' in o else '-'))
print(out['pairs'])

# ════ 路線圖：⓪ VoxelMorph → ① SVF → ② cascade／coarse-to-fine → ③ 合併 ════
H = 4.0
fig = plt.figure(figsize=(W, H), facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
GREEN, AMBER = '#3E9B3A', '#D08C2B'
COL = {'vxm': '#4B5D73', 'svf': '#D08C2B', 'casc': '#2F7FD0', 'pyr': '#0E7C7B', 'comb': '#7A4FA0'}   # 同 make_p3_deform.py


def box(x, y, w, h, title, lines, col, status, scol):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc='white', ec=col, lw=1.8))
    ax.text(x + 0.13, y + h - 0.14, title, ha='left', va='top', fontsize=12, fontweight='bold', color=col)
    ax.text(x + 0.13, y + h - 0.5, '\n'.join(lines), ha='left', va='top', fontsize=9.6, color=INK, linespacing=1.45)
    ax.text(x + w - 0.1, y + 0.1, status, ha='right', va='bottom', fontsize=9.3, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc=scol, ec='none'))


def stat(k):
    o = out[k]
    f = o['fold_vox']
    fs = '%s' % format(round(f), ',') if f >= 10 else '%.1f' % f
    return 'Dice %.4f、%.2f M 參數' % (o['dice'], o['params'] / 1e6), 'folding %s voxel／位' % fs


BH = 1.62
box(0.1, 1.2, 2.3, BH, '① VoxelMorph', ['原論文設定（mix_exp3）', 'displacement field', *stat('vxm')], COL['vxm'], '完成', GREEN)
box(3.15, 1.2, 2.3, BH, '② SVF', ['僅改為 SVF（mix_exp6）', 'scaling and squaring', *stat('svf')], COL['svf'], '完成', GREEN)
box(6.5, 2.3, 2.5, BH, '③ ＋ Cascade（RCN）', ['串接 2 個 U-Net', 'mix_cascade_rcn', *stat('casc')], COL['casc'], '完成', GREEN)
box(6.5, 0.1, 2.5, BH, '③ ＋ C2F（DWP）', ['雙流特徵金字塔', 'mix_pyramid_dwp', *stat('pyr')], COL['pyr'], '完成', GREEN)
box(9.58, 1.2, 2.47, BH, '④ Cascade × C2F', ['每一級改為雙流金字塔', 'mix_cascade_pyramid_rcn_dwp', *stat('comb')], COL['comb'], '下降原因檢查中', AMBER)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14, color=MUTED, lw=1.3))


def label(x, y, t, va):
    ax.text(x, y, t, ha='center', va=va, fontsize=9, color=MUTED)


arrow(2.43, 2.01, 3.12, 2.01); label(2.78, 2.1, '改為 SVF', 'bottom')
arrow(5.48, 2.35, 6.47, 3.0); label(5.98, 2.86, '加 cascade', 'bottom')
arrow(5.48, 1.67, 6.47, 1.0); label(5.98, 1.18, '加 C2F', 'top')
arrow(9.03, 3.0, 9.55, 2.5)
arrow(9.03, 1.0, 9.55, 1.5)
label(9.3, 2.0, '合併', 'center')
save(fig, '1111_p3_route.png')

# ════ 合併版：val Dice 與訓練影像項（未加原作細節 × 2 次、加原作細節 × 1 次）════
SER = [('comb_old', '未加原作細節（第 1 次）', '#B5487A', '-'),
       ('rep2', '未加原作細節（重新訓練）', '#B5487A', (0, (4, 2))),
       ('comb', '加原作細節（RCN＋DWP）', COL['comb'], '-')]
fig, axs = plt.subplots(1, 2, figsize=(W, 3.7))
fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.14, wspace=0.2)
ax = axs[0]
for k, lab, c, ls in SER:
    vc = out[k]['val_curve']
    es = sorted(e for e in vc if e >= 50)
    ax.plot(es, [vc[e] for e in es], color=c, ls=ls, lw=2, label=lab)
ax.axvspan(150, 250, color=RULE, alpha=0.35, lw=0, zorder=0)
ax.text(200, 0.8194, '第 150～250 輪（取平均）', ha='center', va='bottom', fontsize=9, color=MUTED)
ax.set_ylim(0.800, 0.820)
ax.set_xlabel('epoch', fontsize=10, color=MUTED)
ax.set_title('(a) val Dice（51 位；每 5 輪評估一次）', fontsize=10.5, loc='left', color=INK)
clean(ax)
ax = axs[1]
for k, lab, c, ls in SER:
    tr = TR[k]
    es = sorted(tr)
    ax.plot(es, [-tr[e] for e in es], color=c, ls=ls, lw=2, label=lab)
ax.set_ylim(0.20, 0.28)
ax.set_xlabel('epoch', fontsize=10, color=MUTED)
ax.set_title('(b) 訓練之影像相似度（local NCC，越高越好；每輪 100 步平均）', fontsize=10.5, loc='left', color=INK)
clean(ax)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
save(fig, '1111_p3_combo.png')

# ════ 結果：每位受試者之 Dice（箱形圖）＋右側各項指標 ════
ROWS = [('vxm', '① VoxelMorph'), ('svf', '② SVF'), ('casc', '③ ＋Cascade'), ('pyr', '③ ＋C2F'), ('comb', '④ Cascade × C2F'), ('affine', 'Affine（起點）')]
GRAYC = '#9A9A93'
aff_full = {r['file']: r for r in read_csv(J('models', 'mix_exp3', 'dice_baseline.csv'))}
DA = np.array([float(aff_full[f]['dice_mean']) for f in files])
out['affine'] = dict(dice=float(DA.mean()))
fig = plt.figure(figsize=(W, 4.1))
ax = fig.add_axes([0.15, 0.13, 0.47, 0.79])
rng = np.random.RandomState(0)
for i, (k, lab) in enumerate(ROWS[::-1]):
    v = DA if k == 'affine' else D[k]
    c = GRAYC if k == 'affine' else COL[k]
    ax.boxplot(v, positions=[i], vert=False, widths=0.52, patch_artist=True, showfliers=False,
               medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=c, lw=1.1), capprops=dict(color=c, lw=1.1),
               boxprops=dict(facecolor=c + '33', edgecolor=c, lw=1.2), zorder=2)
    ax.scatter(v, i + rng.uniform(-0.17, 0.17, len(v)), s=9, color=c, alpha=0.75, edgecolor='none', zorder=3)
ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS[::-1]], fontsize=10.5)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlim(0.62, 0.86)
ax.set_xlabel('Dice（test 51 位；論文之 30 個結構平均；每點一位）', fontsize=10, color=MUTED)
clean(ax, grid='x')
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)


def fmt_f(o):
    return format(round(o['fold_vox']), ',') if o['fold_vox'] >= 10 else '%.1f' % o['fold_vox']


# 每欄：(x, 欄名, 格式, 取值, 較佳方向, 是否取 log)；底色只比五個配準模型，參數量不上色
CC = [(0.665, 'Dice', lambda o: '%.4f' % o['dice'], 'dice', 'high', False), (0.74, 'HD95\n（mm）', lambda o: '%.2f' % o['hd95'], 'hd95', 'low', False),
      (0.815, 'Folding\n（voxel／位）', fmt_f, 'fold_vox', 'low', True), (0.885, 'SDlogJ', lambda o: '%.2f' % o['sdlogj'], 'sdlogj', 'low', False),
      (0.955, '參數量', lambda o: '%.2f M' % (o['params'] / 1e6), 'params', None, False)]
REG = [k for k, _ in ROWS if k != 'affine']
for x, head, fn, key, better, lg in CC:
    fig.text(x, 0.975, head, ha='center', va='top', fontsize=9.5, color=MUTED, fontweight='bold', linespacing=1.15)
    tt = dict(zip(REG, heat_t([out[k][key] for k in REG], better, lg))) if better else {}
    for i, (k, lab) in enumerate(ROWS[::-1]):
        yy = fig.transFigure.inverted().transform(ax.transData.transform((0, i)))[1]
        t = fn(out[k]) if (k != 'affine' or head == 'Dice') else '—'
        if k in tt:
            heat_cell(fig, x, yy, t, tt[k], w=0.06, h=0.078)
        else:
            fig.text(x, yy, t, ha='center', va='center', fontsize=10.5, color=INK)
fig.text(0.81, 0.02, '底色：同一欄內比較，深 = 較佳；粗體 = 該欄最佳', ha='center', fontsize=8.5, color=MUTED)
save(fig, '1111_p3_box.png')

# ════ 各結構之 Dice（左右合併，51 位平均）════
STRUCT = [('大腦白質', (2, 41)), ('大腦皮質', (3, 42)), ('側腦室', (4, 43)), ('小腦白質', (7, 46)), ('小腦皮質', (8, 47)), ('視丘', (10, 49)),
          ('尾狀核', (11, 50)), ('殼核', (12, 51)), ('蒼白球', (13, 52)), ('海馬迴', (17, 53)), ('杏仁核', (18, 54)), ('腹側間腦', (28, 60)),
          ('脈絡叢', (31, 63)), ('第三腦室', (14,)), ('第四腦室', (15,)), ('腦幹', (16,)), ('腦脊髓液', (24,))]


def struct_mean(rows, labs):
    return float(np.mean([np.mean([float(rows[f]['label_%d' % l]) for l in labs]) for f in files]))


SD = {'affine': {nm: struct_mean(aff_full, labs) for nm, labs in STRUCT}}
for k in ('vxm', 'svf', 'casc', 'pyr', 'comb'):
    SD[k] = {nm: struct_mean(T[k], labs) for nm, labs in STRUCT}
out['struct'] = SD
order = sorted([nm for nm, _ in STRUCT], key=lambda nm: SD['svf'][nm])          # SVF 由低到高（上 = 最難）
# 熱圖：數字是 Dice（不放差值）；顏色只在每一列的五個配準模型之間比（該列最低 = 淺、最高 = 深），粗體 = 該列最高
KS = [('affine', 'Affine'), ('vxm', '①\nVoxelMorph'), ('svf', '②\nSVF'), ('casc', '③\n＋Cascade'), ('pyr', '③\n＋C2F'), ('comb', '④\nCascade×C2F')]
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list('row', ['#EEF3F8', '#2F5F9E'])
fig, ax = plt.subplots(figsize=(W * 0.6, 5.3))
fig.subplots_adjust(left=0.15, right=0.995, top=0.88, bottom=0.02)
ax.set_xlim(-0.5, len(KS) - 0.5)
ax.set_ylim(len(order) - 0.5, -0.5)
for i, nm in enumerate(order):
    reg = [SD[k][nm] for k, _ in KS[1:]]
    lo, hi = min(reg), max(reg)
    for j, (k, _) in enumerate(KS):
        v = SD[k][nm]
        if k == 'affine':
            fc, tc, fw = PAPER, MUTED, 'normal'
        else:
            t = (v - lo) / (hi - lo) if hi > lo else 0.5
            fc = CMAP(t)
            tc = 'white' if t > 0.6 else INK
            fw = 'bold' if v == hi else 'normal'
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc=fc, ec=PAPER, lw=1.5))
        ax.text(j, i, '%.3f' % v, ha='center', va='center', fontsize=9.3, color=tc, fontweight=fw)
ax.set_yticks(range(len(order)), order, fontsize=9.8)
ax.xaxis.tick_top()
ax.set_xticks(range(len(KS)), [lab for _, lab in KS], fontsize=9.3)
for t_, (k, _) in zip(ax.get_xticklabels(), KS):
    t_.set_color(GRAYC if k == 'affine' else COL[k])
    t_.set_fontweight('bold')
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
save(fig, '1111_p3_struct.png')
below = [nm for nm in order if SD['comb'][nm] < min(SD['casc'][nm], SD['pyr'][nm])]
out['comb_below_both'] = below
print('合併版低於單獨兩者的結構：', below)
json.dump(out, open(os.path.join(OUT, '1111_p3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[OK] 1111_p3.json 重寫（加 affine、struct）')
