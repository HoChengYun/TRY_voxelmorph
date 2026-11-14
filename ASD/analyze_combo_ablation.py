# -*- coding: utf-8 -*-
"""合併版（cascade × coarse-to-fine）消融五顆之分析（2026-11-13；手冊 §31.14）。只推論、不訓練。

五顆（設定相同，只差開關；test 51 位）：
  old       mix_cascade_pyramid          不開（第 1 次）
  rep2      mix_cascade_pyramid_rep2     不開（重練）
  rcn       mix_cascade_pyramid_rcn      只開 --rcn（①②）
  dwp       mix_cascade_pyramid_dwp      只開 --dwp（③④）
  both      mix_cascade_pyramid_rcn_dwp  兩個都開
  ms1       mix_cascade_pyramid_dwp_ms1  開 --dwp，④ 改加在第 1 級（--dwp-ms-stage first；資料夾存在才加）

A. 拆兩級（同 ASD/slides_src/2026-10-28_redpen/make_stage_diag.py 的做法，多了每一層）：
   每一顆模型、每位受試者：只跑第 1 級後的 Dice、兩級跑完的 Dice、各級的平均位移；
   以及每一級裡面每一層（1/16、1/8、1/4、1/2、1）合成後的總形變大小（mm，腦內平均）。
   Dice 同 ASD/test_dice.py：30 個結構、seg 最近鄰搬、兩邊都沒有的結構不計（這裡在 GPU 上算，數字相同）。
B. 訓練過程圖：log\\<實驗名>.txt 每一輪 100 步的平均（影像項、平滑項、--dwp 的子層 loss ms）＋ dice_curve_val.csv。

輸出（models\\combo_ablation\\，gitignore）：stage.json、stage.png、train_curves.png
用法：python ASD\\analyze_combo_ablation.py           （A＋B）
      python ASD\\analyze_combo_ablation.py --plot-only  （A 已算過，只重畫）
"""
import os
import re
import sys
import glob
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUTD = os.path.join(ROOT, 'models', 'combo_ablation')
os.makedirs(OUTD, exist_ok=True)
for s in (sys.stdout, sys.stderr):
    s.reconfigure(encoding='utf-8', errors='replace')

# key：(資料夾, 選定的 epoch, 顯示名稱, 顏色, 線型)
RUNS = {
    'old': ('mix_cascade_pyramid', '0140', '不開（第 1 次）', '#4B5D73', '-'),
    'rep2': ('mix_cascade_pyramid_rep2', '0250', '不開（重練）', '#4B5D73', (0, (4, 2))),
    'rcn': ('mix_cascade_pyramid_rcn', '0200', '只開 --rcn（①②）', '#2F7FD0', '-'),
    'dwp': ('mix_cascade_pyramid_dwp', '0215', '只開 --dwp（③④）', '#0E7C7B', '-'),
    'both': ('mix_cascade_pyramid_rcn_dwp', '0155', '兩個都開', '#7A4FA0', '-'),
}
# 第 4 顆（--dwp --dwp-ms-stage first，④ 改加在第 1 級；11-13 排上 AI）：帶回來之後自動加進來，epoch 從 test CSV 的檔名讀
_ms1 = glob.glob(os.path.join(ROOT, 'models', 'mix_cascade_pyramid_dwp_ms1', 'dice_[0-9][0-9][0-9][0-9].csv'))
if len(_ms1) == 1:
    RUNS['ms1'] = ('mix_cascade_pyramid_dwp_ms1', os.path.basename(_ms1[0])[5:9], '--dwp，④ 改加在第 1 級', '#D08C2B', '-')
LV = ['1/16', '1/8', '1/4', '1/2', '1']
ap = argparse.ArgumentParser()
ap.add_argument('--plot-only', action='store_true')
args = ap.parse_args()
SJ = os.path.join(OUTD, 'stage.json')

# ════ A. 拆兩級 ════
if not args.plot_only:
    import torch
    import torch.nn.functional as F
    from arch import load_model, vxm
    dev = 'cuda'
    atlas = np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
    aseg = torch.from_numpy(np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int64)).to(dev)
    LABELS = np.load(os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))['labels'].astype(int)
    FG = torch.from_numpy(atlas > 0).to(dev)
    FGL = {k: (F.avg_pool3d(FG.float()[None, None], 2 ** k, 2 ** k)[0, 0] > 0.5) if k else FG for k in range(5)}
    files = sorted(glob.glob(os.path.join(ROOT, 'data', 'mixed_preprocessed_v2', 'test', '*.npz')))
    assert len(files) == 51
    tgt = torch.from_numpy(atlas)[None, None].to(dev)
    nn_warp = vxm.torch.layers.SpatialTransformer(atlas.shape, mode='nearest').to(dev)

    def dice(ws):
        d = []
        for l in LABELS:
            a, b = ws == l, aseg == l
            sa, sb = int(a.sum()), int(b.sum())
            if sa + sb:
                d.append(2.0 * int((a & b).sum()) / (sa + sb))
        return float(np.mean(d))

    def mag(u, k=0):                               # 腦內平均位移（mm）；第 k 層的單位是該層的格子，乘 2^k
        return float((torch.sqrt((u[0] ** 2).sum(0)) * 2 ** k)[FGL[k]].mean())

    OUT = {'n': len(files)}
    for key, (e, ep, *_r) in RUNS.items():
        m = load_model(os.path.join(ROOT, 'models', e, ep + '.pt'), dev).to(dev).eval()
        assert m.stage == 'pyramid' and len(m.stages) == 2, e
        R, LVM = [], []
        with torch.no_grad():
            for f in files:
                z = np.load(f)
                src = torch.from_numpy(z['vol'].astype(np.float32))[None, None].to(dev)
                seg = torch.from_numpy(z['seg'].astype(np.float32))[None, None].to(dev)
                U, row, lvrow = None, [], []
                for st in m.stages:                      # 同 VxmCascade.forward
                    s_in = src if U is None else m.transformer(src, U)
                    u, _pres, Us = st.fields(s_in, tgt, mult=m.flow_mult, keep_levels=True)
                    lvrow += [mag(Uk, st.n - i) for i, Uk in enumerate(Us)]     # 這一級內，每一層合成後的總形變
                    U = u if U is None else u + m.transformer(U, u)
                    ws = torch.round(nn_warp(seg, U)[0, 0]).long()
                    row += [dice(ws), mag(u), mag(U)]
                R.append(row)
                LVM.append(lvrow)
        R, LVM = np.array(R), np.array(LVM)
        OUT[key] = {'d1': R[:, 0].tolist(), 'u1': R[:, 1].tolist(), 'd2': R[:, 3].tolist(), 'u2': R[:, 4].tolist(),
                    'U': R[:, 5].tolist(), 'lv1': LVM[:, :5].mean(0).tolist(), 'lv2': LVM[:, 5:].mean(0).tolist()}
        o = OUT[key]
        print('%-28s 第 1 級後 %.4f（|φ1| %.2f mm）→ 兩級後 %.4f（|φ2| %.2f mm，總 %.2f mm）'
              % (e, np.mean(o['d1']), np.mean(o['u1']), np.mean(o['d2']), np.mean(o['u2']), np.mean(o['U'])), flush=True)
        print('    第 1 級各層總形變 mm：' + '、'.join('%s %.2f' % (l, v) for l, v in zip(LV, o['lv1'])))
        print('    第 2 級各層總形變 mm：' + '、'.join('%s %.2f' % (l, v) for l, v in zip(LV, o['lv2'])), flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(OUT, open(SJ, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    # 跟 test CSV 對答案：兩級後的 Dice 應該等於 test_dice.py 的結果
    import csv
    for key, (e, ep, *_r) in RUNS.items():
        ref = np.mean([float(r['dice_mean']) for r in csv.DictReader(open(os.path.join(ROOT, 'models', e, 'dice_%s.csv' % ep), encoding='utf-8'))])
        print('核對 %-6s 兩級後 %.4f，test CSV %.4f' % (key, np.mean(OUT[key]['d2']), ref))
        assert abs(np.mean(OUT[key]['d2']) - ref) < 1e-3, key

# ════ 畫圖 ════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, RULE, PAPER = '#141A1D', '#5F6A6B', '#D9D9D2', '#FAFAF8'
plt.rcParams['figure.facecolor'] = PAPER
plt.rcParams['axes.facecolor'] = PAPER


def clean(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=RULE, lw=0.6)
        ax.set_axisbelow(True)


def save(fig, name):
    p = os.path.join(OUTD, name)
    fig.savefig(p, dpi=180, facecolor=PAPER)
    plt.close(fig)
    print('[OK]', p)


S = json.load(open(SJ, encoding='utf-8'))
K = list(RUNS)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1.25]))
fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.2, wspace=0.28)
x = np.arange(len(K))
# (a) 第 1 級後、兩級後的 Dice
ax = axs[0]
for i, k in enumerate(K):
    c = RUNS[k][3]
    d1, d2 = np.mean(S[k]['d1']), np.mean(S[k]['d2'])
    ax.plot([i, i], [d1, d2], color=c, lw=1.2, alpha=0.6)
    ax.scatter([i], [d1], marker='o', s=60, facecolor='white', edgecolor=c, lw=1.8, zorder=3)
    ax.scatter([i], [d2], marker='o', s=60, color=c, zorder=3)
    ax.text(i + 0.12, d1, '%.4f' % d1, va='center', fontsize=8.5, color=MUTED)
    ax.text(i + 0.12, d2, '%.4f' % d2, va='center', fontsize=8.5, color=INK)
ax.set_xticks(x, [RUNS[k][2] for k in K], rotation=25, ha='right', fontsize=9)
ax.set_xlim(-0.4, len(K) - 0.2)
ax.set_title('(a) Dice：空心 = 只跑第 1 級、實心 = 兩級跑完', fontsize=10.5, loc='left', color=INK)
clean(ax)
# (b) 各級平均位移
ax = axs[1]
w = 0.36
for i, k in enumerate(K):
    c = RUNS[k][3]
    u1, u2 = np.mean(S[k]['u1']), np.mean(S[k]['u2'])
    ax.bar(i - w / 2, u1, w, color=c, alpha=0.45, edgecolor=c)
    ax.bar(i + w / 2, u2, w, color=c, edgecolor=c)
    ax.text(i - w / 2, u1 + 0.04, '%.2f' % u1, ha='center', fontsize=8, color=MUTED)
    ax.text(i + w / 2, u2 + 0.04, '%.2f' % u2, ha='center', fontsize=8, color=INK)
ax.set_xticks(x, [RUNS[k][2] for k in K], rotation=25, ha='right', fontsize=9)
ax.set_ylabel('腦內平均位移（mm）', fontsize=9.5, color=MUTED)
ax.set_title('(b) 各級的位移：淺 = 第 1 級 φ1、深 = 第 2 級 φ2', fontsize=10.5, loc='left', color=INK)
clean(ax)
# (c) 每一級裡，各層合成後的總形變
ax = axs[2]
for k in K:
    c, ls = RUNS[k][3], RUNS[k][4]
    ax.plot(range(5), S[k]['lv1'], color=c, ls=ls, lw=1.6, marker='o', ms=4, mfc='white', alpha=0.85)
    ax.plot(range(5, 10), S[k]['lv2'], color=c, ls=ls, lw=1.6, marker='o', ms=4, label=RUNS[k][2])
ax.axvline(4.5, color=RULE, lw=1)
ax.text(2, ax.get_ylim()[1], '第 1 級', ha='center', va='bottom', fontsize=9.5, color=MUTED)
ax.text(7, ax.get_ylim()[1], '第 2 級（④ 只加在 1/16～1/4）', ha='center', va='bottom', fontsize=9.5, color=MUTED)
ax.axvspan(4.6, 7.4, color='#E8F3F2', zorder=0)
ax.set_xticks(range(10), LV + LV, fontsize=8.5)
ax.set_ylabel('合成後之總形變（mm，腦內平均）', fontsize=9.5, color=MUTED)
ax.set_title('(c) 每一級裡，每一層合成後的總形變', fontsize=10.5, loc='left', color=INK, pad=16)
ax.legend(fontsize=8.5, frameon=False, loc='upper left')
clean(ax)
fig.suptitle('合併版拆兩級（test 51 位，各模型選定之 epoch）', fontsize=12, x=0.05, ha='left', color=INK)
save(fig, 'stage.png')

# ════ B. 訓練過程 ════
PAT = re.compile(r'epoch:\s*(\d+)\s+step:.*?loss:\s*(-?[\d.]+)\s*\((-?[\d.]+),\s*(-?[\d.]+)\)(?:\s*ms:\s*(-?[\d.]+))?')


def read_log(name):
    raw = open(os.path.join(ROOT, 'log', name + '.txt'), 'rb').read()
    for enc in ('utf-8', 'utf-16', 'cp950'):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if '�' not in t and 'epoch' in t:
            return t
    raise RuntimeError(name)


TR = {}
for k, (e, *_r) in RUNS.items():
    ep = {}
    for mm in PAT.finditer(read_log(e)):
        ep.setdefault(int(mm.group(1)), []).append([float(mm.group(3)), float(mm.group(4)),
                                                    float(mm.group(5)) if mm.group(5) else np.nan])
    assert len(ep) == 250 and all(len(v) == 100 for v in ep.values()), (k, len(ep))
    a = np.array([np.mean(ep[i], 0) for i in sorted(ep)])
    TR[k] = {'img': -a[:, 0], 'smooth': a[:, 1], 'ms': a[:, 2]}
VC = {}
for k, (e, *_r) in RUNS.items():
    import csv
    r = list(csv.DictReader(open(os.path.join(ROOT, 'models', e, 'dice_curve_val.csv'), encoding='utf-8')))
    VC[k] = (np.array([int(x['epoch']) for x in r]), np.array([float(x['dice_mean']) for x in r]))
E = np.arange(1, 251)


def roll(v, n=5):
    return np.convolve(v, np.ones(n) / n, mode='valid')


fig, axs = plt.subplots(2, 2, figsize=(14, 8.6))
fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.07, wspace=0.18, hspace=0.32)
for k in K:
    c, ls, lab = RUNS[k][3], RUNS[k][4], RUNS[k][2]
    axs[0, 0].plot(E, TR[k]['img'], color=c, ls=ls, lw=1.4, label=lab)
    axs[0, 1].plot(E, TR[k]['smooth'], color=c, ls=ls, lw=1.4, label=lab)
    ep, d = VC[k]
    m_ = ep >= 50
    axs[1, 0].plot(ep[m_], d[m_], color=c, ls=ls, lw=0.8, alpha=0.35)
    axs[1, 0].plot(ep[m_][2:-2], roll(d[m_]), color=c, ls=ls, lw=2, label=lab)
    if not np.all(np.isnan(TR[k]['ms'])):
        axs[1, 1].plot(E, -TR[k]['ms'], color=c, ls=ls, lw=1.4, label=lab)
axs[0, 0].set_ylim(0.2, 0.28)
axs[0, 0].set_title('(a) 訓練之影像相似度（最後輸出之 local NCC，越高越好；每輪 100 步平均）', fontsize=10.5, loc='left', color=INK)
axs[0, 1].set_title('(b) 訓練之平滑項（每級每層之 λ‖∇v‖² 總和，越低越平滑）', fontsize=10.5, loc='left', color=INK)
axs[0, 1].set_ylim(0.010, 0.022)
axs[1, 0].set_title('(c) val Dice（每 5 輪；細線 = 原始，粗線 = 5 點移動平均）', fontsize=10.5, loc='left', color=INK)
axs[1, 0].axvspan(150, 250, color=RULE, alpha=0.3, lw=0, zorder=0)
axs[1, 0].set_ylim(0.800, 0.820)
axs[1, 1].set_title('(d) --dwp 之子層 loss（④；取負號，越高越好）', fontsize=10.5, loc='left', color=INK)
inset = axs[0, 0].inset_axes([0.45, 0.1, 0.5, 0.5])
for k in K:
    inset.plot(E[150:], TR[k]['img'][150:], color=RUNS[k][3], ls=RUNS[k][4], lw=1.4)
inset.set_title('第 151～250 輪放大', fontsize=8.5, color=MUTED)
clean(inset)
for ax in axs.flat:
    ax.set_xlabel('epoch', fontsize=9.5, color=MUTED)
    clean(ax)
axs[0, 1].legend(fontsize=9, frameon=False, loc='upper right')
fig.suptitle('合併版五顆之訓練過程（設定相同，只差 --rcn／--dwp）', fontsize=12.5, x=0.06, ha='left', color=INK)
save(fig, 'train_curves.png')

# ════ 數字 ════
print('\n每一輪平均（第 1～50 輪 / 第 151～250 輪）：')
for k in K:
    print('  %-8s 影像相似度 %.4f / %.4f    平滑項 %.5f / %.5f' % (k, TR[k]['img'][:50].mean(), TR[k]['img'][150:].mean(),
                                                               TR[k]['smooth'][:50].mean(), TR[k]['smooth'][150:].mean()))
print('\n第 1 級後 Dice，配對比較（vs rep2）：')
for k in ('rcn', 'dwp', 'both'):
    a, b = np.array(S[k]['d1']), np.array(S['rep2']['d1'])
    print('  %-5s %.4f vs %.4f，%d/51 位較高，p = %.3g' % (k, a.mean(), b.mean(), int((a > b).sum()), wilcoxon(a, b).pvalue))
