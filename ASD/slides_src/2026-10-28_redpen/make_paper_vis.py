# -*- coding: utf-8 -*-
"""10/28 簡報「論文常見之視覺化」試作（2026-10-23 使用者：「每一個都蠻值得嘗試，先記起來…這幾個先試看看」）
→ models\\deck_charts\\1028_try_box.png、1028_try_lambda.png、1028_try_errgrid.png、1028_try_c2f.png、1028_try.json

  1. 每個結構的 Dice 箱形圖：test 51 位，左右合併、依 ANTs 平均排序——VoxelMorph（TMI 2019）Fig. 5 的畫法
  2. λ 對 Dice 的曲線（另一格是 folding）——VoxelMorph Fig. 7（論文用 validation，這裡用 test 的正式結果，跟簡報其他頁一致）
  3. 誤差圖（形變後影像 − atlas）＋變形網格——DWP（Jian et al.）Fig. 2 的第 3、4 列（誤差圖、warped grid，角落標平均位移）；
     10-27 加第三列：位移場 RGB（x、y、z → R、G、B，VoxelMorph Fig. 6 第 4 欄的顯示法；各方法同一刻度 ±8 mm）
  4. coarse-to-fine 每一層的總形變與對應的影像——Dual-PRNet++ Fig. 6（由粗到細各層的形變、對應的 warped volume）

資料
  Dice、HD95、folding：各模型 test CSV（dice_<epoch>.csv、surface_<epoch>.csv；ANTs 是 models\\ants_syn\\dice_syn_test.csv）
  形變後影像：make_ants_vis.py 的快取（1028_vis_cache_T054.npz、…_tm_mixv2.npz）、ANTs 的 models\\ants_syn\\vis\\T054.npz
  位移場：ANTs、TransMorph 用上面兩個檔裡的 flow；VoxelMorph 系列在這裡重跑推論，只存冠狀那一刀＋腦內平均位移，
          快取於 models\\deck_charts\\1028_try_cache_T054.npz；coarse-to-fine 各層用 arch.VxmPyramid.fields(keep_levels=True)
  位移場的慣例都是 VoxelMorph 的：moved(x) = moving(x + u(x))，u 定義在 atlas 的格子上、單位 voxel（= mm）

用法
  python ASD\\slides_src\\2026-10-28_redpen\\make_paper_vis.py
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import binary_fill_holes as ndi_fill

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
J = lambda *p: os.path.join(ROOT, *p)
sys.path.insert(0, J('ASD'))

ap = argparse.ArgumentParser()
ap.add_argument('--subject', default='T054')
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
S = args.subject

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#D9D9D2'
TEAL, BLUE, AMBER, RED, GRAY, PURPLE, SLATE, GREEN = '#0E7C7B', '#2F7FD0', '#D08C2B', '#B03A2E', '#9A9A93', '#7A4FA0', '#4B5D73', '#3E9B3A'
SLATE_L = '#8A9BB0'                              # TransMorph-diff（10-28）
PLUM, PLUM_L = '#8E5B8A', '#B98AB4'               # 半監督 γ = 0.5、5（10-28）
W = 12.13
OUT = J('models', 'deck_charts')
FG_VOX = 1867705                                 # atlas 非背景 voxel 數（jneg_fg_pct 之分母，同 make_arch_res.py）

atlas_vol = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
atlas_seg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
FG = atlas_vol > 0
assert int(FG.sum()) == FG_VOX
d = np.load(J('data', 'mixed_preprocessed_v2', 'test', S + '.npz'))
vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
LABELS = [int(x) for x in np.load(J('voxelmorph-code', 'data', 'labels.npz'))['labels']]


def read_csv(*p):
    with open(J('models', *p), encoding='utf-8') as fh:
        return {r['file']: r for r in csv.DictReader(fh)}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def clean(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(RULE); ax.spines['bottom'].set_color(RULE)
    ax.tick_params(colors=INK, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=RULE, lw=0.6)
        ax.set_axisbelow(True)


OUT_D = {}
# 三個架構修改模型讀 deck_models.json（換模型只改那個檔，見 換模型.md）
from deck_models import XC, EC, XP, EP, XM, EM

# ══ 1. 每個結構的 Dice 箱形圖（VoxelMorph Fig. 5）═══════════════════════════════════
# 30 個結構左右合併成 17 個（同論文「We average Dice scores of the left and right brain hemispheres」）
STRUCT17 = [('大腦白質', (2, 41)), ('大腦皮質', (3, 42)), ('側腦室', (4, 43)), ('小腦白質', (7, 46)), ('小腦皮質', (8, 47)),
            ('視丘', (10, 49)), ('尾核', (11, 50)), ('殼核', (12, 51)), ('蒼白球', (13, 52)), ('第三腦室', (14,)),
            ('第四腦室', (15,)), ('腦幹', (16,)), ('海馬', (17, 53)), ('杏仁核', (18, 54)), ('腦脊髓液', (24,)),
            ('腹側間腦', (28, 60)), ('脈絡叢', (31, 63))]
assert sorted(l for _, ls in STRUCT17 for l in ls) == sorted(LABELS)
BOXM = [('ants', 'ANTs SyN', PURPLE, ('ants_syn', 'dice_syn_test.csv')),
        ('mix_exp3', 'VoxelMorph（disp.）', GREEN, ('mix_exp3', 'dice_0240.csv')),
        ('mix_exp6', 'VoxelMorph（SVF）', TEAL, ('mix_exp6', 'dice_0190.csv')),
        ('tm_mixv2', 'TransMorph（disp.）', SLATE, ('tm_mixv2', 'dice_0205.csv')),
        # 10-28 使用者：「很多一起比較的圖沒有補上 transmorph diff」
        ('tm_diff_mixv2', 'TransMorph-diff', SLATE_L, ('tm_diff_mixv2', 'dice_0075.csv')),
        (XC, 'Cascade', BLUE, (XC, 'dice_%s.csv' % EC)),
        (XP, 'Coarse-to-fine', AMBER, (XP, 'dice_%s.csv' % EP)),
        (XM, 'Cascade + C2F', RED, (XM, 'dice_%s.csv' % EM)),
        # 10-28 使用者：「p21 可以加上半監督」（標籤項即為評估之 30 結構，標 *）
        ('mix_exp8', 'Semi-sup. γ = 0.5 *', PLUM, ('mix_exp8', 'dice_0250.csv')),
        ('mix_exp9', 'Semi-sup. γ = 5 *', PLUM_L, ('mix_exp9', 'dice_0245.csv'))]
TAB = {k: read_csv(*c) for k, _, _, c in BOXM}
FILES = sorted(TAB['ants'])
assert all(sorted(t) == FILES for t in TAB.values()) and len(FILES) == 51


def struct_vals(k, labs):
    """每位受試者：該結構左右兩邊 Dice 的平均（某邊沒有該結構時只取另一邊）。"""
    out = []
    for f in FILES:
        v = [float(TAB[k][f]['label_%d' % l]) for l in labs if TAB[k][f]['label_%d' % l] not in ('', 'nan')]
        out.append(np.nanmean(v) if v else np.nan)
    return np.array(out)


BV = {n: {k: struct_vals(k, labs) for k, _, _, _ in BOXM} for n, labs in STRUCT17}
order = sorted(BV, key=lambda n: -np.nanmean(BV[n]['ants']))     # 依 ANTs 平均 Dice 由高到低（論文的排法）
OUT_D['box'] = {n: {k: float(np.nanmean(BV[n][k])) for k, _, _, _ in BOXM} for n in order}
OUT_D['box_order'] = order

fig, axs = plt.subplots(2, 1, figsize=(W, 5.3), facecolor=PAPER)
fig.subplots_adjust(left=0.06, right=0.995, top=0.84, bottom=0.07, hspace=0.42)
halves = [order[:9], order[9:]]
# 10-28 加 TransMorph-diff 後，它的小腦（0.25～0.78）會把上排其他方法壓成一條線 → 上排縱軸自 YCUT 起，低於的標箭頭與中位數
YCUT = 0.68
nm = len(BOXM)
bw = 0.075 if nm > 8 else (0.095 if nm > 7 else 0.105)
for ax, names in zip(axs, halves):
    for i, n in enumerate(names):
        for j, (k, lab, c, _) in enumerate(BOXM):
            x = i + (j - (nm - 1) / 2) * (bw + 0.012)
            v = BV[n][k][~np.isnan(BV[n][k])]
            ax.boxplot(v, positions=[x], widths=bw, patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=c, edgecolor=c, alpha=0.85, lw=0.6),
                       medianprops=dict(color='white', lw=1.0), whiskerprops=dict(color=c, lw=0.7),
                       capprops=dict(color=c, lw=0.7), flierprops=dict(marker='.', ms=2.5, mfc=c, mec='none', alpha=0.7))
            if ax is axs[0] and np.nanmedian(v) < YCUT:
                ax.annotate('↓%.2f' % np.nanmedian(v), (x, YCUT + 0.004), ha='center', va='bottom', fontsize=7.5, color=c,
                            fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
        if i:
            ax.axvline(i - 0.5, color=RULE, lw=0.6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=10)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylabel('Dice', fontsize=10)
    clean(ax)
axs[0].set_ylim(YCUT, 1.0)
fig.legend([Patch(facecolor=c, edgecolor=c) for _, _, c, _ in BOXM], [lab for _, lab, _, _ in BOXM], loc='upper center',
           ncol=(nm + 1) // 2 if nm > 7 else nm, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, 0.875, 'test 51 位；左右兩側合併為一個結構；結構依 ANTs SyN 之平均 Dice 由高到低排列（同 VoxelMorph 論文 Fig. 5）',
         ha='center', fontsize=9, color=MUTED)
save(fig, '1028_try_box.png')

# ══ 2. λ 對 Dice 的曲線（VoxelMorph Fig. 7）＋ folding ════════════════════════════════
# 速度場（SVF，全尺寸積分）：λ 0.5／1／2 = mix_exp7／6／5；位移場（displacement）：λ 1／2 = mix_exp3／4。int_downsize 都是 1（實際權重 = λ）
LAM = {'SVF': [(0.5, 'mix_exp7', '0250'), (1.0, 'mix_exp6', '0190'), (2.0, 'mix_exp5', '0150')],
       'displacement': [(1.0, 'mix_exp3', '0240'), (2.0, 'mix_exp4', '0230')]}
OUT_D['lambda'] = {}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 4.0), facecolor=PAPER)
fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.14, wspace=0.25)
for fam, c, mk in [('SVF', TEAL, 'o'), ('displacement', GREEN, 's')]:
    xs, dm, se, fd = [], [], [], []
    for lam, e, ep in LAM[fam]:
        t = read_csv(e, 'surface_%s.csv' % ep)
        dv = np.array([float(r['dice_mean']) for r in t.values()])
        fv = np.array([float(r['jneg_fg_pct']) for r in t.values()]) * FG_VOX / 100
        assert len(dv) == 51
        xs.append(lam); dm.append(dv.mean()); se.append(dv.std(ddof=1) / np.sqrt(len(dv))); fd.append(fv.mean())
        OUT_D['lambda'].setdefault(fam, []).append({'lambda': lam, 'exp': e, 'dice': float(dv.mean()), 'fold': float(fv.mean())})
    lab = 'SVF（mix_exp5～7）' if fam == 'SVF' else 'Displacement field（mix_exp3、4）'
    a1.errorbar(xs, dm, yerr=se, color=c, marker=mk, ms=6, lw=1.6, capsize=3, label=lab)
    for x, y in zip(xs, dm):
        a1.annotate('%.4f' % y, (x, y), textcoords='offset points', xytext=(-8, 9) if fam == 'SVF' else (10, -4), ha='right' if fam == 'SVF' else 'left',
                    fontsize=9, color=c)
    a2.plot(xs, fd, color=c, marker=mk, ms=6, lw=1.6, label=lab)
    for x, y in zip(xs, fd):
        a2.annotate('{:,.1f}'.format(y) if y < 100 else '{:,.0f}'.format(y), (x, y), textcoords='offset points',
                    xytext=(8, 4), fontsize=9, color=c)
for a in (a1, a2):
    a.set_xscale('log', base=2)
    a.set_xticks([0.5, 1, 2]); a.set_xticklabels(['0.5', '1', '2'])
    a.set_xlim(0.4, 2.6)
    a.set_xlabel('λ（平滑項之權重）', fontsize=10)
    clean(a, 'both')
a1.set_ylabel('Dice（30 結構平均；test 51 位）', fontsize=10)
a1.set_title('(a) Dice（平均 ± 標準誤）', fontsize=10.5, loc='left')
a2.set_yscale('symlog', linthresh=1)
a2.set_ylim(-0.3, 1e5)
a2.set_ylabel('腦內 |J| ≤ 0 之 voxel 數（每位平均）', fontsize=10)
a2.set_title('(b) Folding（symlog 刻度）', fontsize=10.5, loc='left')
a1.legend(fontsize=9, frameon=False, loc='lower left')
fig.text(0.5, 0.955, 'NCC（窗格 9）＋ λ × 平滑項；全尺寸積分、250 epochs，其餘設定相同；同 VoxelMorph 論文 Fig. 7（論文為 validation，此處為 test）',
         ha='center', fontsize=9, color=MUTED)
save(fig, '1028_try_lambda.png')

# ══ 3、4 共用：冠狀面（同 make_ants_vis.py：經過 atlas 視丘的中心）＋位移場 ════════════════════════
ctr = lambda labs: np.argwhere(np.isin(atlas_seg, labs)).mean(0)
y0 = int(round(ctr([10, 49])[1]))
xs_, zs_ = np.where(FG[:, y0, :])
SX, SZ = slice(max(xs_.min() - 4, 0), xs_.max() + 5), slice(max(zs_.min() - 4, 0), zs_.max() + 5)
raw = lambda a: a[SX, y0, SZ]                    # 3D → 冠狀那一刀（還沒轉向）
cut2 = np.rot90                                  # 顯示：上方 = 上（z 大）、右方 = 受試者右側（x 大），同 make_ants_vis.py
CONT = [((4, 43), '#2F7FD0'), ((10, 49), '#D08C2B'), ((12, 51), '#7A4FA0'), ((17, 53), '#0E7C7B'),
        ((28, 60), '#D62728'), ((16,), '#9C6B3F')]   # 同 make_ants_vis.py 冠狀面的 6 個結構與顏色
CONT_N = ['側腦室', '視丘', '殼核', '海馬', '腹側間腦', '腦幹']

vis = dict(np.load(J('models', 'deck_charts', '1028_vis_cache_%s.npz' % S)))
A = np.load(J('models', 'ants_syn', 'vis', S + '.npz'))
TMc = np.load(J('models', 'deck_charts', '1028_vis_cache_%s_tm_mixv2.npz' % S))
TDc = np.load(J('models', 'deck_charts', '1028_vis_cache_%s_tm_diff_mixv2.npz' % S))    # 10-28 TransMorph-diff
assert 'flow' in TMc.files, 'TransMorph 快取沒有 flow：TransMorph 專案的 mixv2\\export_vis.py 重跑一次'
umag = lambda f: float(np.sqrt((f.astype(np.float32) ** 2).sum(0))[FG].mean())   # 腦內平均位移（mm）
WARP = {'moving': vol, 'ants': A['warped'], 'tm_mixv2': TMc['warped'], 'tm_diff_mixv2': TDc['warped']}
UCOR = {'moving': np.zeros((3,) + raw(vol).shape, np.float32), 'ants': np.stack([raw(c) for c in A['flow']]),
        'tm_mixv2': np.stack([raw(c) for c in TMc['flow'].astype(np.float32)]),
        'tm_diff_mixv2': np.stack([raw(c) for c in TDc['flow'].astype(np.float32)])}
UMAG = {'moving': 0.0, 'ants': umag(A['flow']), 'tm_mixv2': umag(TMc['flow']), 'tm_diff_mixv2': umag(TDc['flow'])}

SPECS = [('mix_exp3', '0240'), ('mix_exp6', '0190'), (XC, EC), (XP, EP), (XM, EM), ('mix_exp8', '0250'), ('mix_exp9', '0245')]
cpath = J('models', 'deck_charts', '1028_try_cache_%s.npz' % S)
C = dict(np.load(cpath)) if os.path.exists(cpath) else {}
C2F_OK = 'c2f_dice' in C and str(C.get('c2f_exp')) == XP + '_' + EP      # 各層的快取記著是哪一顆，換模型就重算
need = [(e, ep) for e, ep in SPECS if e + '_ucor' not in C] + ([(XP, EP)] if not C2F_OK else [])
if need:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '-1' else ''
    import torch
    from arch import load_model, vxm             # arch 先把 VXM_BACKEND 設成 pytorch 再 import voxelmorph
    dev = 'cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu'
    src = torch.from_numpy(vol)[None, None].to(dev)
    tgt = torch.from_numpy(atlas_vol)[None, None].to(dev)
    with torch.no_grad():
        for e, ep in SPECS:
            if e + '_ucor' in C:
                continue
            print('推論', e, ep)
            m = load_model(J('models', e, ep + '.pt'), dev).to(dev).eval()
            _, flow = m(src, tgt, registration=True)
            f = flow[0].cpu().numpy()
            C[e + '_ucor'] = np.stack([raw(c) for c in f]).astype(np.float32)
            C[e + '_umag'] = np.array(umag(f))
            del m, flow
        if not C2F_OK:                           # 4. coarse-to-fine 各層（VxmPyramid.fields 的 keep_levels）
            print('由粗到細各層', XP, EP)
            m = load_model(J('models', XP, EP + '.pt'), dev).to(dev).eval()
            U, _, Us = m.fields(src, tgt, keep_levels=True)
            nn_warp = vxm.torch.layers.SpatialTransformer(vol.shape, mode='nearest').to(dev)
            segt = torch.from_numpy(seg.astype(np.float32))[None, None].to(dev)
            imgs, ucs, dices, ums = [], [], [], []
            for i, Uk in enumerate(Us):           # Us[0] = 最粗（1/16）… Us[-1] = 原尺寸
                for _ in range(len(Us) - 1 - i):  # 放大回原尺寸（同模型裡的 up_flow：內插、向量 ×2）
                    Uk = m.up_flow(Uk)
                w = m.warp[0](src, Uk)[0, 0].cpu().numpy()
                ws = nn_warp(segt, Uk)[0, 0].cpu().numpy().astype(np.int32)
                dl = [(2.0 * ((ws == l) & (atlas_seg == l)).sum() / ((ws == l).sum() + (atlas_seg == l).sum()))
                      for l in LABELS if (ws == l).sum() + (atlas_seg == l).sum() > 0]
                f = Uk[0].cpu().numpy()
                imgs.append(raw(w)); ucs.append(np.stack([raw(c) for c in f])); dices.append(np.mean(dl)); ums.append(umag(f))
            assert np.allclose(Us[-1].cpu().numpy(), U.cpu().numpy())
            C['c2f_img'], C['c2f_ucor'] = np.stack(imgs).astype(np.float32), np.stack(ucs).astype(np.float32)
            C['c2f_dice'], C['c2f_umag'], C['c2f_exp'] = np.array(dices), np.array(ums), np.array(XP + '_' + EP)
            del m
    np.savez_compressed(cpath, **C)
for e, _ in SPECS:
    WARP[e] = vis[e + '_warped']
    UCOR[e], UMAG[e] = C[e + '_ucor'], float(C[e + '_umag'])

# 對答案：c2f 最後一層的 Dice 要等於該模型 test CSV 的 T054（同一個算法）
d_csv = float(read_csv(XP, 'dice_%s.csv' % EP)[S + '.npz']['dice_mean'])
assert abs(C['c2f_dice'][-1] - d_csv) < 2e-3, (C['c2f_dice'][-1], d_csv)


def grid(ax, u, step=5, color=INK, lw=0.35):
    """變形網格：atlas 上每 step 個 voxel 一條線，畫 x + u(x) 的平面分量（冠狀：左右 = 軸 0、上下 = 軸 2）。"""
    dh, dv = cut2(u[0]), -cut2(u[2])             # rot90 後列號往下增加 = z 往下減少，所以上下分量取負號
    H, Wd = dh.shape
    ii, jj = np.mgrid[0:H, 0:Wd]
    X, Y = jj + dh, ii + dv
    for i in range(0, H, step):
        ax.plot(X[i], Y[i], color=color, lw=lw)
    for j in range(0, Wd, step):
        ax.plot(X[:, j], Y[:, j], color=color, lw=lw)
    ax.set_xlim(-0.5, Wd - 0.5); ax.set_ylim(H - 0.5, -0.5)


def tag(ax, s, dark=False, fs=7.5):
    ax.text(0.03, 0.03, s, transform=ax.transAxes, fontsize=fs, va='bottom', color='white' if dark else INK,
            bbox=dict(boxstyle='round,pad=0.15', fc='black' if dark else 'white', ec='none', alpha=0.6 if dark else 0.85))


# ══ 3. 誤差圖＋變形網格（DWP Fig. 2 第 3、4 列）＋位移場 RGB ═══════════════════════════════
# 10-27 使用者：「P21 是不是還可以加那種形變場表示方向的 RGB 圖」「直接加進第 21 頁」→ 第三列。
# x、y、z 位移各對應 R、G、B 是 VoxelMorph（TMI 2019）Fig. 6 第 4 欄的顯示法（圖說：「Displacement in each spatial dimension is mapped
# to each of the RGB color channels」，沒寫刻度）；這裡所有方法用同一個刻度（±RGB_MM），大小才能互相比：分量 = 0 為中間灰、+RGB_MM 該色最亮、−RGB_MM 最暗。
# 版面：原本兩組各 5 欄 × 2 列（誤差、網格），改成 10 欄排一列 × 3 列，圖下面放條列（每格大小跟原本差不多）
RGB_MM = 8.0
NAME = {'moving': 'Moving\n（affine）', 'ants': 'ANTs SyN', 'mix_exp3': 'VoxelMorph\n（disp.）', 'mix_exp6': 'VoxelMorph\n（SVF）',
        'tm_mixv2': 'TransMorph\n（disp.）', 'tm_diff_mixv2': 'TransMorph\n-diff', XC: 'Cascade', XP: 'Coarse-\nto-fine',
        XM: 'Cascade\n+ C2F', 'mix_exp8': 'Semi-sup.*\n（γ = 0.5）', 'mix_exp9': 'Semi-sup.*\n（γ = 5）'}
KEYS = ['moving', 'ants', 'mix_exp3', 'mix_exp6', 'tm_mixv2', 'tm_diff_mixv2', XC, XP, XM, 'mix_exp8', 'mix_exp9']
BREAK = {6, 9}                                   # 這幾欄前面多空一點：傳統＋原始架構 | 架構修改 | 半監督
fgc = cut2(raw(FG))
FILLED = ndi_fill(fgc).astype(float)               # RGB 列只描腦的外框（腦室那些洞不描）
MAE = {k: float(np.abs(WARP[k] - atlas_vol)[FG].mean()) for k in NAME}
OUT_D['errgrid'] = {k: {'mae': MAE[k], 'umag': UMAG[k]} for k in NAME}
OUT_D['rgb_mm'] = RGB_MM
OUT_D['rgb_clip'] = {k: float((np.abs(UCOR[k]) > RGB_MM)[:, raw(FG)].mean()) for k in NAME}     # 腦內分量超出刻度的比例
h, w = fgc.shape
FW, LM, GC, GX, LEGW, SUP, TT, G1, BB = 12.13, 0.24, 0.05, 0.13, 1.3, 0.3, 0.36, 0.04, 0.04
NK = len(KEYS)
pw = (FW - LM - 0.03 - (NK - 1) * GC - len(BREAK) * GX - GC - LEGW) / NK
ph = pw * h / w
FH = SUP + TT + 3 * ph + 2 * G1 + BB
fig = plt.figure(figsize=(FW, FH), facecolor=PAPER)
xs = []
for j in range(NK):
    xs.append(LM + j * (pw + GC) + sum(GX for b in BREAK if j >= b))
tops = [SUP + TT + r * (ph + G1) for r in range(3)]
ax_at = lambda r, x, wd=pw: fig.add_axes([x / FW, (FH - tops[r] - ph) / FH, wd / FW, ph / FH])
for r, lab in enumerate(['誤差', '網格', '位移場']):
    fig.text(LM * 0.45 / FW, (FH - tops[r] - ph / 2) / FH, lab, rotation=90, ha='center', va='center', fontsize=9, color=MUTED)


def rgb(u):
    """位移場 → RGB：R = 左右（x）、G = 前後（y，冠狀面的穿面方向）、B = 上下（z）；±RGB_MM 線性對應 0～1。"""
    return np.clip(0.5 + np.stack([cut2(u[c]) for c in range(3)], -1) / (2 * RGB_MM), 0, 1)


for j, k in enumerate(KEYS):
    a, b, c = ax_at(0, xs[j]), ax_at(1, xs[j]), ax_at(2, xs[j])
    for q in (a, b, c):
        q.set_xticks([]); q.set_yticks([])
    a.imshow(cut2(raw(WARP[k] - atlas_vol)), cmap='RdBu_r', vmin=-0.5, vmax=0.5, interpolation='nearest')
    a.set_title(NAME[k], fontsize=8.5, color=INK, pad=2, linespacing=1.0)
    tag(a, 'MAE %.3f' % MAE[k], fs=6.8)
    b.imshow(np.where(fgc, 0.9, 1.0), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    grid(b, UCOR[k], lw=0.3)
    tag(b, '|u| %.2f' % UMAG[k], fs=6.8)
    c.imshow(rgb(UCOR[k]), interpolation='nearest')
    c.contour(FILLED, levels=[0.5], colors='white', linewidths=0.4, alpha=0.8)
# 說明欄（右邊）
lx = xs[-1] + pw + GC + 0.06
la = ax_at(0, lx, LEGW - 0.06); la.axis('off')
cax = fig.add_axes([(lx + 0.1) / FW, (FH - tops[0] - 0.1) / FH, (LEGW - 0.3) / FW, 0.07 / FH])
cb = fig.colorbar(ScalarMappable(Normalize(-0.5, 0.5), 'RdBu_r'), cax=cax, orientation='horizontal', ticks=[-0.5, 0, 0.5])
cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
la.text(0.45, 0.4, '形變後影像 − atlas\n（影像值 0～1）\n左下：腦內 MAE', ha='center', va='center', fontsize=7, color=MUTED,
        transform=la.transAxes)
lb = ax_at(1, lx, LEGW - 0.06); lb.axis('off')
lb.text(0.45, 0.5, 'atlas 上每 5 mm 一條線\n經 x + u(x) 映射\n（平面分量）\n左下：腦內平均 |u|（mm）', ha='center', va='center',
        fontsize=7, color=MUTED, transform=lb.transAxes)
lc = ax_at(2, lx, LEGW - 0.06); lc.axis('off')
for n, (col, t) in enumerate([((1, 0.5, 0.5), 'R：左右（x）'), ((0.5, 1, 0.5), 'G：前後（y）'), ((0.5, 0.5, 1), 'B：上下（z）')]):
    yy = 0.86 - n * 0.2
    lc.add_patch(plt.Rectangle((0.02, yy - 0.07), 0.13, 0.14, color=col, transform=lc.transAxes, clip_on=False))
    lc.text(0.2, yy, t, va='center', fontsize=7, color=INK, transform=lc.transAxes)
lc.text(0.02, 0.12, '分量 ±%g mm 飽和\n0 為中間灰；白線＝腦' % RGB_MM, va='center', fontsize=6.6, color=MUTED, transform=lc.transAxes)
fig.text(0.5, 1 - 0.14 / FH, '%s，冠狀面（經過視丘）；誤差圖與變形網格同 DWP（Jian et al.）Fig. 2，位移場 RGB 同 VoxelMorph Fig. 6'
         '（各方法同一刻度）' % S, ha='center', va='center', fontsize=9.5, color=MUTED)
save(fig, '1028_try_errgrid.png')

# ══ 4. coarse-to-fine 每一層（Dual-PRNet++ Fig. 6）═══════════════════════════════════
LV = ['1/16', '1/8', '1/4', '1/2', '原尺寸']
OUT_D['c2f'] = {'level': LV, 'dice': [float(x) for x in C['c2f_dice']], 'umag': [float(x) for x in C['c2f_umag']],
                'affine_dice': float(read_csv('mix_exp2', 'dice_baseline.csv')[S + '.npz']['dice_mean'])}
cols = ['fixed', 'moving'] + list(range(5))
h, w = fgc.shape
ncol = len(cols)
pw = (W - 0.3 - (ncol - 1) * 0.06 - 0.05) / ncol
ph = pw * h / w
FH = 0.32 + 0.36 + 2 * ph + 0.08 + 0.05
fig = plt.figure(figsize=(W, FH), facecolor=PAPER)
for j, c in enumerate(cols):
    x0 = 0.3 + j * (pw + 0.06)
    a = fig.add_axes([x0 / W, (FH - 0.32 - 0.36 - ph) / FH, pw / W, ph / FH])
    b = fig.add_axes([x0 / W, (FH - 0.32 - 0.36 - 2 * ph - 0.08) / FH, pw / W, ph / FH])
    for q in (a, b):
        q.set_xticks([]); q.set_yticks([])
    img = atlas_vol[SX, y0, SZ] if c == 'fixed' else (raw(vol) if c == 'moving' else C['c2f_img'][c])
    a.imshow(cut2(img), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    for labs, cc in CONT:
        a.contour(cut2(raw(np.isin(atlas_seg, labs))).astype(float), levels=[0.5], colors=cc, linewidths=0.7)
    if c == 'fixed':
        a.set_title('Fixed（atlas）', fontsize=9.5, color=INK, pad=3)
        b.axis('off')
        b.legend([Line2D([0], [0], color=cc, lw=1.6) for _, cc in CONT], ['atlas ' + n for n in CONT_N], loc='center',
                 fontsize=8, frameon=False, handlelength=1.2, labelspacing=0.3)
        continue
    if c == 'moving':
        a.set_title('Moving（affine）', fontsize=9.5, color=INK, pad=3)
        tag(a, 'Dice %.3f' % OUT_D['c2f']['affine_dice'], dark=True, fs=8)
        u, um = UCOR['moving'], 0.0
    else:
        a.set_title('第 %d 層（%s）' % (c + 1, LV[c]), fontsize=9.5, color=INK, pad=3)
        tag(a, 'Dice %.3f' % C['c2f_dice'][c], dark=True, fs=8)
        u, um = C['c2f_ucor'][c], float(C['c2f_umag'][c])
    b.imshow(np.where(fgc, 0.9, 1.0), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    grid(b, u)
    tag(b, '|u| %.2f mm' % um, fs=8)
fig.text(0.3 * 0.45 / W, (FH - 0.32 - 0.36 - ph / 2) / FH, '形變後影像', rotation=90, ha='center', va='center', fontsize=9, color=MUTED)
fig.text(0.3 * 0.45 / W, (FH - 0.32 - 0.36 - 1.5 * ph - 0.08) / FH, '累積形變（網格）', rotation=90, ha='center', va='center',
         fontsize=9, color=MUTED)
fig.text(0.5, 1 - 0.16 / FH, '%s，冠狀面；%s（coarse-to-fine）由粗到細每一層合成後之總形變 U_k（放大到原尺寸）及其形變後影像；'
         'Dice 為 30 結構平均；同 Dual-PRNet++ Fig. 6' % (S, XP), ha='center', va='center', fontsize=9.5, color=MUTED)
save(fig, '1028_try_c2f.png')

pj = os.path.join(OUT, '1028_try.json')
json.dump(OUT_D, open(pj, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('->', pj)
print('λ：', json.dumps(OUT_D['lambda'], ensure_ascii=False))
print('c2f：', ['%s %.3f %.2fmm' % (l, dd, uu) for l, dd, uu in zip(LV, OUT_D['c2f']['dice'], OUT_D['c2f']['umag'])])
print('MAE/|u|：', {k: ('%.4f' % v['mae'], '%.2f' % v['umag']) for k, v in OUT_D['errgrid'].items()})
