# -*- coding: utf-8 -*-
"""10/28 簡報 ⑧：同一位受試者（T054）各方法之形變結果，冠狀、矢狀、水平三個方向各一張，外加各結構之 Dice
→ models\\deck_charts\\1028_ants_vis.png（冠狀）、1028_ants_vis_sagittal.png、1028_ants_vis_axial.png、1028_ants_vis_dice.json

10-23 使用者：「放一下好了（單獨 cascade、coarse-to-fine）」「再放 VoxelMorph 速度場」「還有半監督」「把 table 放到其他 page」
「同方向 log|J| 和形變後影像放一頁」→ 每個方向一張 4 列 × 6 欄，分兩段：
  第 1 段：Fixed、Moving、ANTs SyN、VoxelMorph（displacement，mix_exp3）、VoxelMorph（SVF，mix_exp6）、TransMorph
  第 2 段：Cascade、Coarse-to-fine、Cascade + coarse-to-fine、半監督 γ = 0.5（mix_exp8）、γ = 5（mix_exp9）
  每段上列為形變後影像 m∘φ ＋ atlas 之結構輪廓，下列為同一方法之 log|J|（atlas 空間；|J| ≤ 0 以黃點標示）
  Fixed、Moving 沒有形變，下面兩格放圖例
切面：冠狀面經過 atlas 視丘的中心；矢狀面中線往左 --offset 個 voxel（老師：矢狀面不要切正中間；使用者：「偏 2、3 個 voxel 那種」）；
      水平面在殼核與視丘中心高度的中間。三個方向都是受試者左側在影像左側。
輪廓（10-23「區域結構可以放更多」）：都是 Dice 評估的 30 個結構，每個方向挑該切面上面積夠大的 6～7 個，同一個結構同一種顏色；
      大腦皮質、白質到處都是，畫了會蓋滿，不畫；胼胝體不在 30 個結構內，不畫。

ANTs 之結果讀 models\\ants_syn\\vis\\<受試者>.npz（ASD/ants_syn_baseline.py --save-dir；位移場定義在 atlas 格子上、指向受試者，
與 VoxelMorph 之 SpatialTransformer 同一慣例，--check 已驗證）；VoxelMorph 各模型在這裡重跑推論（GPU），結果快取於
models\\deck_charts\\1028_vis_cache_<受試者>.npz（缺哪個模型才補算哪個）。Dice 取各模型 test CSV 之該列（正式結果），不在這裡重算。
TransMorph 要 timm（vxm_env 沒有），推論在 TransMorph 專案先做（mixv2\\export_vis.py），存成
models\\deck_charts\\1028_vis_cache_<受試者>_tm_mixv2.npz。

用法
  python ASD\\slides_src\\2026-10-28_redpen\\make_ants_vis.py --subject T054
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
J = lambda *p: os.path.join(ROOT, *p)
sys.path.insert(0, J('ASD'))

ap = argparse.ArgumentParser()
ap.add_argument('--subject', default='T054')
ap.add_argument('--gpu', default='0')
ap.add_argument('--offset', type=int, default=2, help='矢狀面：中線往左偏幾個 voxel')
args = ap.parse_args()

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
W, HMAX = 12.13, 5.4                             # 投影片上放圖的範圍（吋）：圖照原尺寸放，字的大小才準

atlas_vol = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
atlas_seg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
S = args.subject
d = np.load(J('data', 'mixed_preprocessed_v2', 'test', S + '.npz'))
vol = d['vol'].astype(np.float32)
FG = atlas_vol > 0

# (key, 圖上的名稱, test CSV)；半監督的標籤項就是評估的 30 個結構，Dice 不能直接比（手冊 §27.2）→ 標 *
# 三個架構修改模型讀 deck_models.json（換模型只改那個檔，見 換模型.md）；推論快取以實驗名為 key，換了會自動補算
from deck_models import XC, EC, XP, EP, XM, EM
METHODS = [('fixed', 'Fixed\n（atlas）', None),
           ('moving', 'Moving\n（affine）', ('mix_exp2', 'dice_baseline.csv')),
           ('ants', 'ANTs SyN', ('ants_syn', 'dice_syn_test.csv')),
           ('mix_exp3', 'VoxelMorph\n（disp.）', ('mix_exp3', 'dice_0240.csv')),
           ('mix_exp6', 'VoxelMorph\n（SVF）', ('mix_exp6', 'dice_0190.csv')),
           ('tm_mixv2', 'TransMorph\n（disp.）', ('tm_mixv2', 'dice_0205.csv')),
           # 10-28 使用者：「很多一起比較的圖沒有補上 transmorph diff」→ 第一段多一欄（官方訓練方式，MSE）
           ('tm_diff_mixv2', 'TransMorph\n-diff', ('tm_diff_mixv2', 'dice_0075.csv')),
           (XC, 'Cascade', (XC, 'dice_%s.csv' % EC)),
           (XP, 'Coarse-to-fine', (XP, 'dice_%s.csv' % EP)),
           (XM, 'Cascade\n+ C2F', (XM, 'dice_%s.csv' % EM)),
           ('mix_exp8', 'Semi-sup.\n（γ = 0.5）', ('mix_exp8', 'dice_0250.csv')),
           ('mix_exp9', 'Semi-sup.\n（γ = 5）', ('mix_exp9', 'dice_0245.csv'))]
NAME = {k: n for k, n, _ in METHODS}
CSV = {k: J('models', *c) for k, _, c in METHODS if c}
SEMI = {'mix_exp8', 'mix_exp9'}
GROUPS = [['fixed', 'moving', 'ants', 'mix_exp3', 'mix_exp6', 'tm_mixv2', 'tm_diff_mixv2'],
          [XC, XP, XM, 'mix_exp8', 'mix_exp9', None]]
SPECS = [('mix_exp3', '0240'), ('mix_exp6', '0190'), (XC, EC), (XP, EP), (XM, EM), ('mix_exp8', '0250'), ('mix_exp9', '0245')]


def jac_det(f):
    g = [[np.gradient(f[c], axis=a) for a in range(3)] for c in range(3)]
    return ((1 + g[0][0]) * ((1 + g[1][1]) * (1 + g[2][2]) - g[1][2] * g[2][1])
            - g[0][1] * (g[1][0] * (1 + g[2][2]) - g[1][2] * g[2][0])
            + g[0][2] * (g[1][0] * g[2][1] - (1 + g[1][1]) * g[2][0]))


# ── ANTs ────────────────────────────────────────────────────────────
pa = J('models', 'ants_syn', 'vis', S + '.npz')
if not os.path.exists(pa):
    sys.exit('[X] 還沒有 %s（ASD/ants_syn_baseline.py --save-dir models\\ants_syn\\vis 跑完才有）' % pa)
A = np.load(pa)
R = {'ants': {'warped': A['warped'], 'det': A['det']}}

# ── VoxelMorph 各模型（快取；缺哪個才補算哪個）─────────────────────────────────
cache = J('models', 'deck_charts', '1028_vis_cache_%s.npz' % S)
C = dict(np.load(cache)) if os.path.exists(cache) else {}
todo = [(e, ep) for e, ep in SPECS if e + '_warped' not in C]
if todo:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '-1' else ''
    import torch
    from arch import load_model
    dev = 'cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for e, ep in todo:
            print('推論', e, ep)
            m = load_model(J('models', e, ep + '.pt'), dev).to(dev).eval()
            moved, flow = m(torch.from_numpy(vol)[None, None].to(dev), torch.from_numpy(atlas_vol)[None, None].to(dev),
                            registration=True)
            C[e + '_warped'] = moved[0, 0].cpu().numpy()
            C[e + '_det'] = jac_det(flow[0].cpu().numpy()).astype(np.float32)
            del m, moved, flow
            if dev == 'cuda':
                torch.cuda.empty_cache()
    np.savez_compressed(cache, **C)
for e, _ in SPECS:
    R[e] = {'warped': C[e + '_warped'], 'det': C[e + '_det']}

# ── TransMorph（TransMorph 專案的 mixv2/export_vis.py 先存好）─────────────────────────────
for tk in ('tm_mixv2', 'tm_diff_mixv2'):
    ptm = J('models', 'deck_charts', '1028_vis_cache_%s_%s.npz' % (S, tk))
    if not os.path.exists(ptm):
        sys.exit('[X] 還沒有 %s（TransMorph 專案 mixv2\\export_vis.py 先跑）' % ptm)
    c = np.load(ptm)
    R[tk] = {'warped': c['warped'], 'det': c['det']}


def test_row(path):
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['file'] == S + '.npz':
                return r


ROW = {k: test_row(p) for k, p in CSV.items()}
DICE = {k: float(r['dice_mean']) for k, r in ROW.items()}

# ── 切面（RAS：軸 0 = 左→右、軸 1 = 後→前、軸 2 = 下→上）─────────────────────────
ctr = lambda labs: np.argwhere(np.isin(atlas_seg, labs)).mean(0)
SD_ = {'側腦室': ((4, 43), '#2F7FD0'), '視丘': ((10, 49), '#D08C2B'), '殼核': ((12, 51), '#7A4FA0'),
       '海馬': ((17, 53), '#0E7C7B'), '尾核': ((11, 50), '#3E9B3A'), '蒼白球': ((13, 52), '#E377C2'),
       '腹側間腦': ((28, 60), '#D62728'), '腦幹': ((16,), '#9C6B3F'), '第四腦室': ((15,), '#17BECF'),
       '第三腦室': ((14,), '#BCBD22'), '小腦皮質': ((8, 47), '#B5179E')}
pick = lambda *ns: [(SD_[n][0], SD_[n][1], n) for n in ns]


def crop2(m2):
    """二維遮罩外框外推 4～5 個 voxel 的範圍。"""
    a, b = np.where(m2)
    return slice(max(a.min() - 4, 0), a.max() + 5), slice(max(b.min() - 4, 0), b.max() + 5)


VIEWS = []
y = int(round(ctr([10, 49])[1]))                 # 冠狀面：經過 atlas 視丘的中心
sx, sz = crop2(FG[:, y, :])
VIEWS.append(dict(cut=lambda a, sx=sx, sz=sz: np.rot90(a[sx, y, sz]), out='1028_ants_vis.png', where='冠狀面（經過視丘）',
                  struct=pick('側腦室', '視丘', '殼核', '海馬', '腹側間腦', '腦幹')))
# 矢狀面：中線 = 左右成對結構中心的中點之平均（atlas 約 95.9），往左 --offset 個 voxel；前方在右
mid = np.mean([(ctr(a)[0] + ctr(b)[0]) / 2 for a, b in [(2, 41), (3, 42), (4, 43), (10, 49), (17, 53), (12, 51), (8, 47)]])
x = int(round(mid)) - args.offset
sy, sz2 = crop2(FG[x, :, :])
VIEWS.append(dict(cut=lambda a, sy=sy, sz=sz2: np.rot90(a[x, sy, sz]), out='1028_ants_vis_sagittal.png',
                  where='矢狀面（中線往左 %d voxels）' % args.offset,
                  struct=pick('側腦室', '視丘', '腹側間腦', '腦幹', '第三腦室', '第四腦室', '小腦皮質')))
z = int(round((ctr([12, 51])[2] + ctr([10, 49])[2]) / 2))   # 水平面：殼核與視丘中心高度的中間；前方在上
sx3, sy3 = crop2(FG[:, :, z])
VIEWS.append(dict(cut=lambda a, sx=sx3, sy=sy3: np.rot90(a[sx, sy, z]), out='1028_ants_vis_axial.png',
                  where='水平面（經過基底核與視丘）', struct=pick('側腦室', '視丘', '殼核', '尾核', '蒼白球', '海馬')))

# 版面（吋）：4 列 × NC 欄（10-28 起 7 欄）；兩段各「影像＋log|J|」；面板大小由高度決定（投影片高 HMAX）
SUP, TT, G1, G2, BB, LM, GC = 0.3, 0.36, 0.04, 0.1, 0.05, 0.24, 0.05
NC = max(len(g) for g in GROUPS)


def draw(V):
    cut, STRUCT = V['cut'], V['struct']
    h, w = cut(FG).shape
    ph = (HMAX - SUP - 2 * TT - 2 * G1 - G2 - BB) / 4
    pw = ph * w / h
    if LM + NC * pw + (NC - 1) * GC + 0.05 > W:  # 太寬就改由寬度決定
        pw = (W - LM - (NC - 1) * GC - 0.05) / NC
        ph = pw * h / w
    FS = 9 if pw >= 1.0 else 8                   # 水平面的格子較窄（約 0.85 吋），字小一號
    FW, FH = LM + NC * pw + (NC - 1) * GC + 0.05, SUP + 2 * TT + 4 * ph + 2 * G1 + G2 + BB
    fig = plt.figure(figsize=(FW, FH), facecolor=PAPER)
    tops = [SUP + TT, SUP + TT + ph + G1, SUP + 2 * TT + 2 * ph + G1 + G2, SUP + 2 * TT + 3 * ph + 2 * G1 + G2]
    ax_at = lambda r, j: fig.add_axes([(LM + j * (pw + GC)) / FW, (FH - tops[r] - ph) / FH, pw / FW, ph / FH])
    for r, lab in enumerate(['影像', 'log|J|', '影像', 'log|J|']):
        fig.text(LM * 0.45 / FW, (FH - tops[r] - ph / 2) / FH, lab, rotation=90, ha='center', va='center', fontsize=FS, color=MUTED)
    fmask = cut(FG)
    for g, keys in enumerate(GROUPS):
        for j, k in enumerate(keys):
            a, b = ax_at(2 * g, j), ax_at(2 * g + 1, j)
            for q in (a, b):
                q.set_xticks([]); q.set_yticks([])
            if k is None:                        # 第 2 段最後一欄：半監督的註記、log|J| 色條
                a.axis('off')
                a.text(0.5, 0.5, '* 半監督：\n訓練時之標籤項\n即為評估之 30 結構\nDice 不可\n直接比較',
                       ha='center', va='center', fontsize=FS - 1, color=MUTED, transform=a.transAxes)
                b.axis('off')
                cax = fig.add_axes([(LM + j * (pw + GC) + 0.08 * pw) / FW, (FH - tops[2 * g + 1] - 0.45 * ph) / FH,
                                    0.84 * pw / FW, 0.07 / FH])
                cb = fig.colorbar(ScalarMappable(Normalize(-1, 1), 'RdBu_r'), cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
                cb.ax.tick_params(labelsize=FS - 1.5, length=2, pad=1)
                cb.set_label('log|J|', fontsize=FS - 1, labelpad=1)
                continue
            img = atlas_vol if k == 'fixed' else (vol if k == 'moving' else R[k]['warped'])
            a.imshow(cut(img), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            for labs, colr, _ in STRUCT:
                a.contour(cut(np.isin(atlas_seg, labs)).astype(float), levels=[0.5], colors=colr, linewidths=0.6)
            a.set_title(NAME[k], fontsize=FS, color=INK, pad=2, linespacing=1.0)
            if k in DICE:
                a.text(0.03, 0.03, 'Dice %.3f%s' % (DICE[k], '*' if k in SEMI else ''), transform=a.transAxes, fontsize=FS - 1.5,
                       color='white', va='bottom', bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.55))
            if k in ('fixed', 'moving'):         # 沒有形變：下面兩格放圖例
                b.axis('off')
                if k == 'fixed':
                    b.text(0.5, 0.5, 'log|J|\n（atlas 空間）\n紅：體積放大\n藍：體積縮小\n黃點：|J| ≤ 0\n淺灰：不計\n左下數字：\n腦內 |J| ≤ 0',
                           ha='center', va='center', fontsize=FS - 1.5, color=MUTED, transform=b.transAxes, linespacing=1.15)
                else:
                    b.legend([Line2D([0], [0], color=cc, lw=1.6) for _, cc, _ in STRUCT], ['atlas ' + n for _, _, n in STRUCT],
                             loc='center', fontsize=FS - 1.5, frameon=False, handlelength=1.2, labelspacing=0.25, borderaxespad=0)
                continue
            det = R[k]['det']
            lj = np.log(np.clip(det, 1e-3, None))
            b.imshow(np.full(fmask.shape, 0.85), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            b.imshow(np.ma.masked_where(~fmask, cut(lj)), cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            fy, fx = np.where(cut(det <= 0) & fmask)
            b.plot(fx, fy, '.', color='#FFD400', ms=2.2, mec='#5A4A00', mew=0.2)
            b.text(0.03, 0.03, '{:,}'.format(int(((det <= 0) & FG).sum())), transform=b.transAxes, fontsize=FS - 1.5, color=INK,
                   va='bottom', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    fig.text(0.5, 1 - 0.14 / FH, '%s，%s；輪廓為 atlas 之結構；Dice 為 30 結構之平均' % (S, V['where']),
             ha='center', va='center', fontsize=FS + 1, color=MUTED)
    p = J('models', 'deck_charts', V['out'])
    fig.savefig(p, dpi=200, facecolor=PAPER)
    plt.close(fig)
    print('-> %s（%.2f × %.2f 吋）' % (p, FW, FH))


for V in VIEWS:
    draw(V)

# ── 各結構之 Dice（另一頁的表）：三個方向畫到的結構，左右成對者取左右平均；加 30 結構平均 ─────────────────
ORDER_S = ['側腦室', '第三腦室', '第四腦室', '視丘', '尾核', '殼核', '蒼白球', '海馬', '腹側間腦', '腦幹', '小腦皮質']
sd = {n: {k: float(np.mean([float(ROW[k]['label_%d' % l]) for l in SD_[n][0]])) for k in CSV} for n in ORDER_S}
sd['30 結構平均'] = dict(DICE)
pj = J('models', 'deck_charts', '1028_ants_vis_dice.json')
json.dump({'subject': S, 'methods': [k for k, _, c in METHODS if c], 'names': {k: NAME[k] for k in CSV},
           'semi': sorted(SEMI), 'struct_dice': sd}, open(pj, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('->', pj)
for k in CSV:
    print('  %-20s Dice %.3f%s' % (k, DICE[k], '' if k == 'moving' else '  腦內 |J| ≤ 0：{:,}'.format(int(((R[k]['det'] <= 0) & FG).sum()))))
