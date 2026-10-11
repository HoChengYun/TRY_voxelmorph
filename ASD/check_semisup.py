#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""半監督（mix_exp8／9）的 Dice 大幅上升，是不是只發生在「訓練時就拿來算標籤項」的那 30 個結構上？2026-10-11。

mix_exp8／9 的標籤項（train_semisup.py）用的就是 Dice 評估的同一組 30 個結構（labels.npz），
所以 test Dice 0.90／0.93 有一部分是「訓練目標＝評估指標」。這支用三件事檢查：
  ① 沒參與訓練的結構（FreeSurfer 有標、但不在 30 個裡面）的 Dice：
       Accumbens（26、58）、Inf-Lat-Vent（5、44）、胼胝體（251～255 合併成一個）
     —— 論文（Balakrishnan et al., TMI 2019）§V-C 也是這樣檢查：訓練用一部分結構、評估沒用到的結構
  ② 影像本身對得多好：搬過去的影像 vs atlas 的局部 NCC（窗格 9，跟訓練的相似度項同一種）、非背景內的 MSE
     —— 標籤項會不會拿影像對齊去換
  ③ 30 個結構的 Dice（跟 test_dice.py 的 dice_<epoch>.csv 對答案，確認搬法一樣）

做法照抄 test_dice.py：registration=True 拿全尺寸 pos_flow、標籤用 mode='nearest' 搬、影像用 bilinear。
affine（不做形變）也算一次當起點。

用法：
    python ASD\\check_semisup.py --test-dir data\\mixed_preprocessed_v2\\test ^
        --models mix_exp6\\0190 mix_cascade\\0170 mix_exp8\\0250 mix_exp9\\0245
    加寬 2 倍的模型（mix_wide_vel）在 8 GB 筆電 GPU 塞不下，另外跑一次加 --gpu -1（CPU，51 位約 15 分鐘）
輸出：models\\<實驗>\\semisup_check_<epoch>.csv（每位一列）、models\\semisup_check_affine.csv，螢幕印平均與配對比較。
"""
import os
import sys
import glob
import csv
import argparse
import numpy as np
import torch

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'voxelmorph-code'))
import voxelmorph as vxm
from arch import load_model

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ap = argparse.ArgumentParser()
ap.add_argument('--test-dir', required=True)
ap.add_argument('--models', nargs='+', required=True, help='models 底下的「實驗\\epoch」，例如 mix_exp8\\0250')
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--ref', default=None, help='配對比較的對照（預設：--models 的第一個）')
ap.add_argument('--gpu', default='0', help='-1：網路與搬移改在 CPU 跑（加寬 2 倍的模型在 8 GB 筆電 GPU 塞不下）；NCC 仍在 GPU 算')
args = ap.parse_args()

device = 'cuda' if torch.cuda.is_available() and args.gpu != '-1' else 'cpu'
mdev = 'cuda' if torch.cuda.is_available() else 'cpu'     # NCC 用的裝置（losses.py 的 NCC 寫死 .to("cuda")）
L30 = np.load(args.labels)['labels'].astype(int).tolist()
# 沒參與訓練的結構：FreeSurfer 有標、test 51 位都有、且夠大（最少也有約 200 voxel）。視交叉（85，最小 44）、血管（30、62）太小不用
HELD = {'accumbens_L': [26], 'accumbens_R': [58], 'inflatvent_L': [5], 'inflatvent_R': [44],
        'corpus_callosum': [251, 252, 253, 254, 255]}
assert not any(l in L30 for v in HELD.values() for l in v), '保留結構不能跟 30 個重疊'

atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
FG = atlas_vol > 0
inshape = atlas_vol.shape
atlas_t = torch.from_numpy(atlas_vol)[None, None].to(device)
atlas_m = torch.from_numpy(atlas_vol)[None, None].to(mdev)
warp_lin = vxm.torch.layers.SpatialTransformer(inshape, mode='bilinear').to(device)
warp_nn = vxm.torch.layers.SpatialTransformer(inshape, mode='nearest').to(device)
ncc = vxm.losses.NCC()                     # 窗格 9，跟訓練的相似度項同一種（loss = −mean(cc)）
files = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))


def dice_set(a, b, labs):
    x, y = np.isin(a, labs), np.isin(b, labs)
    s = x.sum() + y.sum()
    return np.nan if s == 0 else 2.0 * (x & y).sum() / s


def row(name, seg_w, moved):
    r = {'file': name}
    r['dice30'] = float(np.nanmean([dice_set(seg_w, atlas_seg, [l]) for l in L30]))
    for k, labs in HELD.items():
        r[k] = dice_set(seg_w, atlas_seg, labs)
    r['heldout_mean'] = float(np.nanmean([r['accumbens_L'], r['accumbens_R'], r['inflatvent_L'], r['inflatvent_R'],
                                          r['corpus_callosum']]))
    m = torch.from_numpy(moved)[None, None].to(mdev)
    with torch.no_grad():
        r['ncc'] = float(-ncc.loss(atlas_m, m))
    r['mse_fg'] = float(((moved - atlas_vol)[FG] ** 2).mean())
    return r


def run(model_path):
    rows = []
    if model_path is None:                 # affine：不做形變
        for f in files:
            d = np.load(f)
            rows.append(row(os.path.basename(f), d['seg'].astype(np.int32), d['vol'].astype(np.float32)))
        return rows
    model = load_model(model_path, device).to(device)
    model.eval()
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            v = torch.from_numpy(d['vol'].astype(np.float32))[None, None].to(device)
            _, flow = model(v, atlas_t, registration=True)
            s = torch.from_numpy(d['seg'].astype(np.float32))[None, None].to(device)
            seg_w = np.round(warp_nn(s, flow)[0, 0].cpu().numpy()).astype(np.int32)
            moved = warp_lin(v, flow)[0, 0].cpu().numpy()
            rows.append(row(os.path.basename(f), seg_w, moved))
    return rows


def save(rows, out):
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('->', out)


KEYS = ['dice30', 'heldout_mean', 'accumbens_L', 'accumbens_R', 'inflatvent_L', 'inflatvent_R', 'corpus_callosum', 'ncc', 'mse_fg']
res = {}
out = os.path.join(ROOT, 'models', 'semisup_check_affine.csv')
res['affine'] = run(None)
save(res['affine'], out)
for m in args.models:
    exp, ep = os.path.normpath(m).split(os.sep)[-2:]
    res[m] = run(os.path.join(ROOT, 'models', exp, ep + '.pt'))
    save(res[m], os.path.join(ROOT, 'models', exp, 'semisup_check_%s.csv' % ep))

print('\n%-20s' % '' + ''.join('%15s' % k for k in KEYS))
for k, rows in res.items():
    print('%-20s' % k + ''.join('%15.4f' % np.nanmean([r[c] for r in rows]) for c in KEYS))

from scipy.stats import wilcoxon
ref = args.ref or args.models[0]
print('\n配對比較（相對 %s；Wilcoxon）' % ref)
for m in args.models:
    if m == ref:
        continue
    for c in ('dice30', 'heldout_mean', 'corpus_callosum', 'accumbens_L', 'accumbens_R', 'ncc', 'mse_fg'):
        a = np.array([r[c] for r in res[m]], float)
        b = np.array([r[c] for r in res[ref]], float)
        ok = ~(np.isnan(a) | np.isnan(b))
        p = wilcoxon(a[ok], b[ok]).pvalue
        print('  %-20s %-16s %+.4f  （%d/%d 位較高，p = %.2g）' % (m, c, (a - b)[ok].mean(), (a > b)[ok].sum(), ok.sum(), p))
