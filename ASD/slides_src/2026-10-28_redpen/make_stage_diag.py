# -*- coding: utf-8 -*-
"""合併模型（cascade × coarse-to-fine）舊版 vs 照原作版：每一顆做完後的 Dice 與平均位移（test 51 位，只推論）。2026-10-28。

照原作補齊細節後（--rcn --dwp），合併模型從 0.8192 掉到 0.8149，比單獨的 cascade、coarse-to-fine 都低。
把兩顆拆開看：只跑第 1 顆的 Dice、兩顆都跑完的 Dice，以及每一顆在腦內的平均位移（mm）。
Dice 算法同 ASD/test_dice.py（30 結構、最近鄰搬 seg、兩邊都沒有的結構不計）。筆電 8 GB 約 25 分鐘。

  deck_data_stage.json   {'old': {...}, 'new': {...}}，每項是 51 位的平均：d1、d2（Dice）、u1、u2、U（mm）
用法：python ASD\\slides_src\\2026-10-28_redpen\\make_stage_diag.py
"""
import os
import sys
import glob
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'ASD'))
import torch                                       # noqa: E402
from arch import load_model, vxm                   # noqa: E402

MODELS = {'old': ('mix_cascade_pyramid', '0140'), 'new': ('mix_cascade_pyramid_rcn_dwp', '0155')}
dev = 'cuda'
atlas = np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
aseg = np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
LABELS = np.load(os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))['labels'].astype(int)
FG = torch.from_numpy(atlas > 0).to(dev)
files = sorted(glob.glob(os.path.join(ROOT, 'data', 'mixed_preprocessed_v2', 'test', '*.npz')))
tgt = torch.from_numpy(atlas)[None, None].to(dev)
nn_warp = vxm.torch.layers.SpatialTransformer(atlas.shape, mode='nearest').to(dev)


def dice(ws):
    d = [2.0 * ((ws == l) & (aseg == l)).sum() / ((ws == l).sum() + (aseg == l).sum()) for l in LABELS
         if (ws == l).sum() + (aseg == l).sum() > 0]
    return float(np.mean(d))


mag = lambda u: float(torch.sqrt((u[0] ** 2).sum(0))[FG].mean())
OUT = {'n': len(files), 'models': MODELS}
for key, (e, ep) in MODELS.items():
    m = load_model(os.path.join(ROOT, 'models', e, ep + '.pt'), dev).to(dev).eval()
    R = []
    with torch.no_grad():
        for f in files:
            z = np.load(f)
            src = torch.from_numpy(z['vol'].astype(np.float32))[None, None].to(dev)
            seg = torch.from_numpy(z['seg'].astype(np.float32))[None, None].to(dev)
            U, row = None, []
            for st in m.stages:                      # 同 VxmCascade.forward
                s_in = src if U is None else m.transformer(src, U)
                u = m._run_stage(st, s_in, tgt)[0]
                U = u if U is None else u + m.transformer(U, u)
                row += [dice(np.round(nn_warp(seg, U)[0, 0].cpu().numpy()).astype(np.int32)), mag(u), mag(U)]
            R.append(row)
    R = np.array(R)
    OUT[key] = {'d1': float(R[:, 0].mean()), 'u1': float(R[:, 1].mean()), 'd2': float(R[:, 3].mean()),
                'u2': float(R[:, 4].mean()), 'U': float(R[:, 5].mean()), 'd1_all': R[:, 0].tolist(), 'd2_all': R[:, 3].tolist()}
    print('%s：第 1 顆後 Dice %.4f（|u1| %.2f mm）→ 第 2 顆後 Dice %.4f（|u2| %.2f mm，總 |U| %.2f mm）'
          % (e, OUT[key]['d1'], OUT[key]['u1'], OUT[key]['d2'], OUT[key]['u2'], OUT[key]['U']), flush=True)
    del m
    torch.cuda.empty_cache()
d1o, d1n = np.array(OUT['old']['d1_all']), np.array(OUT['new']['d1_all'])
OUT['stage1_new_lower'] = int((d1n < d1o).sum())
with open(os.path.join(HERE, 'deck_data_stage.json'), 'w', encoding='utf-8') as fh:
    json.dump(OUT, fh, ensure_ascii=False, indent=1)
print('第 1 顆後照原作版較低：%d/%d 位' % (OUT['stage1_new_lower'], len(files)))
