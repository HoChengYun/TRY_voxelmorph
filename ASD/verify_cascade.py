# -*- coding: utf-8 -*-
"""驗證 ASD/arch.py 的串接網路（2026-10-06；CLAUDE.md 待辦 5）。四件事都要過：

1. 只串 1 顆、放進一顆現成模型的權重 → 輸出跟 VxmDense.forward 一模一樣（stage_flows 抄得對）
2. 串 2 顆、兩顆都放同一顆的權重 → 每位的 Dice 要跟 test_multipass.py 第 2 次一樣
   （位移接法對，而且跟第 0 步「同一顆連跑兩次」是同一件事）
3. 存檔、再用 load_model() 讀回來 → 輸出一模一樣，config 記著 arch / n_cascades
4. load_model() 讀以前的模型 → 還是 VxmDense（舊的 .pt 不受影響）

用法（先跑過 test_multipass.py，第 2 項要用它的 CSV 對答案）：
    python ASD\\verify_cascade.py --model models\\mix_exp6\\0190.pt --test-dir data\\mixed_preprocessed_v2\\test --n 3
"""
import os
import sys
import csv
import glob
import argparse
import tempfile
import numpy as np

os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
os.environ.setdefault('VXM_BACKEND', 'pytorch')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, help='一顆現成的 VxmDense .pt')
ap.add_argument('--test-dir', required=True)
ap.add_argument('--n', type=int, default=3, help='第 2 項對幾位')
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch                                                                  # noqa: E402
import voxelmorph as vxm                                                      # noqa: E402
from arch import load_model, cascade_from_single                              # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mp = os.path.normpath(args.model)
atlas = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
LABELS = np.load(args.labels)['labels'].astype(int).tolist()
files = sorted(glob.glob(os.path.join(os.path.normpath(args.test_dir), '*.npz')))[:args.n]
a_t = torch.from_numpy(atlas)[None, None].to(device)
warp_nn = vxm.torch.layers.SpatialTransformer(atlas.shape, mode='nearest').to(device)


def load_subject(f):
    d = np.load(f)
    return (torch.from_numpy(d['vol'].astype(np.float32))[None, None].to(device),
            torch.from_numpy(d['seg'].astype(np.float32))[None, None].to(device))


def dice_mean(seg_t, flow):
    sw = np.round(warp_nn(seg_t, flow)[0, 0].cpu().numpy()).astype(np.int32)
    vals = []
    for lab in LABELS:
        x, y = sw == lab, atlas_seg == lab
        s = x.sum() + y.sum()
        vals.append(np.nan if s == 0 else 2.0 * (x & y).sum() / s)
    return float(np.nanmean(vals))


ok = True
with torch.no_grad():
    v, s = load_subject(files[0])

    # 1. 串 1 顆 = VxmDense
    single = vxm.networks.VxmDense.load(mp, device).to(device).eval()
    c1 = cascade_from_single(mp, 1, device).to(device).eval()
    m0, f0 = single(v, a_t, registration=True)
    m1, f1 = c1(v, a_t, registration=True)
    e = max((f0 - f1).abs().max().item(), (m0 - m1).abs().max().item())
    print('[%s] 1. 串 1 顆 vs VxmDense：形變場、搬好的影像最大差 %.1e' % ('v' if e < 1e-5 else 'X', e))
    ok &= e < 1e-5
    del single, c1, m0, f0, m1, f1

    # 2. 串 2 顆（同一顆權重）= test_multipass.py 第 2 次
    ep = os.path.basename(mp)[:-3]
    ref_csv = os.path.join(os.path.dirname(mp), 'multipass_%s.csv' % ep)
    c2 = cascade_from_single(mp, 2, device).to(device).eval()
    if not os.path.exists(ref_csv):
        print('[X] 2. 找不到 %s（先跑 test_multipass.py）' % ref_csv)
        ok = False
    else:
        with open(ref_csv, encoding='utf-8') as fh:
            ref = {r['file']: float(r['dice_mean']) for r in csv.DictReader(fh) if r['pass'] == '2'}
        diffs = []
        for f in files:
            vv, ss = load_subject(f)
            _, U = c2(vv, a_t, registration=True)
            dm = dice_mean(ss, U)
            diffs.append(abs(dm - ref[os.path.basename(f)]))
            print('      %-14s 串 2 顆 %.4f   第 0 步第 2 次 %.4f' % (os.path.basename(f), dm, ref[os.path.basename(f)]))
        e = max(diffs)
        print('[%s] 2. 串 2 顆（同一顆權重）vs test_multipass 第 2 次：%d 位，Dice 最大差 %.1e'
              % ('v' if e < 1e-3 else 'X', len(diffs), e))
        ok &= e < 1e-3

    # 3. 存檔 → load_model 讀回來
    tmp = os.path.join(tempfile.gettempdir(), 'verify_cascade_tmp.pt')
    c2.save(tmp)
    c2b = load_model(tmp, device).to(device).eval()
    os.remove(tmp)
    ma, fa = c2(v, a_t, registration=True)
    mb, fb = c2b(v, a_t, registration=True)
    e = max((fa - fb).abs().max().item(), (ma - mb).abs().max().item())
    good = (type(c2b).__name__ == 'VxmCascade' and c2b.config.get('arch') == 'cascade'
            and c2b.config.get('n_cascades') == 2 and e == 0)
    print('[%s] 3. 存檔再讀回來：%s，config arch=%s n_cascades=%s，輸出最大差 %.1e'
          % ('v' if good else 'X', type(c2b).__name__, c2b.config.get('arch'), c2b.config.get('n_cascades'), e))
    ok &= good

    # 4. 舊模型照舊
    old = load_model(mp, device)
    good = type(old).__name__ == 'VxmDense'
    print('[%s] 4. load_model 讀以前的模型：%s' % ('v' if good else 'X', type(old).__name__))
    ok &= good

print()
print('全部通過' if ok else '有沒通過的，看上面 [X]')
sys.exit(0 if ok else 1)
