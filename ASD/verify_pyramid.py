# -*- coding: utf-8 -*-
"""驗證 ASD/arch.py 的由粗到細網路 VxmPyramid（2026-10-06；CLAUDE.md 待辦 5 第 2 步）。四件事都要過：

1. 一開始幾乎不動：每一層出速度場的卷積初始權重很小（同 VxmDense），總位移要接近 0、搬好的影像≈原圖
2. 平移測試（驗證每一層放大 2 倍、向量也乘 2、位移接法、x／y／z 三個方向沒有弄反）：
   把每一層的速度場都設成 0，只在幾層加一個固定的平移，總位移（離邊界遠的地方）要剛好等於各層平移換算到原尺寸的和：
       最粗層 (c, 0, 0) → 原尺寸 (16c, 0, 0)；第 2 層 (0, 0, d) → (0, 0, 4d)；原尺寸 (0, b, 0) → (0, b, 0)
3. 存檔、再用 load_model() 讀回來 → 輸出一模一樣，config 記著 arch = pyramid
4. 參數量（跟 VxmDense 的 301,411 比）

用法：python ASD\\verify_pyramid.py --test-dir data\\mixed_preprocessed_v2\\test
"""
import os
import sys
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
ap.add_argument('--test-dir', required=True)
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch                                    # noqa: E402
from arch import VxmPyramid, load_model         # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
atlas = np.load(args.atlas)['vol'].astype(np.float32)
f0 = sorted(glob.glob(os.path.join(os.path.normpath(args.test_dir), '*.npz')))[0]
v = torch.from_numpy(np.load(f0)['vol'].astype(np.float32))[None, None].to(device)
a = torch.from_numpy(atlas)[None, None].to(device)
inshape = atlas.shape
ok = True

with torch.no_grad():
    # 1. 一開始幾乎不動
    model = VxmPyramid(inshape).to(device).eval()
    moved, U = model(v, a, registration=True)
    e_u, e_m = U.abs().max().item(), (moved - v).abs().max().item()
    good = e_u < 0.1 and tuple(U.shape) == (1, 3) + inshape
    print('[%s] 1. 一開始：總位移最大 %.1e 格、搬好的影像跟原圖最大差 %.1e（位移場大小 %s）'
          % ('v' if good else 'X', e_u, e_m, tuple(U.shape)))
    ok &= good

    # 2. 平移測試。用真實大小：最粗那層 12×14×12；影像太小的話最粗層只剩幾格，
    #    積分、放大時「邊界外補 0」的影響會蓋到中間（試過 64×80×64，最粗層 4×5×4，x 方向就差 0.12）
    small = inshape
    t = VxmPyramid(small).to(device).eval()
    for f in t.flows:
        f.weight.zero_()
        f.bias.zero_()
    c, d, b = 0.25, 0.5, 1.5
    t.flows[t.n].bias.copy_(torch.tensor([c, 0.0, 0.0]))     # 最粗層（1/16）
    t.flows[2].bias.copy_(torch.tensor([0.0, 0.0, d]))      # 第 2 層（1/4）
    t.flows[0].bias.copy_(torch.tensor([0.0, b, 0.0]))      # 原尺寸
    x = torch.rand((1, 1) + small, device=device)
    _, Us = t(x, x, registration=True)
    inner = Us[0, :, 48:-48, 56:-56, 48:-48]                  # 離邊界至少 3 個最粗格（邊界外補 0 會影響最外圈）
    want = torch.tensor([c * 2 ** t.n, b, d * 4], device=device).view(3, 1, 1, 1)
    e = (inner - want).abs().max().item()
    print('[%s] 2. 平移測試：要 (%.2f, %.2f, %.2f)，實際中間區域 (%.4f, %.4f, %.4f)，最大差 %.1e'
          % ('v' if e < 1e-3 else 'X', c * 2 ** t.n, b, d * 4,
             inner[0].mean().item(), inner[1].mean().item(), inner[2].mean().item(), e))
    ok &= e < 1e-3

    # 3. 存檔 → load_model 讀回來（先把網路弄成「有在動」，不然兩邊都≈0 比不出來）
    for f in model.flows:
        f.weight.normal_(0, 1e-2)
    tmp = os.path.join(tempfile.gettempdir(), 'verify_pyramid_tmp.pt')
    model.save(tmp)
    m2 = load_model(tmp, device).to(device).eval()
    os.remove(tmp)
    ma, ua = model(v, a, registration=True)
    mb, ub = m2(v, a, registration=True)
    e = max((ua - ub).abs().max().item(), (ma - mb).abs().max().item())
    good = type(m2).__name__ == 'VxmPyramid' and m2.config.get('arch') == 'pyramid' and e == 0 and ua.abs().max() > 0.1
    print('[%s] 3. 存檔再讀回來：%s，config arch=%s，輸出最大差 %.1e（總位移最大 %.2f 格）'
          % ('v' if good else 'X', type(m2).__name__, m2.config.get('arch'), e, ua.abs().max().item()))
    ok &= good

# 4. 參數量
n_par = sum(p.numel() for p in model.parameters())
print('[i] 4. 參數 %d 個（VxmDense 301,411 個，%.2f 倍）' % (n_par, n_par / 301411))

print()
print('全部通過' if ok else '有沒通過的，看上面 [X]')
sys.exit(0 if ok else 1)
