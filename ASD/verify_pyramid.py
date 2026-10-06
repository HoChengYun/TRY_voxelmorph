# -*- coding: utf-8 -*-
"""驗證 ASD/arch.py 的由粗到細網路 VxmPyramid（2026-10-06；CLAUDE.md 待辦 5 第 2 步）。四件事都要過：

1. 一開始幾乎不動：每一層出速度場的卷積初始權重很小（同 VxmDense），總位移要接近 0、搬好的影像≈原圖
2. 平移測試（驗證每一層放大 2 倍、向量也乘 2、位移接法、x／y／z 三個方向沒有弄反）：
   把每一層的速度場都設成 0，只在幾層加一個固定的平移，總位移（離邊界遠的地方）要剛好等於各層平移換算到原尺寸的和：
       最粗層 (c, 0, 0) → 原尺寸 (16c, 0, 0)；第 2 層 (0, 0, d) → (0, 0, 4d)；原尺寸 (0, b, 0) → (0, b, 0)
3. 存檔、再用 load_model() 讀回來 → 輸出一模一樣，config 記著 arch = pyramid
4. 參數量（跟 VxmDense 的 301,411 比）

兩個疊在一起（VxmCascade stage='pyramid'，2026-10-06 加）：
5. 只串 1 顆、放同一組權重 → 跟由粗到細本身一模一樣
6. 串 2 顆、兩顆放同一組權重 → 跟「由粗到細自己跑兩次、手動把位移接起來」一樣
7. 平移測試：第 1 顆只在最粗層加 (c, 0, 0)、第 2 顆只在原尺寸加 (0, b, 0) → 總位移 (16c, b, 0)
8. 存檔、再用 load_model() 讀回來 → 輸出一模一樣，config 記著 arch = cascade、stage = pyramid

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
from arch import VxmPyramid, VxmCascade, load_model         # noqa: E402

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
    del m2, mb, ub

    # 5. 串 1 顆由粗到細 = 由粗到細本身
    sd = model.state_dict()
    c1 = VxmCascade(inshape, n_cascades=1, int_downsize=1, stage='pyramid').to(device).eval()
    c1.stages[0].load_state_dict(sd)
    m1, u1 = c1(v, a, registration=True)
    e = max((u1 - ua).abs().max().item(), (m1 - ma).abs().max().item())
    print('[%s] 5. 串 1 顆由粗到細 vs 由粗到細本身：最大差 %.1e' % ('v' if e < 1e-5 else 'X', e))
    ok &= e < 1e-5
    del c1, m1, u1

    # 6. 串 2 顆（同一組權重）= 由粗到細自己跑兩次、手動接起來
    warp = model.warp[0]
    _, U1 = model(v, a, registration=True)
    _, u2 = model(warp(v, U1), a, registration=True)
    U_ref = u2 + warp(U1, u2)
    c2 = VxmCascade(inshape, n_cascades=2, int_downsize=1, stage='pyramid').to(device).eval()
    for s in c2.stages:
        s.load_state_dict(sd)
    _, U2 = c2(v, a, registration=True)
    e = (U2 - U_ref).abs().max().item()
    print('[%s] 6. 串 2 顆（同一組權重）vs 自己跑兩次再接：總位移最大差 %.1e（總位移最大 %.2f 格）'
          % ('v' if e < 1e-4 else 'X', e, U2.abs().max().item()))
    ok &= e < 1e-4
    del U1, u2, U_ref, U2

    # 7. 疊在一起的平移測試
    for s in c2.stages:
        for f in s.flows:
            f.weight.zero_()
            f.bias.zero_()
    c2.stages[0].flows[c2.stages[0].n].bias.copy_(torch.tensor([c, 0.0, 0.0]))   # 第 1 顆的最粗層
    c2.stages[1].flows[0].bias.copy_(torch.tensor([0.0, b, 0.0]))                 # 第 2 顆的原尺寸
    _, Ut = c2(v, a, registration=True)
    inner = Ut[0, :, 48:-48, 56:-56, 48:-48]
    want = torch.tensor([c * 2 ** c2.stages[0].n, b, 0.0], device=device).view(3, 1, 1, 1)
    e = (inner - want).abs().max().item()
    print('[%s] 7. 疊在一起的平移測試：要 (%.2f, %.2f, 0)，最大差 %.1e' % ('v' if e < 1e-3 else 'X', c * 16, b, e))
    ok &= e < 1e-3
    del Ut

    # 8. 存檔 → load_model 讀回來
    for s in c2.stages:
        for f in s.flows:
            f.weight.normal_(0, 1e-2)
    c2.save(tmp)
    c3 = load_model(tmp, device).to(device).eval()
    os.remove(tmp)
    _, ux = c2(v, a, registration=True)
    _, uy = c3(v, a, registration=True)
    e = (ux - uy).abs().max().item()
    good = (type(c3).__name__ == 'VxmCascade' and c3.config.get('stage') == 'pyramid'
            and c3.config.get('n_cascades') == 2 and e == 0)
    print('[%s] 8. 疊在一起存檔再讀回來：%s，config arch=%s stage=%s，輸出最大差 %.1e'
          % ('v' if good else 'X', type(c3).__name__, c3.config.get('arch'), c3.config.get('stage'), e))
    ok &= good

# 4. 參數量
n_par = sum(p.numel() for p in model.parameters())
print('[i] 4. 參數 %d 個（VxmDense 301,411 個，%.2f 倍）；疊在一起（串 2 顆）%d 個'
      % (n_par, n_par / 301411, sum(p.numel() for p in c2.parameters())))

print()
print('全部通過' if ok else '有沒通過的，看上面 [X]')
sys.exit(0 if ok else 1)
