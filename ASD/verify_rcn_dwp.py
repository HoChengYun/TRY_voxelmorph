# -*- coding: utf-8 -*-
"""驗證 2026-10-21 照原作程式補上的開關（ASD/arch.py 的 flow_mult、border、detach_compose；train_arch.py --rcn、--dwp）。

1. 舊模型完全不受影響：同一組權重、同一位受試者，git 裡改之前的 arch.py（--old 給路徑）跟現在的 arch.py
   算出來的總位移逐 voxel 相同；Dice 跟 AI 上的正式 CSV 一致（mix_cascade、mix_pyramid、mix_cascade_pyramid）
2. 每個開關做的事（小影像、隨機權重，讓形變不是 0）：
   flow_mult = 0.5 → 每顆的速度場剛好是一半；detach_compose → 往前算的形變完全一樣、只有梯度不同；
   border → 跟補 0 的差別集中在影像邊界附近；return_levels → 各層總形變的大小、最後一層＝輸出
3. 訓練跑得起來（--crop、--max-steps）、--load-model 開關不一致時會擋下來：另外用 train_arch.py 跑（見手冊 §29）

用法：python ASD\\verify_rcn_dwp.py --old <改之前的 arch.py> --gpu 0
"""
import os
import sys
import csv
import argparse
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--old', required=True, help='改之前的 arch.py（git show HEAD~:ASD/arch.py > 某處）')
ap.add_argument('--subject', default='A0131')
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch                                             # noqa: E402
import arch as new                                       # noqa: E402

spec = importlib.util.spec_from_file_location('arch_old', args.old)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
dev = 'cuda'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print('  [%s] %s' % ('v' if cond else 'X', msg), flush=True)


# ── 1. 舊模型 ────────────────────────────────────────────────────────
print('1. 舊模型：改之前 vs 現在的 arch.py（同一組權重）')
atlas = np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
aseg = np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
labels = np.load(os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))['labels'].astype(int)
d = np.load(os.path.join(ROOT, 'data', 'mixed_preprocessed_v2', 'test', args.subject + '.npz'))
src = torch.from_numpy(d['vol'].astype(np.float32))[None, None].to(dev)
tgt = torch.from_numpy(atlas)[None, None].to(dev)
seg = torch.from_numpy(d['seg'].astype(np.float32))[None, None].to(dev)
import voxelmorph as vxm                                 # noqa: E402
warp_nn = vxm.torch.layers.SpatialTransformer(atlas.shape, mode='nearest').to(dev)
for e, ep in [('mix_cascade', '0170'), ('mix_pyramid', '0245'), ('mix_cascade_pyramid', '0140')]:
    p = os.path.join(ROOT, 'models', e, ep + '.pt')
    with torch.no_grad():
        mo = old.load_model(p, dev).to(dev).eval()
        _, Uo = mo(src, tgt, registration=True)
        Uo = Uo.cpu()
        del mo
        mn = new.load_model(p, dev).to(dev).eval()
        _, Un = mn(src, tgt, registration=True)
        sw = np.round(warp_nn(seg, Un)[0, 0].cpu().numpy()).astype(np.int32)
        Un = Un.cpu()
    cfg_new = {k: mn.config.get(k) for k in ('flow_mult', 'border', 'detach_compose')}
    del mn
    torch.cuda.empty_cache()
    diff = float((Un - Uo).abs().max())
    dice = np.nanmean([2 * ((sw == l) & (aseg == l)).sum() / max(1, (sw == l).sum() + (aseg == l).sum()) for l in labels])
    ref = {r['file']: float(r['dice_mean']) for r in csv.DictReader(open(os.path.join(ROOT, 'models', e, 'dice_%s.csv' % ep),
                                                                        encoding='utf-8'))}[args.subject + '.npz']
    check(diff == 0.0, '%-20s 總位移最大差 %.3g（要 0）；存檔裡沒有新開關：%s' % (e, diff, cfg_new))
    check(abs(dice - ref) < 5e-4, '%-20s %s 的 Dice %.6f，正式 CSV %.6f（差 %.1e）' % (e, args.subject, dice, ref, dice - ref))

# ── 2. 開關 ─────────────────────────────────────────────────────────
print('2. 開關（96x112x96、隨機權重：輸出層權重放大到 N(0, 0.02)，形變才不會是 0）')
shape = (96, 112, 96)
torch.manual_seed(0)
g = torch.Generator().manual_seed(1)
xs = torch.rand((1, 1) + shape, generator=g).to(dev)
xt = torch.rand((1, 1) + shape, generator=g).to(dev)


def big_flow_init(model):
    for name, prm in model.named_parameters():
        if ('flow' in name) and name.endswith('weight'):
            with torch.no_grad():
                prm.normal_(0, 0.02, generator=torch.Generator(device=prm.device).manual_seed(7))


# 2a. flow_mult（串接，每顆 VxmDense）
a = new.VxmCascade(shape, int_steps=7, int_downsize=1, n_cascades=2).to(dev)
big_flow_init(a)
b = new.VxmCascade(shape, int_steps=7, int_downsize=1, n_cascades=2, flow_mult=0.5).to(dev)
b.load_state_dict(a.state_dict())
with torch.no_grad():
    _, pa = a(xs, xt)
    _, pb = b(xs, xt)
r0 = float((pb[0] / pa[0])[pa[0].abs() > 1e-6].mean())
check(abs(r0 - 0.5) < 1e-5, 'flow_mult = 0.5：第一顆的速度場 / 原本 = %.6f（要 0.5；第二顆的輸入不同，不比）' % r0)
check(b.config['flow_mult'] == 0.5 and a.config['flow_mult'] == 1.0, 'flow_mult 有存進 config（讀檔時蓋得回來）')

# 2b. border（串接那一層）
c = new.VxmCascade(shape, int_steps=7, int_downsize=1, n_cascades=2, border=True).to(dev)
c.load_state_dict(a.state_dict())
with torch.no_grad():
    _, Ua = a(xs, xt, registration=True)
    _, Uc = c(xs, xt, registration=True)
    mag = float(Ua.abs().max())
idx = np.ogrid[tuple(slice(0, s) for s in shape)]
bd = np.minimum(idx[0], shape[0] - 1 - idx[0])
for i, s in zip(idx[1:], shape[1:]):
    bd = np.minimum(bd, np.minimum(i, s - 1 - i))
df = (Uc - Ua).abs().amax(dim=(0, 1)).cpu().numpy()
print('      位移最大 %.2f voxel；border 跟補 0 的差：離邊界 0～2 格最大 %.3g、3～9 格 %.3g、≥ 10 格 %.3g'
      % (mag, df[bd <= 2].max(), df[(bd >= 3) & (bd <= 9)].max(), df[bd >= 10].max()))
check(df[bd <= 2].max() > df[bd >= 10].max(), 'border 的影響集中在影像邊界')
check(type(c.transformer).__name__ == 'BorderTransformer' and type(a.transformer).__name__ == 'SpatialTransformer',
      'border = True 才換成 BorderTransformer；預設照舊')

# 2c. detach_compose（由粗到細）
pn = new.VxmPyramid(shape, int_steps=7, int_downsize=1).to(dev)
big_flow_init(pn)
pd = new.VxmPyramid(shape, int_steps=7, int_downsize=1, detach_compose=True).to(dev)
pd.load_state_dict(pn.state_dict())
outs = []
for m in (pn, pd):
    m.zero_grad()
    moved, pres = m(xs, xt)
    loss = ((moved - xt) ** 2).mean() + sum((p ** 2).mean() for p in pres)
    loss.backward()
    _, U = m(xs, xt, registration=True)
    outs.append((U.detach(), m.flows[2].weight.grad.clone(), m.flows[4].weight.grad.clone()))
du = float((outs[0][0] - outs[1][0]).abs().max())
g2 = float((outs[0][1] - outs[1][1]).abs().max() / outs[0][1].abs().max())
g4 = float((outs[0][2] - outs[1][2]).abs().max() / outs[0][2].abs().max())
check(du == 0.0, 'detach_compose：往前算的總位移完全一樣（最大差 %.3g）' % du)
check(g2 > 1e-4, 'detach_compose：梯度不同（1/4 層輸出卷積的梯度相對差 %.3g、1/16 層 %.3g）' % (g2, g4))

# 2d. return_levels
with torch.no_grad():
    moved, pres, (s_in, Us, warps) = pd(xs, xt, return_levels=True)
    _, U = pd(xs, xt, registration=True)
shapes = [tuple(u.shape[2:]) for u in Us]
check(len(Us) == pd.n + 1 and shapes[0] == tuple(s // 16 for s in shape) and shapes[-1] == shape,
      'VxmPyramid return_levels：%d 層，大小 %s' % (len(Us), shapes))
check(float((Us[-1] - U).abs().max()) == 0.0 and s_in is xs, '最後一層＝輸出的總位移、輸入影像就是 source')
cp = new.VxmCascade(shape, int_steps=7, int_downsize=1, n_cascades=2, stage='pyramid', flow_mult=0.5, border=True,
                    detach_compose=True).to(dev)
big_flow_init(cp)
with torch.no_grad():
    moved, pres, (s_in, Us, warps) = cp(xs, xt, return_levels=True)
    moved2, pres2 = cp(xs, xt)
    m1, U1 = cp.stages[0].fields(xs, xt, mult=0.5)
    s_ref = cp.transformer(xs, m1)
check(len(pres) == len(pres2) == 2 * (cp.stages[0].n + 1) and float((moved - moved2).abs().max()) == 0.0,
      '串接＋由粗到細 return_levels：平滑項 %d 項、輸出跟一般 forward 一樣' % len(pres))
check(float((s_in - s_ref).abs().max()) == 0.0 and len(Us) == cp.stages[-1].n + 1,
      '子層 loss 用的是最後一顆的輸入（第一顆 warp 過的影像）')
check(all(st.detach_compose for st in cp.stages), 'detach_compose 傳到每一顆')

print()
print('全部通過' if ok else '有沒通過的項目')
sys.exit(0 if ok else 1)
