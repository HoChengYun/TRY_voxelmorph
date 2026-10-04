#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""訓練時也用 FreeSurfer 標籤的 VoxelMorph（scan-to-atlas，半監督）。2026-10-04。

照論文（Balakrishnan et al., TMI 2019）式 (9)、(10)：
    L = L_sim(atlas, 受試者∘φ) + λ·L_smooth + γ·L_seg
    L_seg = −(1/K)·Σ_k Dice_k(atlas 的標籤, 受試者的標籤∘φ)，K = 30 個評估結構（voxelmorph-code/data/labels.npz）
受試者的標籤轉成 30 個通道的 one-hot，用跟影像**同一個形變場**（積分完、全尺寸）線性內插搬過去
（變成 0～1 的軟標籤，才算得出梯度），再跟 atlas 的 one-hot 算 soft Dice。
**測試時只用影像、不需要標籤**：存出來的 .pt 跟一般模型一樣，test_dice.py、visualize_*.py 照常能用。

其他全部跟 voxelmorph-code/scripts/torch/train.py 一樣：同樣的參數名稱、同樣的抽樣方式（每步隨機抽一位）、
同樣的平滑項（Grad('l2', loss_mult=int_downsize)，算在積分前的形變場上）、同樣用 VxmDense.save 存 .pt。
log 多一項：  loss: 總計  (影像項, 平滑項, 標籤項)   —— ASD/plot_loss_curve.py 讀得懂三項。

🔴 γ 的尺度：論文的 γ 是搭 MSE、λ = 0.02。我們用 NCC、λ = 1，損失量級差約 50 倍，γ 要跟 λ 一起放大：
     論文 γ = 0.01 → 我們 0.5；論文 γ = 0.1 → 我們 5（γ / λ 的比例跟論文一樣）
   直接照抄 0.01 的話，標籤項小到幾乎沒有作用（跟 CLAUDE.md「λ 的尺度取決於 image-loss」同一件事）。

⚠️ 不支援 --bidir、--multichannel（這條線用不到）。

用法（一般透過 ASD/run_train.py --seg-weight 呼叫，不直接跑）：
    python ASD\\train_semisup.py data\\mixed_preprocessed_v2\\train --atlas IXI\\atlas_mni152_09c_v3.npz ^
        --atlas-seg IXI\\atlas_mni152_09c_v3_seg.npz --model-dir models\\mix_exp8 --epochs 250 --gpu 0 ^
        --image-loss ncc --lambda 1.0 --int-steps 7 --int-downsize 1 --seg-weight 0.5
"""
import os
import glob
import time
import argparse
import numpy as np
import torch

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('datadir', help='train 資料夾（npz 要有 vol 和 seg）')
parser.add_argument('--atlas', required=True, help='atlas 影像 npz（key: vol）')
parser.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'),
                    help='atlas 標籤 npz（key: seg），要跟 --atlas 同一版、同一套標籤工具')
parser.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'),
                    help='算標籤項用哪些結構（預設：Dice 評估用的同一組 30 個）')
parser.add_argument('--model-dir', default='models')
parser.add_argument('--gpu', default='0')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--steps-per-epoch', type=int, default=100)
parser.add_argument('--load-model', help='從這顆 .pt 接著訓練（--resume 用）')
parser.add_argument('--initial-epoch', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--cudnn-nondet', action='store_true')
parser.add_argument('--enc', type=int, nargs='+')
parser.add_argument('--dec', type=int, nargs='+')
parser.add_argument('--int-steps', type=int, default=7)
parser.add_argument('--int-downsize', type=int, default=2)
parser.add_argument('--image-loss', default='mse')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01)
parser.add_argument('--seg-weight', type=float, required=True, help='標籤項的權重 γ（NCC 時建議 0.5 或 5，見檔頭）')
parser.add_argument('--max-steps', type=int, default=0, help='測試用：總共只跑幾步就停（0 = 不限）')
args = parser.parse_args()

# ── 資料 ─────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(args.datadir, '*.npz')))
assert files, '找不到訓練資料：%s' % args.datadir
with np.load(files[0]) as z:
    assert 'seg' in z.files, '%s 沒有 seg，不能用標籤訓練（要 preprocess_fs.py 產生的 npz）' % files[0]
    inshape = z['vol'].shape
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
labels = np.load(args.labels)['labels'].astype(np.int32)
assert atlas_vol.shape == inshape == atlas_seg.shape, \
    '大小對不上：資料 %s、atlas %s、atlas 標籤 %s' % (inshape, atlas_vol.shape, atlas_seg.shape)
missing = [int(l) for l in labels if not (atlas_seg == l).any()]
assert not missing, 'atlas 標籤裡沒有這些結構：%s（--atlas-seg 是不是用錯了？）' % missing


def sample():
    """跟 vxm.generators.volgen 一樣：每步從全部訓練資料裡隨機抽（可重複）。"""
    vols, segs = [], []
    for i in np.random.randint(len(files), size=args.batch_size):
        with np.load(files[i]) as z:
            vols.append(z['vol'].astype(np.float32))
            segs.append(z['seg'].astype(np.int32))
    return np.stack(vols), np.stack(segs)


# ── 裝置、模型 ───────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda'
torch.backends.cudnn.deterministic = not args.cudnn_nondet
os.makedirs(args.model_dir, exist_ok=True)

enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
if args.load_model:
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    model = vxm.networks.VxmDense(inshape=inshape, nb_unet_features=[enc_nf, dec_nf], bidir=False,
                                  int_steps=args.int_steps, int_downsize=args.int_downsize)
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.image_loss == 'ncc':
    image_loss = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss = vxm.losses.MSE().loss
else:
    raise ValueError('--image-loss 只能是 mse 或 ncc，收到 "%s"' % args.image_loss)
grad_loss = vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss      # 同 train.py:139
dice_loss = vxm.losses.Dice().loss                                         # = −平均 Dice（論文式 9）

lab_t = torch.from_numpy(labels).to(device).view(1, -1, 1, 1, 1)
onehot = lambda seg: (seg.unsqueeze(1) == lab_t).float()                   # (B, D, H, W) -> (B, K, D, H, W)
atlas_t = torch.from_numpy(atlas_vol)[None, None].to(device).repeat(args.batch_size, 1, 1, 1, 1)
atlas_oh = onehot(torch.from_numpy(atlas_seg)[None].to(device)).repeat(args.batch_size, 1, 1, 1, 1)


def forward(source, target):
    """跟 VxmDense.forward 同一套計算，但同時拿到：
    積分前的形變場（給平滑項，跟 train.py 一樣）、積分完全尺寸的形變場（拿來搬標籤，跟搬影像用同一個）。"""
    x = model.unet_model(torch.cat([source, target], dim=1))
    pos_flow = model.flow(x)
    if model.resize:
        pos_flow = model.resize(pos_flow)
    preint_flow = pos_flow
    if model.integrate:
        pos_flow = model.integrate(pos_flow)
        if model.fullsize:
            pos_flow = model.fullsize(pos_flow)
    return model.transformer(source, pos_flow), preint_flow, pos_flow


print('標籤項：%d 個結構，γ = %g；影像項 %s，λ = %g；int_steps %d、int_downsize %d；%d 位訓練資料'
      % (len(labels), args.seg_weight, args.image_loss, args.weight, args.int_steps, args.int_downsize, len(files)),
      flush=True)

# ── 訓練 ─────────────────────────────────────────────────────────────
done_steps = 0
for epoch in range(args.initial_epoch, args.epochs):
    model.save(os.path.join(args.model_dir, '%04d.pt' % epoch))
    for step in range(args.steps_per_epoch):
        t0 = time.time()
        vol, seg = sample()
        source = torch.from_numpy(vol)[:, None].to(device)
        moved, preint_flow, pos_flow = forward(source, atlas_t)
        if done_steps == 0:
            # 第一步自我檢查：自己算的形變場、搬移結果要跟 VxmDense.forward 一樣，不然標籤搬的跟影像不是同一個形變
            with torch.no_grad():
                ref_moved, ref_flow = model(source, atlas_t, registration=True)
            assert torch.allclose(ref_flow, pos_flow, atol=1e-4) and torch.allclose(ref_moved, moved, atol=1e-4), \
                '自己算的形變場跟 VxmDense.forward 對不上'
            print('[v] 第一步檢查：形變場、搬移結果跟 VxmDense.forward 一致', flush=True)
            del ref_moved, ref_flow
        moved_oh = model.transformer(onehot(torch.from_numpy(seg).to(device)), pos_flow)

        terms = [image_loss(atlas_t, moved),
                 args.weight * grad_loss(None, preint_flow),
                 args.seg_weight * dice_loss(atlas_oh, moved_oh)]
        loss = sum(terms)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('  '.join(('epoch: %04d' % (epoch + 1),
                         ('step: %d/%d' % (step + 1, args.steps_per_epoch)).ljust(14),
                         'time: %.2f sec' % (time.time() - t0),
                         'loss: %.6f  (%s)' % (loss.item(), ', '.join('%.6f' % t.item() for t in terms)))),
              flush=True)
        done_steps += 1
        if args.max_steps and done_steps >= args.max_steps:
            model.save(os.path.join(args.model_dir, '%04d.pt' % (epoch + 1)))
            print('--max-steps %d 到了，停止（測試用）' % args.max_steps, flush=True)
            raise SystemExit(0)

model.save(os.path.join(args.model_dir, '%04d.pt' % args.epochs))
