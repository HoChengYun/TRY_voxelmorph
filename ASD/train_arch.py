#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""改架構的訓練腳本（2026-10-06；CLAUDE.md 待辦 5）。網路都在 ASD/arch.py：
    --arch cascade：把 n 顆 VoxelMorph 串起來（第 1 步，RCN）
    --arch pyramid：由粗到細（第 2 步）—— 兩張影像各自抽特徵，解碼器每一層都出形變、先把移動影像的特徵拉過去再修
                    （串接的說明換成「每一層」：影像項只算最後，每一層積分前的速度場都罰平滑；只支援 --int-downsize 1）

跟 voxelmorph-code/scripts/torch/train.py 一樣的部分：
    參數名稱、每步從訓練資料隨機抽一位、scan-to-atlas（受試者 → atlas）、NCC／MSE、
    平滑項 Grad('l2', loss_mult=int_downsize) 算在積分前的形變場上、每個 epoch 開頭存一個 .pt、log 格式。
串接的部分照 RCN（Zhao et al., ICCV 2019）：
    - 每一顆有自己的權重，一起訓練（end-to-end）
    - 影像項只算最後搬好的影像：L_sim(atlas, 受試者∘U_n)
    - 每一顆的形變場都罰平滑：λ · Σ_k Grad(v_k)
log：  loss: 總計  (影像項, 平滑項)   —— 平滑項是 n 顆加起來；ASD/plot_loss_curve.py 讀得懂。
存出來的 .pt 帶 config['arch']，test_dice.py 等腳本透過 ASD/arch.py 的 load_model() 讀。

用法（一般透過 ASD/run_train.py --arch cascade 呼叫，不直接跑）：
    python ASD\\train_arch.py data\\mixed_preprocessed_v2\\train --atlas IXI\\atlas_mni152_09c_v3.npz ^
        --model-dir models\\mix_cascade --epochs 250 --gpu 0 --image-loss ncc --lambda 1.0 ^
        --int-steps 7 --int-downsize 1 --arch cascade --n-cascades 2
筆電試跑（縮小尺寸、只跑幾步、最後印顯存峰值）：再加 --crop 96 112 96 --max-steps 5
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import torch

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch import VxmCascade, VxmPyramid, load_model        # noqa: E402
import voxelmorph as vxm                         # noqa: E402

for _stream in (sys.stdout, sys.stderr):       # 直接跑、又導向檔案時也是 UTF-8（log 不要混編碼，見 CLAUDE.md）
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

parser = argparse.ArgumentParser()
parser.add_argument('datadir', help='train 資料夾（npz 的 key: vol）')
parser.add_argument('--atlas', required=True, help='atlas 影像 npz（key: vol）')
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
parser.add_argument('--arch', required=True, choices=['cascade', 'pyramid'])
parser.add_argument('--n-cascades', type=int, default=2, help='--arch cascade 時串幾顆')
parser.add_argument('--crop', type=int, nargs=3, help='測試用：只取中間這麼大一塊（每邊要能被 16 整除）')
parser.add_argument('--max-steps', type=int, default=0, help='測試用：總共只跑幾步就停，並印顯存峰值（0 = 不限）')
args = parser.parse_args()

# ── 資料 ─────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(args.datadir, '*.npz')))
assert files, '找不到訓練資料：%s' % args.datadir
with np.load(files[0]) as z:
    full_shape = z['vol'].shape
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
assert atlas_vol.shape == full_shape, '大小對不上：資料 %s、atlas %s' % (full_shape, atlas_vol.shape)

if args.crop:
    assert all(c % 16 == 0 and c <= s for c, s in zip(args.crop, full_shape)), \
        '--crop 每邊要能被 16 整除、而且不能比影像大：%s vs %s' % (args.crop, full_shape)
    box = tuple(slice((s - c) // 2, (s - c) // 2 + c) for c, s in zip(args.crop, full_shape))
    cut = lambda a: a[box]
    inshape = tuple(args.crop)
else:
    cut = lambda a: a
    inshape = full_shape


def sample():
    """跟 vxm.generators.volgen 一樣：每步從全部訓練資料裡隨機抽（可重複）。"""
    vols = []
    for i in np.random.randint(len(files), size=args.batch_size):
        with np.load(files[i]) as z:
            vols.append(cut(z['vol'].astype(np.float32)))
    return np.stack(vols)


# ── 裝置、模型 ───────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda'
torch.backends.cudnn.deterministic = not args.cudnn_nondet
os.makedirs(args.model_dir, exist_ok=True)

enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
if args.load_model:
    model = load_model(args.load_model, device)
    assert model.config.get('arch') == args.arch and \
        (args.arch != 'cascade' or model.config['n_cascades'] == args.n_cascades), \
        '--load-model 的架構跟這次的參數不一樣：%s' % model.config
elif args.arch == 'cascade':
    model = VxmCascade(inshape, nb_unet_features=[enc_nf, dec_nf], int_steps=args.int_steps,
                       int_downsize=args.int_downsize, n_cascades=args.n_cascades)
else:
    model = VxmPyramid(inshape, nb_unet_features=[enc_nf, dec_nf], int_steps=args.int_steps,
                       int_downsize=args.int_downsize)
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

atlas_t = torch.from_numpy(cut(atlas_vol))[None, None].to(device).repeat(args.batch_size, 1, 1, 1, 1)

n_par = sum(p.numel() for p in model.parameters())
desc = ('串 %d 顆 VoxelMorph（RCN）' % args.n_cascades if args.arch == 'cascade'
        else '由粗到細（兩張影像各自抽特徵、每一層都出形變）')
print('架構：%s，共 %d 個參數；影像項 %s，λ = %g（每一%s都罰）；int_steps %d、int_downsize %d；'
      '%d 位訓練資料；影像 %s%s'
      % (desc, n_par, args.image_loss, args.weight, '顆' if args.arch == 'cascade' else '層',
         args.int_steps, args.int_downsize, len(files),
         'x'.join(map(str, inshape)), '（--crop 測試用）' if args.crop else ''), flush=True)

# ── 訓練 ─────────────────────────────────────────────────────────────
done_steps = 0
for epoch in range(args.initial_epoch, args.epochs):
    model.save(os.path.join(args.model_dir, '%04d.pt' % epoch))
    for step in range(args.steps_per_epoch):
        t0 = time.time()
        source = torch.from_numpy(sample())[:, None].to(device)
        moved, pres = model(source, atlas_t)
        terms = [image_loss(atlas_t, moved),
                 args.weight * sum(grad_loss(None, p) for p in pres)]
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
            print('--max-steps %d 到了，停止（測試用）。顯存峰值：實際用到 %.2f GB、PyTorch 預留 %.2f GB'
                  % (args.max_steps, torch.cuda.max_memory_allocated() / 1024 ** 3,
                     torch.cuda.max_memory_reserved() / 1024 ** 3), flush=True)
            raise SystemExit(0)

model.save(os.path.join(args.model_dir, '%04d.pt' % args.epochs))
