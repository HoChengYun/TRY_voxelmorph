# -*- coding: utf-8 -*-
"""RCN 照原作訓練。網路與損失是 rcn_torch.py（原作 TF 程式的 PyTorch 翻譯）；這支對照原作的 train.py、
network/framework.py（資料增強、Adam）、data_util/brain.py（atlas 配準怎麼取資料）。

照原作（參數名稱、預設值都跟原作 train.py 一樣）：
  -b 底層網路 VTN／VoxelMorph（預設 VTN）、-n 串幾顆（預設 1，不含最前面的 affine）、-r 共用權重重複幾次（預設 1）
  --batch 4、--round 20000 步算一輪、--epochs 5 輪 = 10 萬步
  Adam（TF 預設 β1 0.9、β2 0.999、ε 1e-8），lr 1e-4，6 萬步後 ×1/2、8 萬步後 ×1/4（原作 learningRates[steps // round]）
  訓練時對 moving 加隨機 B-spline 形變（rcn_torch.augment；fixed 不動）
  資料：原作腦的實驗就是 atlas 配準（datasets/brain.json "atlas": "lpba/S01"），fixed = atlas、moving = 受試者；
       受試者每次全部用完就重新洗牌，batch 依序取 4 位（data_util/brain.py generate_atlas）
逼不得已改的（手冊 §31）：
  1. TF 1.4 → PyTorch（rcn_torch.py；verify_rcn_port.py 驗證跟原作逐項相同）
  2. 讀我們的 npz（vol 已經在 [0, 1]；原作 h5 存 0～255、讀進來 /255）；atlas = MNI152 2009c；影像 192×224×192（原作 128³）
  3. 原作 batch 4 = 4 張卡各算 1 對、梯度平均（utils.py MultiGPUs.average_gradients）；
     我們 1 張卡 → 4 對依序算、梯度累加後平均，同一件事
  4. 存檔：原作每小時存一次、只留最新 5 顆；改成每 --save-every 步存一顆 NNNNNN.pt（只有權重），
     之後用我們的 val Dice 挑（eval_repro.py）；另存 last.pt（含 Adam 狀態，-c last.pt 接著練）
  5. 不跑原作訓練中每 100 步的 validation（LPBA 的 Dice，只寫進 TensorBoard）
其他沒改的：資料讀取另開一條執行緒先讀下一位（只是快，順序、內容不變）。
log：train_log.txt 每一步一行（ASD/plot_loss_curve.py 讀得懂的格式），epoch = 每 --save-every 步一段：
  loss: 總（4 對平均）  (最後一顆的相似度 1 − r, 各顆平滑項的和, affine 的 det＋ortho)

用法（筆電試跑：2 步、batch 1）
  python ASD\\reproduce\\train_rcn.py -b VTN -n 1 --train-dir data\\mixed_preprocessed_v2\\train ^
      --atlas IXI\\atlas_mni152_09c_v3.npz --model-dir runs\\rcn_test --batch 1 --max-steps 2
"""
import os
import sys
import glob
import time
import json
import queue
import argparse
import threading

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('-b', '--base_network', type=str, default='VTN', choices=['VTN', 'VoxelMorph'])
ap.add_argument('-n', '--n_cascades', type=int, default=1, help='串幾顆（不含 affine），原作預設 1')
ap.add_argument('-r', '--rep', type=int, default=1, help='共用權重重複幾次，原作預設 1')
ap.add_argument('-g', '--gpu', type=str, default='0')
ap.add_argument('-c', '--checkpoint', type=str, default=None, help='接著練：last.pt')
ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--round', type=int, default=20000, help='幾步算一輪（換 lr 用）')
ap.add_argument('--epochs', type=float, default=5)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--net_args', type=str, default='', help='同原作，例如 "det_factor=0.1, reg_factor=1.0"')
ap.add_argument('--train-dir', required=True, help='受試者 npz（vol）')
ap.add_argument('--atlas', required=True, help='atlas npz（vol）')
ap.add_argument('--model-dir', required=True)
ap.add_argument('--save-every', type=int, default=5000)
ap.add_argument('--seed', type=int, default=None)
ap.add_argument('--max-steps', type=int, default=0, help='測試用：跑幾步就停，並印顯存峰值')
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcn_torch as R                          # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

files = sorted(glob.glob(os.path.join(args.train_dir, '*.npz')))
if not files:
    sys.exit('[X] %s 裡沒有 npz' % args.train_dir)
atlas = np.load(args.atlas)['vol'].astype(np.float32)
image_size = list(atlas.shape)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(args.model_dir, exist_ok=True)

net_args = eval('dict({})'.format(args.net_args))
model = R.RecursiveCascadedNetworks(args.base_network, args.n_cascades, image_size, rep=args.rep, **net_args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
steps = 0
if args.checkpoint:
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck['state_dict'])
    optimizer.load_state_dict(ck['optimizer'])
    steps = ck['steps']

# 原作 train.py
learningRates = [1e-4, 1e-4, 1e-4, 1e-4 / 2, 1e-4 / 4, 1e-4 / 8, 1e-4 / 16, 1e-4 / 32, 1e-4 / 64]


def get_lr(steps):
    m = args.lr / learningRates[0]
    return m * learningRates[steps // args.round]


def generate_atlas(sets):
    """data_util/brain.py generate_atlas(loop=True)：每一圈先洗牌，再依序給。（fixed 固定是 atlas，這裡只給受試者）"""
    sets = list(sets)
    while True:
        np.random.shuffle(sets)
        for d in sets:
            yield d


def loader(q):
    for f in generate_atlas(files):
        q.put((os.path.basename(f), np.load(f)['vol'].astype(np.float32)))


q = queue.Queue(maxsize=2 * args.batch)
threading.Thread(target=loader, args=(q,), daemon=True).start()
atlas_t = torch.from_numpy(atlas)[None, None].to(device)


def save(path, full=False):
    d = {'state_dict': model.state_dict(), 'steps': steps,
         'args': {'base_network': args.base_network, 'n_cascades': args.n_cascades, 'rep': args.rep,
                  'image_size': image_size, 'net_args': net_args}}
    if full:
        d['optimizer'] = optimizer.state_dict()
    torch.save(d, path)


log = open(os.path.join(args.model_dir, 'train_log.txt'), 'a', encoding='utf-8')


def say(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()


total = int(args.epochs * args.round)
say('指令：' + ' '.join(sys.argv))
say('RCN（照原作）：%s × %d 顆（＋affine），rep %d，共 %d 個參數；影像 %s；batch %d；%d 步（lr %g，%d、%d 步後各減半）；'
    '%d 位受試者 → atlas；net_args %s'
    % (args.base_network, args.n_cascades, args.rep, sum(p.numel() for p in model.parameters()),
       'x'.join(map(str, image_size)), args.batch, total, args.lr, 3 * args.round, 4 * args.round, len(files), net_args))
with open(os.path.join(args.model_dir, 'args.json'), 'w', encoding='utf-8') as fo:
    json.dump(vars(args), fo, ensure_ascii=False)

model.train()
while steps < total:
    lr = get_lr(steps)
    for g in optimizer.param_groups:
        g['lr'] = lr
    t0 = time.time()
    optimizer.zero_grad()
    logs = np.zeros(4)
    for _ in range(args.batch):
        _, vol = q.get()
        moving = torch.from_numpy(vol)[None, None].to(device)
        with torch.no_grad():
            moving = R.augment(moving)                 # 只在訓練時（原作 tflearn training mode）
        out = model(atlas_t, moving)                   # img1 = fixed = atlas、img2 = moving = 受試者
        (out['loss'] / args.batch).backward()
        st = out['stems']
        logs += [out['loss'].item(), st[-1]['raw_loss'].item(),
                 sum(s['reg_loss'].item() for s in st[1:] if 'reg_loss' in s), st[0]['loss'].item()]
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    steps += 1
    logs /= args.batch
    say('epoch: %04d  step: %d/%d  time: %.2f sec  loss: %.6f  (%.6f, %.6f, %.6f)  lr %.2e'
        % ((steps - 1) // args.save_every + 1, (steps - 1) % args.save_every + 1, args.save_every,
           time.time() - t0, logs[0], logs[1], logs[2], logs[3], lr))
    if steps % args.save_every == 0 or steps == total:
        save(os.path.join(args.model_dir, '%06d.pt' % steps))
        save(os.path.join(args.model_dir, 'last.pt'), full=True)
    if args.max_steps and steps >= args.max_steps:
        say('--max-steps %d 到了，停止（測試用）。顯存峰值：實際用到 %.2f GB、PyTorch 預留 %.2f GB'
            % (args.max_steps, torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30))
        save(os.path.join(args.model_dir, '%06d.pt' % steps))
        save(os.path.join(args.model_dir, 'last.pt'), full=True)
        sys.exit(0)
say('完成。')
