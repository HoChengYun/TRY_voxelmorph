# -*- coding: utf-8 -*-
"""DWP（Jian et al.，期刊版，γ = 0 非監督）照原作程式訓練：rethink-reg @97175ff 的 tasks/brainmri/train.py，只換資料。

原作程式一個字都不改，這裡只 import（--orig-repo）；要在 venv_dwp 跑（原作的套件版本，requirements_dwp.txt）。
照原作：設定檔 configs/vxm_dualpyd_cfg.py（README 表上的 DWP）→ 模型 VXMPlus_DualWarpPy、RegistrationHead（最細到 1/2、
  再放大）、每一步用原作 run_iter.py 的 run_iter（雙向、左右翻轉 50%、1/16、1/8、1/4 各層也算相似度＋平滑、混合精度 fp16）、
  局部 NCC（窗格 9）、λ 0.5、Adam amsgrad lr 1e-4、每輪 lr × 0.996、100 輪、seed 2023（README）、cudnn.deterministic。
逼不得已改的（手冊 §31）：
  1. 資料：原作 LUMIR 500 顆，batch 2 = 隨機兩顆互相配（前一顆 source、後一顆 target）；
     我們每一步 = [受試者, atlas]（受試者 → atlas），受試者每輪洗牌、每位一次（原作也是每輪每張一次）→ 每輪 418 步
  2. 影像大小 160×192×224 → 192×224×192（設定檔裡每個 image_size 都換）。兩邊都是第一軸左右（LUMIR 轉成 LIA、
     我們 MNI152 RAS），原作沿第一軸翻轉 = 左右翻轉，意思相同
  3. 不跑原作的 val_iter（LUMIR val 的標籤、只寫 wandb）；每輪存一顆，之後用我們的 val Dice 挑 epoch（eval_repro.py）
  4. 不連 wandb：原作每一步 wandb.log 的數字改寫進 train_log.txt（ASD/plot_loss_curve.py 讀得懂）和 train_log.jsonl（全部欄位）
  5. 存檔名：原作 "%04d.pth" % epoch 從 0 起算（0050.pth = 跑完 51 輪）→ 改成跑完幾輪（0001.pth = 跑完 1 輪）；只存權重（同原作）
log 一行：loss: 總（含各層）  (最細那層的相似度 0.5·(fwd + bck), λ × 最細那層的平滑)

用法（筆電試跑：3 位受試者 × 2 輪 = 6 步）
  D:\\MyHome\\MRI\\orig_repos\\venv_dwp\\Scripts\\python.exe ASD\\reproduce\\train_dwp.py data\\mixed_preprocessed_v2\\train ^
      --atlas IXI\\atlas_mni152_09c_v3.npz --model-dir runs\\dwp_test --max-subjects 3 --max-epochs 2
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('train_dir', help='受試者 npz（vol）')
ap.add_argument('--atlas', required=True, help='atlas npz（vol）')
ap.add_argument('--model-dir', required=True)
ap.add_argument('--orig-repo', default=r'D:\MyHome\MRI\orig_repos\rethink-reg')
ap.add_argument('--config', default=None, help='預設 <orig-repo>\\tasks\\brainmri\\configs\\vxm_dualpyd_cfg.py')
ap.add_argument('--random-seed', '-seed', type=int, default=2023)
ap.add_argument('--max-epochs', type=int, default=None, help='預設照設定檔（100）')
ap.add_argument('--gpu', default='0')
ap.add_argument('--max-subjects', type=int, default=0, help='測試用：只用前幾位受試者')
ap.add_argument('--nondeterministic', action='store_true',
                help='不設 cudnn.deterministic（原作有設）。只影響速度與逐位元可重現，方法不變；'
                     '筆電實測反傳 9.9 → 1.8 秒')
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['WANDB_MODE'] = 'disabled'
TASK = os.path.join(args.orig_repo, 'tasks', 'brainmri')
sys.path.insert(0, args.orig_repo)
sys.path.insert(1, TASK)

import torch                                    # noqa: E402
import wandb                                    # noqa: E402
from mmengine import Config                     # noqa: E402
from models import build_loss, build_metrics, build_registration_head, build_flow_estimator   # noqa: E402
from run_iter import run_iter as train_iter     # noqa: E402  原作每一步的訓練
from utils import set_seed                      # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

torch.backends.cudnn.deterministic = not args.nondeterministic       # 原作 train.py 設 True


class ScanToAtlas(torch.utils.data.Dataset):
    """每一筆 = {'image': [受試者, atlas]}，形狀 (2, 1, X, Y, Z)：原作 run_iter 取 image[:1] 當 source、image[1:] 當 target。"""

    def __init__(self, files, atlas):
        self.files = files
        self.atlas = torch.from_numpy(atlas)[None]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        vol = torch.from_numpy(np.load(self.files[i])['vol'].astype(np.float32))[None]
        return {'image': torch.stack([vol, self.atlas])}


def main():
    set_seed(args.random_seed)
    cfg = Config.fromfile(args.config or os.path.join(TASK, 'configs', 'vxm_dualpyd_cfg.py'))
    cfg.update(dict(random_seed=args.random_seed))

    files = sorted(glob.glob(os.path.join(args.train_dir, '*.npz')))
    if args.max_subjects:
        files = files[:args.max_subjects]
    if not files:
        sys.exit('[X] %s 裡沒有 npz' % args.train_dir)
    atlas = np.load(args.atlas)['vol'].astype(np.float32)
    size = list(atlas.shape)
    # ── 逼不得已改的 ──
    cfg.image_size = size
    cfg.registration_cfg.image_size = size
    cfg.model_cfg.config.decoder_cfg.image_size = size
    cfg.gradicon_loss_cfg.image_size = size        # 權重 0，原作照樣會建；大小跟著換
    cfg.out_path = args.model_dir
    cfg.model_dir = ''
    cfg.save_interval = 1
    if args.max_epochs:
        cfg.max_epochs = args.max_epochs
    # ─────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    cfg.dump(os.path.join(args.model_dir, 'train_configs.py'))      # 同原作：留一份實際用的設定
    cfg.update(dict(amp_dtype=getattr(torch, cfg.amp_dtype)))

    model = build_flow_estimator(cfg.model_cfg)
    model.to(cfg.device)

    cfg.update(dict(pyramid=cfg.get('scale_pyramid', None) is not None))
    reg_head = []
    if cfg.pyramid:
        for scale in cfg.scale_pyramid:
            reg_head.append(build_registration_head(dict(type='DownSizeRegistrationHead', image_size=cfg.image_size,
                                                         scale=scale, interp_mode='bilinear')).to(cfg.device))
    reg_head.append(build_registration_head(cfg.registration_cfg).to(cfg.device))

    dataloader = torch.utils.data.DataLoader(ScanToAtlas(files, atlas), batch_size=None, shuffle=True, num_workers=2)
    len_trainset = len(dataloader)

    loss_funcs = dict(sim=build_loss(cfg.sim_loss_cfg).to(cfg.device), dice=build_loss(cfg.dice_loss_cfg),
                      reg=build_loss(cfg.reg_loss_cfg).to(cfg.device),
                      gradicon=build_loss(cfg.gradicon_loss_cfg).to(cfg.device),
                      jacdet=build_loss(cfg.np_jacdet_loss_cfg))
    metric_funcs = dict(jacdet=build_metrics(dict(type='sdlogjac')))     # run_iter 只用到這個（每一步記 log|J|）

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    log = open(os.path.join(args.model_dir, 'train_log.txt'), 'a', encoding='utf-8')
    jl = open(os.path.join(args.model_dir, 'train_log.jsonl'), 'a', encoding='utf-8')

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    last = [time.time()]

    def wandb_log(d, *a, **k):
        """原作每一步呼叫 wandb.log({'train': logging_dict})，改寫進我們的 log。"""
        v = d.get('train')
        if v is None:
            return
        now = time.time()
        it = int(v['iter'])
        sim = 0.5 * (float(v['fwd_sim']) + float(v['bck_sim'])) * cfg.sim_loss_cfg.weight
        say('epoch: %04d  step: %d/%d  time: %.2f sec  loss: %.6f  (%.6f, %.6f)  lr %.3e'
            % (it // len_trainset + 1, it % len_trainset + 1, len_trainset, now - last[0], float(v['total_loss']),
               sim, cfg.reg_loss_cfg.weight * float(v['reg']), optimizer.param_groups[0]['lr']))
        jl.write(json.dumps({k2: float(x) for k2, x in v.items()}) + '\n')
        jl.flush()
        last[0] = now

    wandb.log = wandb_log

    say('指令：' + ' '.join(sys.argv))
    say('DWP（照原作 %s）：%s，%d 個參數；影像 %s；%d 位受試者 → atlas，每輪 %d 步、%d 輪；seed %d；cudnn.deterministic %s'
        % (os.path.basename(args.config or 'vxm_dualpyd_cfg.py'), cfg.model_cfg.type,
           sum(p.numel() for p in model.parameters()), 'x'.join(map(str, size)), len(files), len_trainset,
           cfg.max_epochs, args.random_seed, torch.backends.cudnn.deterministic))
    for epoch in range(cfg.start_epoch, cfg.max_epochs):
        model.train()
        last[0] = time.time()
        train_iter(cfg, model, reg_head, dataloader, loss_funcs, metric_funcs, scaler, optimizer,
                   current_iter=len_trainset * epoch, phase='train')
        lr_scheduler.step()
        if (epoch + 1) % cfg.save_interval == 0 or epoch + 1 == cfg.max_epochs:
            torch.save(model.state_dict(), os.path.join(args.model_dir, '%04d.pth' % (epoch + 1)))
    say('完成。顯存峰值：實際用到 %.2f GB、PyTorch 預留 %.2f GB'
        % (torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30))


if __name__ == '__main__':
    main()
