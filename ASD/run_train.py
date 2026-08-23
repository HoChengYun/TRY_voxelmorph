"""
ASD 訓練包裝：VoxelMorph scan-to-atlas

用法（在專案根目錄，venv 啟動後）：
    python ASD\\run_train.py                       # 預設 ncc + lambda 1.0, 250 epochs
    python ASD\\run_train.py --check-only          # 只做起跑前檢查，不訓練
    python ASD\\run_train.py --resume              # 從最後一個 .pt 續跑
    python ASD\\run_train.py --exp-name asd_exp2 --image-loss mse --lambda 0.01
    python ASD\\run_train.py --epochs 100          # 先跑短的看看

專案根目錄由本檔位置自動推出，換一台機器 clone 到別的路徑也不用改。
用的 python 就是執行本檔的那一個（sys.executable），所以只要 venv 有啟動就一定對。

🔴 lambda 的尺度取決於 image-loss：
     ncc -> 論文（TMI 2019 Fig.7）最佳約 1~2
            train.py 預設的 0.01 是為 mse 設的，搭 ncc 等於幾乎沒有正則化
     mse -> 論文最佳約 0.01~0.02
"""

import os
import sys
import glob
import time
import argparse
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--exp-name', default='asd_exp1')
ap.add_argument('--image-loss', default='ncc', choices=['ncc', 'mse'])
ap.add_argument('--lambda', dest='weight', type=float, default=1.0)
ap.add_argument('--epochs', type=int, default=250)
ap.add_argument('--steps-per-epoch', type=int, default=100)
ap.add_argument('--gpu', default='0')
ap.add_argument('--data-dir', default=None, help='預設 ASD/ASD_preprocessed_v1/train')
ap.add_argument('--atlas', default=None, help='預設 IXI/atlas_mni152_09c_v3.npz')
ap.add_argument('--resume', action='store_true', help='從最後一個 .pt 續跑')
ap.add_argument('--check-only', action='store_true', help='只做起跑前檢查')
ap.add_argument('--yes', action='store_true', help='跳過確認')
args = ap.parse_args()

PY = sys.executable                    # 就是目前這個 venv 的 python
TRAIN = os.path.join(ROOT, 'voxelmorph-code', 'scripts', 'torch', 'train.py')
DATA = args.data_dir or os.path.join(ROOT, 'ASD', 'ASD_preprocessed_v1', 'train')
TEST = os.path.join(ROOT, 'ASD', 'ASD_preprocessed_v1', 'test')
ATLAS = args.atlas or os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz')
MODEL_DIR = os.path.join(ROOT, 'models', args.exp_name)
LOG_DIR = os.path.join(ROOT, 'log')
LOG_FILE = os.path.join(LOG_DIR, args.exp_name + '.txt')
CMD_FILE = os.path.join(LOG_DIR, args.exp_name + '_script.txt')

BAR = '=' * 69
print()
print(BAR)
print('  ASD 訓練：%s' % args.exp_name)
print(BAR)
print()

# ── 起跑前檢查 ───────────────────────────────────────────────────────
print('[1/3] 起跑前檢查')
print('      專案根目錄 : %s' % ROOT)
print('      python     : %s' % PY)

ok = True
for p in (TRAIN, ATLAS):
    if os.path.exists(p):
        print('      [v] %s' % p)
    else:
        print('      [X] 找不到：%s' % p)
        ok = False

for d, tag in ((DATA, 'train'), (TEST, 'test')):
    if os.path.isdir(d):
        n = len(glob.glob(os.path.join(d, '*.npz')))
        print('      [v] %s  (%d 個 npz)' % (d, n))
        if n == 0:
            print('      [X] 資料夾是空的')
            ok = False
    else:
        print('      [X] 找不到：%s' % d)
        print('          前處理過的 npz 要從另一台機器搬過來（約 1.28 GB）；')
        print('          它不在 git 裡（.gitignore 有擋）。')
        ok = False

# GPU
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 1024 ** 3
        print('      [v] GPU: %s  (%.1f GB VRAM)' % (torch.cuda.get_device_name(0), vram))
        if vram < 7.5:
            print('      [!] VRAM < 7.5 GB，192x224x192 可能會 OOM')
    else:
        print('      [X] 沒有偵測到 CUDA GPU')
        print('          NCC loss 的 sum_filt 寫死在 cuda 上（losses.py:28），沒 GPU 跑不了')
        ok = False
    print('      [v] torch %s' % torch.__version__)
except ImportError:
    print('      [X] 這個環境沒有安裝 torch')
    ok = False

print('      [i] atlas: %s  <- 必須跟前處理用的同一版' % os.path.basename(ATLAS))

if not ok:
    print()
    print('  起跑前檢查未通過，已中止。')
    sys.exit(1)

os.makedirs(LOG_DIR, exist_ok=True)

# ── 續跑判斷 ─────────────────────────────────────────────────────────
initial_epoch = 0
existing = sorted(glob.glob(os.path.join(MODEL_DIR, '*.pt')))
if existing:
    last = int(os.path.basename(existing[-1])[:-3])
    if args.resume:
        initial_epoch = last
        print('      [i] --resume：從 epoch %d 續跑' % last)
    else:
        print()
        print('  [!] %s 已經有 %d 個 .pt（最後是 %s）'
              % (MODEL_DIR, len(existing), os.path.basename(existing[-1])))
        print('      直接跑會從 epoch 0 覆蓋。要續跑請加 --resume，或換一個 --exp-name。')
        if not args.yes:
            if input('      仍要從頭開始？(y/N) ').strip().lower() != 'y':
                print('  已取消。')
                sys.exit(0)
elif args.resume:
    print('      [!] --resume 但 %s 裡沒有 .pt，將從 epoch 0 開始' % MODEL_DIR)

# ── 參數摘要 ─────────────────────────────────────────────────────────
hint = '1~2' if args.image_loss == 'ncc' else '0.01~0.02'
remain = args.epochs - initial_epoch
print()
print('[2/3] 訓練設定')
print('      實驗名稱    : %s' % args.exp_name)
print('      image-loss  : %s' % args.image_loss)
print('      lambda      : %g   <- %s 建議 %s' % (args.weight, args.image_loss, hint))
print('      epochs      : %d (從 %d 開始，還要跑 %d)' % (args.epochs, initial_epoch, remain))
print('      steps/epoch : %d' % args.steps_per_epoch)
print('      訓練資料    : %s' % DATA)
print('      模型輸出    : %s' % MODEL_DIR)
print('      記錄檔      : %s' % LOG_FILE)
print()
print('      預估 : 每步約 1.5 秒 -> %.1f 小時（實際看 GPU）'
      % (remain * args.steps_per_epoch * 1.5 / 3600))
print('      磁碟 : 每 epoch 存一個 .pt（1.16 MB）-> 約 %d MB' % round(args.epochs * 1.16))

if args.check_only:
    print()
    print('  --check-only，到此結束。')
    sys.exit(0)

# ── 組指令 ───────────────────────────────────────────────────────────
cmd = [PY, TRAIN, DATA,
       '--atlas', ATLAS,
       '--model-dir', MODEL_DIR,
       '--epochs', str(args.epochs),
       '--steps-per-epoch', str(args.steps_per_epoch),
       '--gpu', args.gpu,
       '--image-loss', args.image_loss,
       '--lambda', str(args.weight)]
if initial_epoch > 0:
    cmd += ['--initial-epoch', str(initial_epoch),
            '--load-model', os.path.join(MODEL_DIR, '%04d.pt' % initial_epoch)]

# 專案慣例：每次跑都要留指令記錄，之後才查得出當初的參數
import platform
with open(CMD_FILE, 'w', encoding='utf-8') as f:
    f.write('# %s 訓練指令記錄\n' % args.exp_name)
    f.write('# 執行時間: %s\n' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    f.write('# 機器: %s\n' % platform.node())
    f.write('# 資料: ASD_preprocessed_v1 (train 150 / test 17, grouping=none, 167 subjects)\n')
    f.write('# 由 ASD/run_train.py 產生\n\n')
    f.write(' '.join('"%s"' % c if ' ' in c else c for c in cmd) + '\n')

print()
print('[3/3] 開始訓練')
print('      指令已記錄到 %s' % CMD_FILE)
print('      另開視窗看進度： python ASD\\tail.py %s' % LOG_FILE)
print()

t0 = time.time()
env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
with open(LOG_FILE, 'w', encoding='utf-8') as log:
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding='utf-8', errors='replace', bufsize=1)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
    except KeyboardInterrupt:
        proc.terminate()
        print('\n  已中斷。用 --resume 可以從最後一個 .pt 續跑。')
        sys.exit(130)
    rc = proc.wait()

hrs = (time.time() - t0) / 3600
print()
print('      耗時 %.2f 小時' % hrs)

if rc != 0:
    print('  訓練回傳非零結束碼 (%d)，請看 %s' % (rc, LOG_FILE))
    print('  中斷的話可以用 --resume 續跑。')
    sys.exit(rc)

n_pt = len(glob.glob(os.path.join(MODEL_DIR, '*.pt')))
print('      產出 %d 個 .pt' % n_pt)

print()
print(BAR)
print('  訓練完成 — 下一步：挑最佳 epoch')
print(BAR)
print()
print('    %s voxelmorph-code\\scripts\\torch\\batch_test_ixi.py \\' % os.path.basename(PY))
print('        --model-dir %s \\' % MODEL_DIR)
print('        --atlas %s \\' % ATLAS)
print('        --test-dir %s \\' % TEST)
print('        --out-dir %s --step 1 --gpu %s' % (MODEL_DIR, args.gpu))
print()
print('  會產生 epoch_curve.csv / .png，用來挑 epoch。')
print('  ⚠️ 目前沒有 val set，epoch 是依 test 曲線挑的 —— 發表時方法學要明講。')
print()
print('  要搬回主機的只有：最佳的幾個 .pt、epoch_curve.csv/.png、log。')
print('  不用把 %d 個 .pt 全搬。' % n_pt)
print()
