"""
ASD 前處理包裝：FreeSurfer 產物 -> VoxelMorph 訓練用 npz

用法（在專案根目錄，venv 啟動後）：
    python ASD\\run_preprocess.py               # 正常跑（會先顯示切分並要你確認）
    python ASD\\run_preprocess.py --dry-run     # 只看切分，不動影像
    python ASD\\run_preprocess.py --yes         # 跳過確認，直接開跑
    python ASD\\run_preprocess.py --save-nii    # 額外輸出每顆的 .nii.gz（多佔約 1.4 GB）

中斷了直接重跑即可 —— preprocess_fs.py 預設 --skip-done，已完成的會略過。
預計耗時：167 顆 x 約 25 秒 = 70~90 分鐘（主要花在 ANTs Affine 配準）。

專案根目錄由本檔位置自動推出；用的 python 就是執行本檔的那一個。
"""

import os
import sys
import glob
import time
import argparse
import platform
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--dry-run', action='store_true', help='只印歸戶與切分，不動影像')
ap.add_argument('--yes', action='store_true', help='跳過確認')
ap.add_argument('--save-nii', action='store_true', help='額外輸出 .nii.gz（約 +1.4 GB）')
ap.add_argument('--out-dir', default=None, help='預設 ASD/ASD_preprocessed_v1')
args = ap.parse_args()

PY = sys.executable
SCRIPT = os.path.join(ROOT, 'ASD', 'preprocess_fs.py')
IMG_DIR = os.path.join(ROOT, 'ASD', 'ASD_data', 'norm')
SEG_DIR = os.path.join(ROOT, 'ASD', 'ASD_data', 'aseg')
ATLAS = os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.nii.gz')
OUT_DIR = args.out_dir or os.path.join(ROOT, 'ASD', 'ASD_preprocessed_v1')
SUBJ_LIST = os.path.join(ROOT, 'ASD', 'subjects_final.txt')
LOG_DIR = os.path.join(ROOT, 'log')
LOG_FILE = os.path.join(LOG_DIR, 'asd_preprocess.txt')
CMD_FILE = os.path.join(LOG_DIR, 'asd_preprocess_script.txt')
VERIFY = os.path.join(ROOT, 'ASD', 'verify_one_subject.py')

BAR = '=' * 69
print()
print(BAR)
print('  ASD 前處理')
print(BAR)
print()

# ── 起跑前檢查 ───────────────────────────────────────────────────────
print('[1/4] 起跑前檢查')
print('      專案根目錄 : %s' % ROOT)
print('      python     : %s' % PY)

ok = True
for p in (SCRIPT, ATLAS, SUBJ_LIST):
    if os.path.exists(p):
        print('      [v] %s' % p)
    else:
        print('      [X] 找不到：%s' % p)
        ok = False

for d in (IMG_DIR, SEG_DIR):
    if os.path.isdir(d):
        fs = glob.glob(os.path.join(d, '*.nii.gz'))
        mb = sum(os.path.getsize(f) for f in fs) / 1048576
        print('      [v] %s  (%d 個, %d MB)' % (d, len(fs), round(mb)))
        if len(fs) != 167:
            print('      [!] 預期 167 個，實際 %d 個' % len(fs))
    else:
        print('      [X] 找不到：%s' % d)
        ok = False

n_subj = 0
if os.path.exists(SUBJ_LIST):
    with open(SUBJ_LIST, encoding='utf-8-sig') as f:
        n_subj = len([l for l in f if l.strip() and not l.strip().startswith('#')])
    print('      [v] 受試者清單：%d 個 ID' % n_subj)

# 資料不該進 git
probe = glob.glob(os.path.join(IMG_DIR, '*.nii.gz'))
if probe:
    r = subprocess.run(['git', 'check-ignore', '-q', probe[0]], cwd=ROOT,
                       capture_output=True)
    if r.returncode == 0:
        print('      [v] .gitignore 有擋住影像資料')
    else:
        print('      [!] 警告：影像資料沒被 .gitignore 擋住，不要 commit！')

if not ok:
    print()
    print('  起跑前檢查未通過，已中止。')
    sys.exit(1)

os.makedirs(LOG_DIR, exist_ok=True)

# ── 共用參數 ─────────────────────────────────────────────────────────
#   --grouping none = 每個掃描各自成一人（使用者 2026-08-23 的決定）
#                     A013/A0131/A0132 與 A016_1/A016_2 都當不同人
#   --list-is-final = 混掃描檢查已通過，解除批次閘門
COMMON = [PY, SCRIPT,
          '--img-dir', IMG_DIR,
          '--seg-dir', SEG_DIR,
          '--atlas', ATLAS,
          '--out-dir', OUT_DIR,
          '--subject-list', SUBJ_LIST,
          '--grouping', 'none']

env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')

# ── 切分預覽 ─────────────────────────────────────────────────────────
print()
print('[2/4] 切分預覽（不動影像）')
print()
rc = subprocess.run(COMMON + ['--dry-run'], cwd=ROOT, env=env).returncode
if rc != 0:
    print('  dry-run 失敗，已中止。')
    sys.exit(1)

if args.dry_run:
    print()
    print('  --dry-run，到此結束。')
    sys.exit(0)

# ── 確認 ─────────────────────────────────────────────────────────────
if not args.yes:
    print()
    print('-' * 69)
    print('  即將處理 %d 顆，預計 70~90 分鐘。' % n_subj)
    print('  輸出到：%s' % OUT_DIR)
    print('  記錄檔：%s' % LOG_FILE)
    print('  （中斷後直接重跑即可續跑，已完成的會略過）')
    print('-' * 69)
    if input('  確定開始？(y/N) ').strip().lower() != 'y':
        print('  已取消。')
        sys.exit(0)

# ── 正式跑 ───────────────────────────────────────────────────────────
cmd = COMMON + ['--list-is-final']
if args.save_nii:
    cmd.append('--save-nii')

with open(CMD_FILE, 'w', encoding='utf-8') as f:
    f.write('# ASD 前處理指令記錄\n')
    f.write('# 執行時間: %s\n' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    f.write('# 機器: %s\n' % platform.node())
    f.write('# 由 ASD/run_preprocess.py 產生\n\n')
    f.write(' '.join('"%s"' % c if ' ' in c else c for c in cmd) + '\n')

print()
print('[3/4] 開始批次處理')
print('      指令已記錄到 %s' % CMD_FILE)
print()

t0 = time.time()
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
        print('\n  已中斷。直接重跑即可續跑（--skip-done 預設開啟）。')
        sys.exit(130)
    rc = proc.wait()

print()
print('      耗時 %.1f 分鐘' % ((time.time() - t0) / 60))

if rc != 0:
    print('  前處理回傳非零結束碼 (%d)，請看 %s' % (rc, LOG_FILE))
    sys.exit(rc)

# ── 跑完檢查 ─────────────────────────────────────────────────────────
print()
print('[4/4] 跑完檢查')
n_train = len(glob.glob(os.path.join(OUT_DIR, 'train', '*.npz')))
n_test = len(glob.glob(os.path.join(OUT_DIR, 'test', '*.npz')))
total = n_train + n_test
print('      train : %d 個 npz' % n_train)
print('      test  : %d 個 npz' % n_test)
print('      合計  : %d / %d' % (total, n_subj))
if total != n_subj:
    print('      [!] 數量對不上，請查 %s 裡的失敗記錄' % LOG_FILE)
else:
    print('      [v] 數量正確')

# 抽驗 3 顆（含左右翻轉檢查）
if os.path.exists(VERIFY) and n_train:
    import random
    print()
    print('      抽驗 3 顆：')
    for f in random.Random(0).sample(
            sorted(glob.glob(os.path.join(OUT_DIR, 'train', '*.npz'))),
            min(3, n_train)):
        r = subprocess.run([PY, VERIFY, '--npz', f], cwd=ROOT, env=env,
                           capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        line = [l.strip() for l in (r.stdout or '').splitlines() if '結果：' in l]
        tag = '[v]' if r.returncode == 0 else '[X]'
        print('        %s %-10s %s' % (tag, os.path.basename(f)[:-4],
                                       line[0] if line else ''))
        if r.returncode != 0:
            for l in (r.stdout or '').splitlines():
                print('            ' + l)

print()
print(BAR)
print('  完成 — 下一步：訓練')
print(BAR)
print()
print('    python ASD\\run_train.py --check-only     # 先檢查')
print('    python ASD\\run_train.py                  # 正式跑')
print()
