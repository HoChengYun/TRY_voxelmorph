"""
前處理包裝：FreeSurfer 產物 -> VoxelMorph 訓練用 npz

用法（在專案根目錄，venv 啟動後）：
    python ASD\\run_preprocess.py --dataset ASD   # 正常跑（會先顯示切分並要你確認）
    python ASD\\run_preprocess.py --dataset DGM --dry-run   # 只看切分，不動影像
    python ASD\\run_preprocess.py --dataset VNT --yes       # 跳過確認，直接開跑
    python ASD\\run_preprocess.py --save-nii      # 額外輸出每顆的 .nii.gz

資料位置：data/<資料集>_data/fs_for_vxm/{norm,aseg}
清單：    data/<資料集>_data/fs_stats/subjects.txt（FreeSurfer 端隨資料附的）
輸出：    data/<資料集>_preprocessed_v1/{train,test}

中斷了直接重跑即可 —— preprocess_fs.py 預設 --skip-done，已完成的會略過。
預計耗時：每顆約 25 秒（主要花在 ANTs Affine 配準）。

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

# Windows 主控台預設 cp950，印到 emoji 會 UnicodeEncodeError 直接中斷程式。
# 不要求使用者記得設 PYTHONIOENCODING —— 忘一次就白跑一輪。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--dry-run', action='store_true', help='只印歸戶與切分，不動影像')
ap.add_argument('--yes', action='store_true', help='跳過確認')
ap.add_argument('--save-nii', action='store_true', help='額外輸出 .nii.gz（約 +1.4 GB）')
ap.add_argument('--dataset', default='ASD',
                help='資料集名稱。原始資料在 data/<名稱>_data/{norm,aseg}，'
                     '輸出到 data/<名稱>_preprocessed_v1')
ap.add_argument('--n4', action='store_true',
                help='做 N4 偏場校正。FreeSurfer 的 norm.mgz 已經過 nu 校正所以不用；'
                     'tigerbx 的 _tbet 是原始強度（實測白質 CoV 15.9%）要開，'
                     '否則兩個 arm 的差異會混進偏場而不只是分割品質')
ap.add_argument('--out-dir', default=None, help='預設 data/<資料集>_preprocessed_v1')
ap.add_argument('--group-map', default=None,
                help='歸戶對照表 TSV（受試者<TAB>人）。預設抓 ASD/<資料集>_groups.txt，'
                     '有的話就用 —— 同一人的多次掃描必須整組在同一個 split')
ap.add_argument('--subject-list', default=None,
                help='預設 ASD/<資料集>_subjects_final.txt；ASD 用 ASD/subjects_final.txt')
args = ap.parse_args()

DS = args.dataset
PY = sys.executable
SCRIPT = os.path.join(ROOT, 'ASD', 'preprocess_fs.py')
DATA_ROOT = os.path.join(ROOT, 'data', DS + '_data')

# 資料夾長相會因為來源而異，全部試過去，第一個「影像與標籤都在」的就用。
#   FreeSurfer 端 2026-09-07 起：data/<名稱>_data/fs_for_vxm/{norm,aseg}
#   更早的版本    ：data/<名稱>_data/{norm,aseg}
#   tigerbx 端    ：data/<名稱>/{img,seg}
# 與其要求對方配合我們的命名，不如這邊多認幾種 —— 交接時少一個出錯的環節。
_BASES = [os.path.join(DATA_ROOT, 'fs_for_vxm'),
          DATA_ROOT,
          os.path.join(ROOT, 'data', DS)]
_PAIRS = [('norm', 'aseg'), ('img', 'seg')]
IMG_DIR = SEG_DIR = None
for _b in _BASES:
    for _i, _s in _PAIRS:
        if os.path.isdir(os.path.join(_b, _i)) and os.path.isdir(os.path.join(_b, _s)):
            IMG_DIR, SEG_DIR = os.path.join(_b, _i), os.path.join(_b, _s)
            break
    if IMG_DIR:
        break
if IMG_DIR is None:                       # 讓後面的起跑前檢查印出人看得懂的錯誤
    IMG_DIR = os.path.join(DATA_ROOT, 'fs_for_vxm', 'norm')
    SEG_DIR = os.path.join(DATA_ROOT, 'fs_for_vxm', 'aseg')

ATLAS = os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.nii.gz')
OUT_DIR = args.out_dir or os.path.join(ROOT, 'data', DS + '_preprocessed_v1')

# 清單優先用資料端隨附的 subjects.txt —— 那份跟影像檔是一起驗過的。
_CANDS = [os.path.join(DATA_ROOT, 'fs_stats', 'subjects.txt'),
          os.path.join(ROOT, 'data', DS, 'subjects.txt'),
          os.path.join(ROOT, 'ASD', DS + '_subjects_final.txt')]
if DS == 'ASD':
    _CANDS.append(os.path.join(ROOT, 'ASD', 'subjects_final.txt'))
SUBJ_LIST = args.subject_list or next((p for p in _CANDS if os.path.exists(p)), _CANDS[0])

# 歸戶對照表：沒明給就找 ASD/<資料集>_groups.txt，存在才用
_GM = args.group_map or os.path.join(ROOT, 'ASD', DS + '_groups.txt')
GROUP_MAP = _GM if os.path.exists(_GM) else None
LOG_DIR = os.path.join(ROOT, 'log')
LOG_FILE = os.path.join(LOG_DIR, '%s_preprocess.txt' % DS.lower())
CMD_FILE = os.path.join(LOG_DIR, '%s_preprocess_script.txt' % DS.lower())
VERIFY = os.path.join(ROOT, 'ASD', 'verify_one_subject.py')

BAR = '=' * 69
print()
print(BAR)
print('  %s 前處理' % DS)
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

n_subj = 0
ids = set()
if os.path.exists(SUBJ_LIST):
    with open(SUBJ_LIST, encoding='utf-8-sig') as f:
        ids = {l.strip() for l in f if l.strip() and not l.strip().startswith('#')}
    n_subj = len(ids)
    print('      [v] 受試者清單：%d 個 ID  (%s)' % (n_subj, os.path.relpath(SUBJ_LIST, ROOT)))

# 清單是唯一事實來源 —— 以前這裡寫死「預期 167 個」，換資料集就會誤報。
for d in (IMG_DIR, SEG_DIR):
    if os.path.isdir(d):
        fs = glob.glob(os.path.join(d, '*.nii.gz'))
        mb = sum(os.path.getsize(f) for f in fs) / 1048576
        print('      [v] %s  (%d 個, %d MB)' % (d, len(fs), round(mb)))
        if ids:
            have = {os.path.basename(p)[:-7] for p in fs}
            miss = sorted(ids - have)
            if miss:
                print('      [X] 清單有但檔案缺 %d 個：%s' % (len(miss), ', '.join(miss[:6])))
                ok = False
    else:
        print('      [X] 找不到：%s' % d)
        ok = False

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
#   --grouping none = 每個掃描各自成一人。
#     2026-09-07 起這是安全的：身分問題已由 FreeSurfer 端查 DICOM 檔頭解決，
#     同一人的重複掃描（A0132、YT13）與非受試者（A016_2 品管掃描）已在
#     fs_stats/subjects.txt 這份清單裡先排掉，清單內不再有同一人的兩筆。
#   --list-is-final = 混掃描檢查已通過，解除批次閘門
COMMON = [PY, SCRIPT,
          '--img-dir', IMG_DIR,
          '--seg-dir', SEG_DIR,
          '--atlas', ATLAS,
          '--out-dir', OUT_DIR,
          '--subject-list', SUBJ_LIST,
          '--grouping', 'none']
if GROUP_MAP:
    COMMON += ['--group-map', GROUP_MAP]
    print('      [v] 歸戶對照表：%s' % os.path.relpath(GROUP_MAP, ROOT))

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
    print('  即將處理 %d 顆，每顆約 25 秒 -> 預計 %d~%d 分鐘。'
          % (n_subj, n_subj*22//60, n_subj*32//60))
    print('  輸出到：%s' % OUT_DIR)
    print('  記錄檔：%s' % LOG_FILE)
    print('  （中斷後直接重跑即可續跑，已完成的會略過）')
    print('-' * 69)
    if input('  確定開始？(y/N) ').strip().lower() != 'y':
        print('  已取消。')
        sys.exit(0)

# ── 正式跑 ───────────────────────────────────────────────────────────
cmd = COMMON + ['--list-is-final']
if args.n4:
    cmd.append('--n4')
if args.save_nii:
    cmd.append('--save-nii')

with open(CMD_FILE, 'w', encoding='utf-8') as f:
    f.write('# %s 前處理指令記錄\n' % DS)
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
print('    python ASD\\run_train.py --dataset %s --check-only   # 先檢查' % DS)
print('    python ASD\\run_train.py --dataset %s                # 單獨訓練這一包' % DS)
print()
print('  要三包混合訓練的話：')
print('    python ASD\\make_mixed_set.py --sources ASD DGM VNT')
print('    python ASD\\run_train.py --dataset mixed --exp-name mix_exp1')
print()
