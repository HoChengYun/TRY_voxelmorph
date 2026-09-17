# -*- coding: utf-8 -*-
"""從已經前處理好的 train/ 再切一份 val/ 出來（test/ 完全不動）。

什麼時候用這支
--------------
要跟舊實驗在**同一批 test** 上比較的時候。`make_split.py` 是整批重抽，test 會換人；
這支只從 train 挖一塊當 val，test 一個檔案都不碰。`--undo` 可以整個搬回去。

依來源資料集分層抽樣（有 mixed_manifest.json 的話），每個掃描各自算一位受試者。

🔴 沒有歸戶（2026-09-16 使用者規定）：不做「是不是同一個人」的判定，也沒有 `--group-map`。

用法
    python ASD\make_val_split.py --prep-dir data\mixed_preprocessed_v2 --val-frac 0.10 --dry-run
    python ASD\make_val_split.py --prep-dir data\mixed_preprocessed_v2 --val-frac 0.10
    python ASD\make_val_split.py --prep-dir data\mixed_preprocessed_v2 --undo
"""
import os
import sys
import glob
import json
import random
import shutil
import argparse

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ap = argparse.ArgumentParser()
ap.add_argument('--prep-dir', required=True,
                help='前處理輸出資料夾，底下要有 train\\ 與 test\\')
ap.add_argument('--val-frac', type=float, default=0.10,
                help='從 train 切出來當 val 的比例（以人數計），預設 0.10')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--undo', action='store_true', help='把 val\\ 全部搬回 train\\')
ap.add_argument('--force', action='store_true', help='val\\ 已存在也重切')
ap.add_argument('--dry-run', action='store_true', help='只印會搬哪些，不動檔案')
args = ap.parse_args()

PREP = os.path.abspath(args.prep_dir)
TRAIN = os.path.join(PREP, 'train')
VAL = os.path.join(PREP, 'val')
TEST = os.path.join(PREP, 'test')
MANIFEST = os.path.join(PREP, 'mixed_manifest.json')
RECORD = os.path.join(PREP, 'val_split.json')

if not os.path.isdir(TRAIN):
    sys.exit('[X] 找不到 %s' % TRAIN)


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding='utf-8') as f:
            return json.load(f)
    return None


def save_manifest(m):
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


# ── --undo：搬回去 ────────────────────────────────────────────────────
if args.undo:
    files = sorted(glob.glob(os.path.join(VAL, '*.npz')))
    if not files:
        sys.exit('[X] %s 底下沒有 npz，沒有東西可以搬回去' % VAL)
    print('把 %d 個檔案從 val\\ 搬回 train\\' % len(files))
    if args.dry_run:
        sys.exit(0)
    for p in files:
        shutil.move(p, os.path.join(TRAIN, os.path.basename(p)))
    os.rmdir(VAL)
    m = load_manifest()
    if m:
        for b in m['members']:
            if m['members'][b]['split'] == 'val':
                m['members'][b]['split'] = 'train'
        save_manifest(m)
    if os.path.exists(RECORD):
        os.remove(RECORD)
    print('  完成：train %d / test %d'
          % (len(glob.glob(os.path.join(TRAIN, '*.npz'))),
             len(glob.glob(os.path.join(TEST, '*.npz')))))
    sys.exit(0)

if os.path.isdir(VAL) and glob.glob(os.path.join(VAL, '*.npz')) and not args.force:
    sys.exit('[X] %s 已經有 npz 了。要重切先 --undo，或加 --force。' % VAL)

train_files = sorted(glob.glob(os.path.join(TRAIN, '*.npz')))
if not train_files:
    sys.exit('[X] %s 底下沒有 npz' % TRAIN)
subjects = [os.path.basename(p)[:-4] for p in train_files]

manifest = load_manifest()
ds_of = {}
for s in subjects:
    if manifest and (s + '.npz') in manifest['members']:
        ds_of[s] = manifest['members'][s + '.npz']['dataset']
    else:
        ds_of[s] = '(單一來源)'


by_ds = {}
for s in subjects:
    by_ds.setdefault(ds_of[s], []).append(s)

# ── 抽樣（依來源分層，每個掃描各自算一位）────────────────────────────
rng = random.Random(args.seed)
val_subjects = []
print('train 現有 %d 個掃描' % len(subjects))
print('分層抽樣（val_frac=%.2f，seed=%d）：' % (args.val_frac, args.seed))
for ds in sorted(by_ds):
    ids = sorted(by_ds[ds])
    rng.shuffle(ids)
    k = max(1, int(round(len(ids) * args.val_frac)))
    val_subjects += ids[:k]
    print('  %-14s %3d 筆 → val %2d 筆' % (ds, len(ids), k))

val_subjects.sort()
print('\nval：%d 筆（%.1f%%）' % (len(val_subjects), 100.0 * len(val_subjects) / len(subjects)))
print('train 剩：%d 筆' % (len(subjects) - len(val_subjects)))
print('test  不動：%d 筆' % len(glob.glob(os.path.join(TEST, '*.npz'))))

if args.dry_run:
    print('\n[dry-run] 沒有動任何檔案。前 10 個會搬的：%s' % val_subjects[:10])
    sys.exit(0)

# ── 搬檔 ─────────────────────────────────────────────────────────────
os.makedirs(VAL, exist_ok=True)
for s in val_subjects:
    shutil.move(os.path.join(TRAIN, s + '.npz'), os.path.join(VAL, s + '.npz'))

if manifest:
    for s in val_subjects:
        manifest['members'][s + '.npz']['split'] = 'val'
    save_manifest(manifest)

with open(RECORD, 'w', encoding='utf-8') as f:
    json.dump({
        'seed': args.seed,
        'val_frac': args.val_frac,
        'grouping': 'none',
        'val_subjects': val_subjects,
        'n_train_after': len(subjects) - len(val_subjects),
        'n_test': len(glob.glob(os.path.join(TEST, '*.npz'))),
    }, f, ensure_ascii=False, indent=2)

print('\n  已搬 %d 個 npz 到 %s' % (len(val_subjects), VAL))
print('  紀錄：%s（--undo 可以整個還原）' % RECORD)
print('\n  接著：')
print('    訓練      python ASD\\run_train.py --train-dir %s --exp-name <實驗名>'
      % os.path.join(args.prep_dir, 'train'))
print('    挑 epoch  python ASD\\test_dice.py --model-dir models\\<實驗名> --test-dir %s'
      % os.path.join(args.prep_dir, 'val'))
print('    最後一次  python ASD\\test_dice.py --model models\\<實驗名>\\<最佳>.pt --test-dir %s'
      % os.path.join(args.prep_dir, 'test'))
