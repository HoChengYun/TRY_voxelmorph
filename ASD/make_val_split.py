# -*- coding: utf-8 -*-
"""從已經前處理好的 train/ 再切一份 val/ 出來（test/ 完全不動）。

為什麼要這支
------------
到 mix_exp1 / tiger_exp1 為止，「最佳 epoch」都是拿 test 的 Dice 曲線挑的
（`test_dice.py --model-dir --test-dir ...\\test` → dice_curve.csv → 取最高那個 epoch）。
那等於用 test 做模型選擇，報出來的 test Dice 會偏樂觀，嚴格說已經不是 held-out。

正確做法是三份：
    train  訓練
    val    挑 epoch、挑超參數（λ、int-steps…）——想看幾次都可以
    test   最後只跑一次，報出來的數字才是誠實的

這支不重跑前處理，直接把 train/ 裡的一部分**移動**到 val/，所以幾秒鐘就好，
而且 `--undo` 可以整個搬回去。

不會造成 leakage 的兩個前提
---------------------------
① 以「人」為單位移動：同一個人的多次掃描不能一半 train 一半 val。
   歸戶表用 `--group-map`（格式同 preprocess_fs.py：<受試者><TAB><人>），
   例如 ASD\\DGM_groups.txt。沒列到的 ID 各自成一人。
② test/ 一個檔案都不碰 —— 這樣新舊實驗仍然可以在同一批 test 上直接比。

分層
----
有 mixed_manifest.json 的話，依「來源資料集」分層抽樣（各包都照比例出人），
val 才不會整包來自同一個 cohort。沒有 manifest 就當成單一來源。

用法
    python ASD\\make_val_split.py --prep-dir data\\mixed_preprocessed_v2 --val-frac 0.10 \\
        --group-map ASD\\DGM_groups.txt --dry-run
    python ASD\\make_val_split.py --prep-dir data\\mixed_preprocessed_v2 --val-frac 0.10 \\
        --group-map ASD\\DGM_groups.txt
    python ASD\\make_val_split.py --prep-dir data\\mixed_preprocessed_v2 --undo
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
ap.add_argument('--group-map', default=None,
                help='歸戶表 TSV：<受試者><TAB><人>，同一人整組一起走')
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

# ── 歸戶 ─────────────────────────────────────────────────────────────
group_of = {}
if args.group_map:
    with open(args.group_map, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                sys.exit('[X] 歸戶表這行不是 <受試者><TAB><人>：%s' % line)
            group_of[parts[0].strip()] = parts[1].strip()

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

persons = {}
for s in subjects:
    persons.setdefault(group_of.get(s, s), []).append(s)

# 同一個人的掃描可能散在兩包裡（例如同一批人被重新收錄成新的資料集）。
# 分層時歸到第一個成員所屬的那包，並印出來 —— 這種人本身就是重要資訊。
cross = []
for pid, ss in sorted(persons.items()):
    if len({ds_of[s] for s in ss}) > 1:
        cross.append((pid, sorted(ss)))
if cross:
    print('[i] 有 %d 位的掃描橫跨多個資料集，分層歸到第一個成員那包：' % len(cross))
    for pid, ss in cross:
        print('      %-14s %s' % (pid, '  '.join('%s(%s)' % (x, ds_of[x]) for x in ss)))
    print()

by_ds = {}
for pid, ss in persons.items():
    by_ds.setdefault(ds_of[sorted(ss)[0]], []).append(pid)

# ── 抽樣（各資料集內部各自抽，四捨五入，至少 1 人）────────────────────
rng = random.Random(args.seed)
val_persons = []
print('train 現有 %d 個掃描 / %d 人' % (len(subjects), len(persons)))
print('分層抽樣（val_frac=%.2f，seed=%d）：' % (args.val_frac, args.seed))
for ds in sorted(by_ds):
    pool = sorted(by_ds[ds])
    rng.shuffle(pool)
    k = max(1, int(round(len(pool) * args.val_frac)))
    val_persons += pool[:k]
    print('  %-14s %3d 人 → val %2d 人' % (ds, len(pool), k))

val_subjects = sorted(s for pid in val_persons for s in persons[pid])
print('\nval：%d 人 / %d 個掃描（%.1f%%）'
      % (len(val_persons), len(val_subjects), 100.0 * len(val_subjects) / len(subjects)))
print('train 剩：%d 個掃描' % (len(subjects) - len(val_subjects)))
print('test  不動：%d 個掃描' % len(glob.glob(os.path.join(TEST, '*.npz'))))

multi = [pid for pid in val_persons if len(persons[pid]) > 1]
if multi:
    print('\n[i] val 裡有 %d 位是多次掃描，整組一起搬：' % len(multi))
    for pid in multi:
        print('      %s -> %s' % (pid, persons[pid]))

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
        'group_map': args.group_map,
        'val_persons': sorted(val_persons),
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
