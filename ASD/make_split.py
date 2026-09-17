# -*- coding: utf-8 -*-
"""把一份已經前處理好的資料重新切成 train / val / test 三段。

做法
----
把 train\\ val\\ test\\ 底下的 npz 全部倒回來，**依來源資料集分層**後按比例重新分配。
每一個掃描各自算一位受試者，固定 seed，重跑結果一樣。

🔴 沒有歸戶（2026-09-16 使用者規定）
-----------------------------------
不做「這兩筆是不是同一個人」的判定，也沒有 `--group-map`。就照拿到的資料分。
（唯一允許的同一人判定是「逐張影像完全相同」，那種情況應該在資料階段就處理掉。）

用法
    python ASD\\make_split.py --prep-dir data\\mixed_preprocessed_v2 --val-frac 0.10 --test-frac 0.10 --dry-run
    python ASD\\make_split.py --prep-dir data\\mixed_preprocessed_v2 --val-frac 0.10 --test-frac 0.10
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
ap.add_argument('--prep-dir', required=True)
ap.add_argument('--val-frac', type=float, default=0.10)
ap.add_argument('--test-frac', type=float, default=0.10)
ap.add_argument('--include', nargs='*', default=['_excluded'],
                help='額外納入的子資料夾（預設把 _excluded\\ 裡的也收回來重切）')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--dry-run', action='store_true')
args = ap.parse_args()

PREP = os.path.abspath(args.prep_dir)
SPLITS = ('train', 'val', 'test')
MANIFEST = os.path.join(PREP, 'mixed_manifest.json')

if args.val_frac + args.test_frac >= 1.0:
    sys.exit('[X] val + test 比例要小於 1')

# ── 收集所有 npz ─────────────────────────────────────────────────────
pool = {}
for sub in list(SPLITS) + list(args.include):
    for p in glob.glob(os.path.join(PREP, sub, '*.npz')):
        pool[os.path.basename(p)[:-4]] = p
if not pool:
    sys.exit('[X] %s 底下沒有 npz' % PREP)
subjects = sorted(pool)
print('收集到 %d 個掃描（每一個各自算一位受試者）' % len(subjects))

manifest = None
if os.path.exists(MANIFEST):
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
ds_of = {s: (manifest['members'][s + '.npz']['dataset']
             if manifest and (s + '.npz') in manifest['members'] else '(單一來源)')
         for s in subjects}

by_ds = {}
for s in subjects:
    by_ds.setdefault(ds_of[s], []).append(s)

# ── 抽樣（各來源內部各自抽）──────────────────────────────────────────
rng = random.Random(args.seed)
target = {}
print('\n分層抽樣（seed=%d，val %.0f%% / test %.0f%%）：'
      % (args.seed, args.val_frac * 100, args.test_frac * 100))
print('  %-14s %5s %6s %5s %6s' % ('來源', '筆數', 'train', 'val', 'test'))
for ds in sorted(by_ds):
    ids = sorted(by_ds[ds])
    rng.shuffle(ids)
    n = len(ids)
    n_te = max(1, int(round(n * args.test_frac)))
    n_va = max(1, int(round(n * args.val_frac)))
    for s in ids[:n_te]:
        target[s] = 'test'
    for s in ids[n_te:n_te + n_va]:
        target[s] = 'val'
    for s in ids[n_te + n_va:]:
        target[s] = 'train'
    print('  %-14s %5d %6d %5d %6d' % (ds, n, n - n_te - n_va, n_va, n_te))

cnt = {sp: sum(1 for v in target.values() if v == sp) for sp in SPLITS}
tot = len(target)
print('\n合計：train %d（%.1f%%）/ val %d（%.1f%%）/ test %d（%.1f%%）'
      % (cnt['train'], 100.0 * cnt['train'] / tot, cnt['val'], 100.0 * cnt['val'] / tot,
         cnt['test'], 100.0 * cnt['test'] / tot))

moved = sum(1 for s in subjects
            if os.path.basename(os.path.dirname(pool[s])) != target[s])
print('需要搬動 %d 個檔案' % moved)

if args.dry_run:
    print('\n[dry-run] 沒有動任何檔案。')
    sys.exit(0)

# ── 搬檔 ─────────────────────────────────────────────────────────────
for sp in SPLITS:
    os.makedirs(os.path.join(PREP, sp), exist_ok=True)
for s in subjects:
    dst = os.path.join(PREP, target[s], s + '.npz')
    if os.path.abspath(pool[s]) != os.path.abspath(dst):
        shutil.move(pool[s], dst)
for sub in args.include:
    d = os.path.join(PREP, sub)
    if os.path.isdir(d) and not glob.glob(os.path.join(d, '*.npz')):
        for junk in glob.glob(os.path.join(d, '*')):
            os.remove(junk)
        os.rmdir(d)
        print('  %s\\ 已清空移除（裡面的都收回來重切了）' % sub)

if manifest:
    for s in subjects:
        manifest['members'].setdefault(s + '.npz', {'dataset': ds_of[s]})
        manifest['members'][s + '.npz']['split'] = target[s]
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

with open(os.path.join(PREP, 'split.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'seed': args.seed,
        'val_frac': args.val_frac,
        'test_frac': args.test_frac,
        'grouping': 'none',
        'counts': cnt,
        'split_of': target,
    }, f, ensure_ascii=False, indent=2)

old_record = os.path.join(PREP, 'val_split.json')
if os.path.exists(old_record):
    os.remove(old_record)          # 舊的 val 切分紀錄已經失效

print('\n  切分紀錄 -> %s' % os.path.join(args.prep_dir, 'split.json'))
print('  接著：python ASD\\run_train.py --train-dir %s --exp-name <實驗名>'
      % os.path.join(args.prep_dir, 'train'))
