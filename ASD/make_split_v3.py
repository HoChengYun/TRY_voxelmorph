# -*- coding: utf-8 -*-
"""v3（三種去頭骨比較，1323 位）的切分：前處理之前先切好，三組共用同一份。

做法跟 make_split.py 一模一樣（依資料包分層、各包內部用同一個 seed 洗牌、
test 取前 round(n*test_frac) 位、val 取接下來 round(n*val_frac) 位、其餘 train；
每個掃描各自算一位受試者，不歸戶），差別只在：
  - make_split.py 是把「已經前處理好的 npz」重新分配；這支在前處理之前就切，
    只讀名單（subjects.tsv 的 subject、dataset 兩欄），不動任何影像。
  - 輸出 preprocess_fs.py 的 split.json 格式（有 split_of），三組前處理都用
    --split-from 指到這個檔，test 就保證是同一群人。

用法
    python ASD\\make_split_v3.py --tsv data\\skullstrip_compare\\orig\\subjects.tsv ^
        --out data\\skullstrip_compare\\split_v3.json --dry-run
    python ASD\\make_split_v3.py --tsv data\\skullstrip_compare\\orig\\subjects.tsv ^
        --out data\\skullstrip_compare\\split_v3.json
"""
import os
import sys
import csv
import json
import random
import argparse

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ap = argparse.ArgumentParser()
ap.add_argument('--tsv', required=True, help='名單 TSV，要有 subject、dataset 兩欄')
ap.add_argument('--out', required=True, help='輸出的 split.json')
ap.add_argument('--check-dirs', nargs='*', default=[],
                help='再確認這些資料夾裡 <subject>.nii.gz 都在（三組影像資料夾）')
ap.add_argument('--val-frac', type=float, default=0.10)
ap.add_argument('--test-frac', type=float, default=0.10)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--dry-run', action='store_true')
args = ap.parse_args()

SPLITS = ('train', 'val', 'test')
if args.val_frac + args.test_frac >= 1.0:
    sys.exit('[X] val + test 比例要小於 1')

with open(args.tsv, encoding='utf-8') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
ds_of = {r['subject']: r['dataset'] for r in rows}
if len(ds_of) != len(rows):
    sys.exit('[X] 名單裡有重複的 subject')
subjects = sorted(ds_of)
print('名單 %d 位（每一個掃描各自算一位受試者，不歸戶）' % len(subjects))

for d in args.check_dirs:
    miss = [s for s in subjects if not os.path.exists(os.path.join(d, s + '.nii.gz'))]
    print('  %s：%s' % (d, '全部都在' if not miss else '缺 %d 位，例如 %s' % (len(miss), miss[:5])))
    if miss:
        sys.exit('[X] 影像資料夾跟名單對不上')

by_ds = {}
for s in subjects:
    by_ds.setdefault(ds_of[s], []).append(s)

# ── 抽樣（各來源內部各自抽；跟 make_split.py 同一段邏輯）──────────────
rng = random.Random(args.seed)
target = {}
print('\n分層抽樣（seed=%d，val %.0f%% / test %.0f%%）：'
      % (args.seed, args.val_frac * 100, args.test_frac * 100))
print('  %-16s %5s %6s %5s %6s' % ('來源', '筆數', 'train', 'val', 'test'))
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
    print('  %-16s %5d %6d %5d %6d' % (ds, n, n - n_te - n_va, n_va, n_te))

cnt = {sp: sum(1 for v in target.values() if v == sp) for sp in SPLITS}
tot = len(target)
print('\n合計：train %d（%.1f%%）/ val %d（%.1f%%）/ test %d（%.1f%%）'
      % (cnt['train'], 100.0 * cnt['train'] / tot, cnt['val'], 100.0 * cnt['val'] / tot,
         cnt['test'], 100.0 * cnt['test'] / tot))

if args.dry_run:
    print('\n[dry-run] 沒有寫檔。')
    sys.exit(0)

if os.path.exists(args.out):
    sys.exit('[X] %s 已經存在；切分只做一次，要重切請自己先改名或刪掉舊檔' % args.out)

out = {
    'note': 'v3 三種去頭骨比較的切分（make_split_v3.py）；三組前處理都用 --split-from 指到這個檔',
    'source_tsv': os.path.relpath(os.path.abspath(args.tsv)),
    'seed': args.seed,
    'val_frac': args.val_frac,
    'test_frac': args.test_frac,
    'counts': cnt,
    'split_of': {s: target[s] for s in subjects},
    'dataset_of': {s: ds_of[s] for s in subjects},
}
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('\n切分紀錄 -> %s' % args.out)
