# -*- coding: utf-8 -*-
"""把一份已經前處理好的資料重新切成 train / val / test 三段。

跟 make_val_split.py 的差別
---------------------------
`make_val_split.py` 只從 train 挖一塊當 val，**test 不動**（要跟舊實驗比時用這支）。
這一支是**整批重抽**：把 train/ val/ test/ 底下的 npz 全部倒回來，照比例重新分。
切分會改變，所以跟舊實驗的數字不能再直接比 —— 這是使用這支的代價。

保證
----
① 以「人」為單位：同一個人的多次掃描整組在同一邊（`--group-map`，格式 <受試者><TAB><人>）
② 依來源資料集分層：各包都照比例出人，不會整個 val 都來自同一個 cohort
③ 切完自動複驗有沒有人橫跨兩邊，有就直接中止

用法
    python ASD\\make_split.py --prep-dir data\\mixed_preprocessed_v2 \\
        --val-frac 0.10 --test-frac 0.10 --group-map ASD\\mixed_v2_groups.txt --dry-run
    python ASD\\make_split.py --prep-dir data\\mixed_preprocessed_v2 \\
        --val-frac 0.10 --test-frac 0.10 --group-map ASD\\mixed_v2_groups.txt
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
ap.add_argument('--group-map', default=None,
                help='歸戶表 TSV：<受試者><TAB><人>，同一人整組一起走')
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
print('收集到 %d 個掃描' % len(subjects))

manifest = None
if os.path.exists(MANIFEST):
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
ds_of = {s: (manifest['members'][s + '.npz']['dataset']
             if manifest and (s + '.npz') in manifest['members'] else '(單一來源)')
         for s in subjects}

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

persons = {}
for s in subjects:
    persons.setdefault(group_of.get(s, s), []).append(s)
print('歸戶後 %d 人（歸戶表 %d 筆）' % (len(persons), len(group_of)))

by_ds = {}
for pid, ss in persons.items():
    by_ds.setdefault(ds_of[sorted(ss)[0]], []).append(pid)

# ── 抽樣 ─────────────────────────────────────────────────────────────
rng = random.Random(args.seed)
assign = {}
print('\n分層抽樣（seed=%d，val %.0f%% / test %.0f%%）：'
      % (args.seed, args.val_frac * 100, args.test_frac * 100))
print('  %-14s %5s %6s %5s %6s' % ('來源', '人數', 'train', 'val', 'test'))
for ds in sorted(by_ds):
    pool_ids = sorted(by_ds[ds])
    rng.shuffle(pool_ids)
    n = len(pool_ids)
    n_te = max(1, int(round(n * args.test_frac)))
    n_va = max(1, int(round(n * args.val_frac)))
    for pid in pool_ids[:n_te]:
        assign[pid] = 'test'
    for pid in pool_ids[n_te:n_te + n_va]:
        assign[pid] = 'val'
    for pid in pool_ids[n_te + n_va:]:
        assign[pid] = 'train'
    print('  %-14s %5d %6d %5d %6d' % (ds, n, n - n_te - n_va, n_va, n_te))

target = {s: assign[pid] for pid, ss in persons.items() for s in ss}
cnt = {sp: sum(1 for v in target.values() if v == sp) for sp in SPLITS}
tot = len(target)
print('\n掃描數：train %d（%.1f%%）/ val %d（%.1f%%）/ test %d（%.1f%%）'
      % (cnt['train'], 100.0 * cnt['train'] / tot, cnt['val'], 100.0 * cnt['val'] / tot,
         cnt['test'], 100.0 * cnt['test'] / tot))

# ── 複驗：同一個人不可以橫跨 ─────────────────────────────────────────
leak = [(pid, ss) for pid, ss in persons.items() if len({target[s] for s in ss}) > 1]
if leak:
    sys.exit('[X] 這些人橫跨兩邊，切分有誤：%s' % leak[:5])
print('[v] 沒有人橫跨 train/val/test')

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
        'group_map': args.group_map,
        'n_person': len(persons),
        'counts': cnt,
        'split_of': target,
    }, f, ensure_ascii=False, indent=2)

for f_ in ('val_split.json',):
    p = os.path.join(PREP, f_)
    if os.path.exists(p):
        os.remove(p)          # 舊的 val 切分紀錄已經失效

print('\n  切分紀錄 -> %s' % os.path.join(args.prep_dir, 'split.json'))
print('  接著：python ASD\\run_train.py --train-dir %s --exp-name <實驗名>'
      % os.path.join(args.prep_dir, 'train'))
