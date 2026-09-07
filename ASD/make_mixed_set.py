# -*- coding: utf-8 -*-
"""把多個資料集的前處理結果併成一份，給混合訓練用。

`train.py` 是 `glob(datadir/*.npz)`，只吃**單一扁平資料夾**，
所以混合訓練必須先把各資料集的 npz 放到同一個目錄。

用硬連結而不是複製 —— 1.3 GB 的資料集併三份就是 4 GB，沒有意義。
硬連結要求來源與目的在**同一個磁碟區**（NTFS 支援）；跨磁碟時自動退回複製。

各來源自己的 train/test 切分**原封不動保留**：train 併 train、test 併 test。
所以只要每個資料集自己的切分是受試者層級的，合併後仍然沒有 leakage。

用法
    python ASD\\make_mixed_set.py --sources ASD NEWDATA
    python ASD\\make_mixed_set.py --sources ASD NEWDATA --out mixed --force

    # 之後訓練
    python ASD\\run_train.py --dataset mixed --exp-name mix_exp1
"""
import os
import sys
import glob
import json
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')

ap = argparse.ArgumentParser()
ap.add_argument('--sources', nargs='+', required=True,
                help='資料集名稱，對應 data/<名稱>_preprocessed_v1')
ap.add_argument('--out', default='mixed', help='輸出到 data/<out>_preprocessed_v1')
ap.add_argument('--prefix', action='store_true',
                help='檔名前面加 <資料集>_ ；預設只在偵測到撞名時才需要')
ap.add_argument('--force', action='store_true', help='輸出目錄已存在就清掉重建')
ap.add_argument('--dry-run', action='store_true', help='只印會做什麼')
args = ap.parse_args()

OUT = os.path.join(DATA, args.out + '_preprocessed_v1')

# ── 檢查來源 ──────────────────────────────────────────────────────────
srcs = []
for name in args.sources:
    d = os.path.join(DATA, name + '_preprocessed_v1')
    if not os.path.isdir(d):
        sys.exit('[X] 找不到 %s\n    先跑：python ASD\\run_preprocess.py --dataset %s' % (d, name))
    n_tr = len(glob.glob(os.path.join(d, 'train', '*.npz')))
    n_te = len(glob.glob(os.path.join(d, 'test', '*.npz')))
    if n_tr == 0 and n_te == 0:
        sys.exit('[X] %s 底下沒有 npz' % d)
    srcs.append((name, d, n_tr, n_te))
    print('  [v] %-12s train %3d / test %3d' % (name, n_tr, n_te))

if len(srcs) < 2:
    print('\n[!] 只給了一個來源，合併沒有意義。')

# ── 撞名檢查：不同資料集若有同名受試者，直接覆蓋會安靜地少一顆 ────────
seen, clash = {}, []
for name, d, _, _ in srcs:
    for split in ('train', 'test'):
        for p in glob.glob(os.path.join(d, split, '*.npz')):
            b = os.path.basename(p)
            if b in seen and seen[b] != name:
                clash.append((b, seen[b], name))
            seen[b] = name
if clash:
    print('\n[X] 偵測到 %d 個撞名的檔案，例如：' % len(clash))
    for b, a1, a2 in clash[:5]:
        print('      %s  同時出現在 %s 和 %s' % (b, a1, a2))
    if not args.prefix:
        sys.exit('\n    加 --prefix 讓檔名帶資料集名稱，或先自行改名。')
    print('    已加 --prefix，會改名成 <資料集>_<原檔名>。')

use_prefix = args.prefix or bool(clash)

# ── 建立輸出 ──────────────────────────────────────────────────────────
if os.path.exists(OUT):
    if not args.force:
        sys.exit('\n[X] %s 已存在。加 --force 清掉重建。' % OUT)
    if not args.dry_run:
        shutil.rmtree(OUT)
    print('\n  已清掉舊的 %s' % OUT)

n_link = n_copy = 0
manifest = {'sources': args.sources, 'prefix': use_prefix, 'members': {}}

for split in ('train', 'test'):
    dst_dir = os.path.join(OUT, split)
    if not args.dry_run:
        os.makedirs(dst_dir, exist_ok=True)
    for name, d, _, _ in srcs:
        for p in sorted(glob.glob(os.path.join(d, split, '*.npz'))):
            b = os.path.basename(p)
            out_name = ('%s_%s' % (name, b)) if use_prefix else b
            dst = os.path.join(dst_dir, out_name)
            manifest['members'][out_name] = {'dataset': name, 'split': split}
            if args.dry_run:
                continue
            try:
                os.link(p, dst)          # 硬連結：不佔額外空間
                n_link += 1
            except OSError:
                shutil.copy2(p, dst)     # 跨磁碟時退回複製
                n_copy += 1

print()
if args.dry_run:
    tr = sum(1 for v in manifest['members'].values() if v['split'] == 'train')
    te = sum(1 for v in manifest['members'].values() if v['split'] == 'test')
    print('[dry-run] 會產生 %s：train %d / test %d' % (OUT, tr, te))
    sys.exit(0)

with open(os.path.join(OUT, 'mixed_manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

tr = len(glob.glob(os.path.join(OUT, 'train', '*.npz')))
te = len(glob.glob(os.path.join(OUT, 'test', '*.npz')))
exp_tr = sum(s[2] for s in srcs)
exp_te = sum(s[3] for s in srcs)

print('  硬連結 %d 個，複製 %d 個' % (n_link, n_copy))
print('  %s' % OUT)
print('    train %d（預期 %d）%s' % (tr, exp_tr, '' if tr == exp_tr else '  [X] 對不上！'))
print('    test  %d（預期 %d）%s' % (te, exp_te, '' if te == exp_te else '  [X] 對不上！'))
print('    mixed_manifest.json —— 記錄每個檔案來自哪個資料集')
if tr != exp_tr or te != exp_te:
    sys.exit(1)
print()
print('  接著跑：python ASD\\run_train.py --dataset %s --exp-name <實驗名>' % args.out)
