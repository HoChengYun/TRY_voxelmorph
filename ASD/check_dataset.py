# -*- coding: utf-8 -*-
"""前處理資料的完整性檢查 —— 搬到別台機器之後跑這個。

為什麼需要
----------
npz 被截斷或複製到一半，`np.load` 不一定會報錯（zip 結構可能還讀得到目錄），
但訓練時只會表現成「Dice 比預期低一點」，看不出是資料壞了。
一份 2.2 GB 的資料靠隨身碟搬，這種事不是假想。

兩種模式
--------
    # 來源機器：產生指紋檔（跟著資料一起複製過去）
    python ASD\\check_dataset.py --write-manifest --datasets ASD DGM VNT

    # 目標機器：比對
    python ASD\\check_dataset.py --check --datasets ASD DGM VNT

沒有 manifest 也能用 —— 只跑內容檢查（可讀、有 vol+seg、shape/dtype/值域正確、
標籤是整數且落在合理集合、與 subjects.txt 對帳）。那擋得住大部分的壞檔，
只是擋不住「內容被改成另一份合法的資料」。
"""
import os
import sys
import glob
import json
import hashlib
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = 'dataset_manifest.json'

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', nargs='+', default=['ASD', 'DGM', 'VNT'])
ap.add_argument('--write-manifest', action='store_true', help='產生指紋檔（在來源機器跑）')
ap.add_argument('--check', action='store_true', help='比對指紋檔（在目標機器跑）')
ap.add_argument('--no-hash', action='store_true', help='跳過 sha256，只做內容檢查（快很多）')
ap.add_argument('--expect-shape', default='192,224,192')
args = ap.parse_args()

SHAPE = tuple(int(x) for x in args.expect_shape.split(','))


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect(path):
    """讀進來實際看內容，不是只看檔案大小。"""
    problems = []
    try:
        d = np.load(path)
    except Exception as e:
        return ['讀不開：%s' % e], None
    keys = set(d.files)
    if 'vol' not in keys:
        problems.append('缺 vol')
        return problems, None
    v = d['vol']
    if v.shape != SHAPE:
        problems.append('vol shape %s != %s' % (v.shape, SHAPE))
    if v.dtype != np.float32:
        problems.append('vol dtype %s != float32' % v.dtype)
    if not np.isfinite(v).all():
        problems.append('vol 有 NaN/Inf')
    else:
        lo, hi = float(v.min()), float(v.max())
        if lo < -1e-6 or hi > 1 + 1e-6:
            problems.append('vol 值域 [%.3f, %.3f] 超出 [0,1]' % (lo, hi))
        if hi < 0.5:
            problems.append('vol 最大值只有 %.3f —— 疑似空的或壞的' % hi)
    if 'seg' in keys:
        s = d['seg']
        if s.shape != SHAPE:
            problems.append('seg shape %s != %s' % (s.shape, SHAPE))
        u = np.unique(s)
        if not np.all(u == u.astype(np.int64)):
            problems.append('seg 有非整數值 —— 內插法用錯了')
        if u.max() > 255 or u.min() < 0:
            problems.append('seg 值 %s 超出 FreeSurfer 標籤範圍' % [u.min(), u.max()])
        if len(u) < 10:
            problems.append('seg 只有 %d 種標籤 —— 疑似壞的' % len(u))
    else:
        problems.append('缺 seg（Dice 評估會做不了）')
    return problems, os.path.getsize(path)


if not (args.write_manifest or args.check):
    print('沒給 --write-manifest 或 --check，只做內容檢查。')

total_bad = 0
for ds in args.datasets:
    prep = os.path.join(ROOT, 'data', ds + '_preprocessed_v1')
    if not os.path.isdir(prep):
        print('[X] 找不到 %s' % prep)
        total_bad += 1
        continue

    files = sorted(glob.glob(os.path.join(prep, '*', '*.npz')))
    print()
    print('=' * 66)
    print('  %s   %d 個 npz' % (ds, len(files)))
    print('=' * 66)

    # 混合集：跟 mixed_manifest.json 對帳（它沒有 subjects.txt）
    # 🔴 2026-09-08 補：原本只查來源三包，漏掉 mixed。實際踩到 —— 三包複製完整，
    #    但 mixed 少了 67 個檔案，train.py 照跑不報錯，只是少看四分之一的資料。
    mman = os.path.join(prep, 'mixed_manifest.json')
    if os.path.exists(mman):
        want = json.load(open(mman, encoding='utf-8'))['members']
        got = {os.path.basename(f) for f in files}
        miss = sorted(set(want) - got)
        extra = sorted(got - set(want))
        if miss or extra:
            print('  [X] 與 mixed_manifest.json 不符：缺 %d 個，多 %d 個' % (len(miss), len(extra)))
            by_ds = {}
            for k in miss:
                by_ds.setdefault(want[k]['dataset'] + '/' + want[k]['split'], []).append(k)
            for k in sorted(by_ds):
                v = by_ds[k]
                print('      缺 %-12s %3d 個：%s%s'
                      % (k, len(v), ', '.join(x[:-4] for x in v[:5]),
                         ' ...' if len(v) > 5 else ''))
            if extra:
                print('      多：%s' % ', '.join(extra[:5]))
            print('      -> 重建：python ASD\\make_mixed_set.py --sources %s --force'
                  % ' '.join(sorted({want[k]['dataset'] for k in want})))
            total_bad += len(miss) + len(extra)
        else:
            print('  [v] 與 mixed_manifest.json 一致（%d 個）' % len(want))

    # 與 subjects.txt 對帳
    slist = os.path.join(ROOT, 'data', ds + '_data', 'fs_stats', 'subjects.txt')
    if os.path.exists(slist):
        ids = {l.strip() for l in open(slist, encoding='utf-8-sig')
               if l.strip() and not l.startswith('#')}
        got = {os.path.basename(f)[:-4] for f in files}
        if ids == got:
            print('  [v] 與 subjects.txt 一致（%d 顆）' % len(ids))
        else:
            print('  [X] 與 subjects.txt 不一致：缺 %s  多 %s'
                  % (sorted(ids - got)[:5] or '無', sorted(got - ids)[:5] or '無'))
            total_bad += 1
    elif not os.path.exists(mman):
        print('  [!] 沒有 subjects.txt，跳過對帳（fs_stats/ 忘了複製？）')

    man_path = os.path.join(prep, MANIFEST)
    old = None
    if args.check:
        if not os.path.exists(man_path):
            print('  [X] 找不到 %s —— 來源機器沒產生，或漏複製' % MANIFEST)
            total_bad += 1
        else:
            old = json.load(open(man_path, encoding='utf-8'))['files']

    man, bad = {}, 0
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, prep).replace('\\', '/')
        problems, size = inspect(f)
        entry = {'size': size}
        if size is not None and not args.no_hash and (args.write_manifest or args.check):
            entry['sha256'] = sha256(f)
        man[rel] = entry

        if old is not None and rel in old:
            o = old[rel]
            if o.get('size') != entry.get('size'):
                problems.append('大小不符：%s -> %s' % (o.get('size'), entry.get('size')))
            elif 'sha256' in o and 'sha256' in entry and o['sha256'] != entry['sha256']:
                problems.append('sha256 不符 —— 內容在搬移中被改動')
        elif old is not None:
            problems.append('指紋檔裡沒有這個檔案')

        if problems:
            bad += 1
            print('  [X] %-14s %s' % (os.path.basename(f)[:-4], '；'.join(problems)))
        if i % 50 == 0:
            print('      ...%d/%d' % (i, len(files)))

    if old is not None:
        missing = sorted(set(old) - set(man))
        if missing:
            print('  [X] 指紋檔有但這裡缺 %d 個：%s' % (len(missing), missing[:5]))
            bad += len(missing)

    print('  %s %d/%d 通過' % ('[v]' if bad == 0 else '[X]', len(files) - bad, len(files)))
    total_bad += bad

    if args.write_manifest:
        json.dump({'dataset': ds, 'shape': list(SHAPE), 'files': man},
                  open(man_path, 'w', encoding='utf-8'), indent=1)
        print('  指紋檔 -> %s' % os.path.relpath(man_path, ROOT))

print()
if total_bad:
    print('🔴 共 %d 個問題。資料不要拿去訓練。' % total_bad)
    sys.exit(1)
print('[v] 全部通過。')
