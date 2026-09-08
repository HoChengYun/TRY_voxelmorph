# -*- coding: utf-8 -*-
"""在 atlas 空間用標籤 Dice 找「同一人的重複掃描」。

為什麼要這支
------------
切分前必須確定「同一個人不會一半進 train、一半進 test」。
檔名規則猜不準（VNT001~VNT009 會被誤判成同一人），DICOM 的人口學欄位在這批也不可信
（技師會複製上一位受試者的登錄資料，D023/D024 與 T029/T028 都是實例）。

🔴 這把尺能做什麼、不能做什麼（2026-09-07 在 ASD 164 顆上實測校準）
--------------------------------------------------------------
    同一次掃描（重複匯出）   0.9793   A0131 / YT13
    同一人不同時間點         0.7318   A0131 / A0132
    不同人 13,366 對         中位數 0.6631  第99百分位 0.7208  最大 0.7473

**同一人不同時間點的 0.7318，比不同人的最大值 0.7473 還低。**
所以這支只抓得到「同一次掃描被存成兩份」，抓不到「同一人掃了兩次」。

⚠️ 我一開始把門檻定在 0.70，是因為拿 1 對同人去比 8 對隨機抽的不同人（0.5743–0.6077），
   樣本太小完全沒看到上尾。全掃之後才發現 0.70 只是第 90 百分位附近，
   結果一口氣「抓到」幾百對誤報。**小樣本校準出來的門檻不能用。**

同一人不同時間點目前沒有可靠的自動判法：
   - 標籤 Dice：見上，分不開
   - 頭形 Dice：ASD 已知不同人的 T040/T060 就到 0.8399
   - FreeSurfer 形態學指標距離：只是在挑「腦大小相近的人」。
     DGM 的 D050/D051（同格式連號、應為不同人）排名卡在兩個真候選中間
   - DICOM 人口學欄位：技師會複製上一位的登錄資料（D023/D024、T029/T028）
→ 只能靠 DICOM 的 StudyInstanceUID / StudyTime，或直接問資料提供者。
   判不出來時採保守做法：**當成同一人歸在一起**。錯了只損失一點切分自由度，
   反過來錯了就是 leakage。

怎麼算得動
----------
DGM 54 顆 = 1431 對、VNT 68 顆 = 2278 對，逐對迴圈太慢。
改成：每個標籤做成 (n_subject, n_voxel) 的 0/1 矩陣，交集用一次矩陣乘法拿到所有配對，
Dice = 2|A∩B| / (|A|+|B|)。30 個標籤就是 30 次 matmul，BLAS 幾秒鐘做完。

預設把影像降取樣 2 倍（省 8 倍記憶體與時間）。降取樣只影響絕對值的第三位小數，
不影響排序 —— 要精確值用 --full 對特定配對重算。

用法
    python ASD\\find_duplicate_scans.py --dataset DGM
    python ASD\\find_duplicate_scans.py --dataset VNT --top 30
    python ASD\\find_duplicate_scans.py --dataset DGM --pairs D038:DGM002 D018:DGM001 --full
"""
import os
import sys
import glob
import argparse
import itertools

import numpy as np

# Windows 主控台預設 cp950，印到 emoji 會 UnicodeEncodeError 直接中斷程式。
# 不要求使用者記得設 PYTHONIOENCODING —— 忘一次就白跑一輪。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 校準點（ASD 164 顆 / 13,366 對實測，見檔頭）
SAME_SCAN = 0.85        # 同一次掃描實測 0.9793；不同人最大 0.7473 -> 中間取值
SUSPICIOUS = 0.7473     # 不同人的實測最大值；超過它才值得人工看
# 沒有「同一人不同時間點」的門檻 —— 那個區間與不同人完全重疊，見檔頭

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='ASD', help='data/<名稱>_preprocessed_v1')
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--top', type=int, default=15, help='列出最像的前 N 對')
ap.add_argument('--stride', type=int, default=2, help='降取樣倍率（--full 等於 1）')
ap.add_argument('--full', action='store_true', help='不降取樣（慢，但值精確）')
ap.add_argument('--pairs', nargs='*', default=None,
                help='只比特定配對，格式 A:B（給了就不做全掃）')
args = ap.parse_args()

stride = 1 if args.full else args.stride
PREP = os.path.join(ROOT, 'data', args.dataset + '_preprocessed_v1')

files = sorted(glob.glob(os.path.join(PREP, 'train', '*.npz'))
               + glob.glob(os.path.join(PREP, 'test', '*.npz')))
if not files:
    sys.exit('[X] %s 底下沒有 npz —— 先跑 run_preprocess.py --dataset %s'
             % (PREP, args.dataset))

names = [os.path.basename(f)[:-4] for f in files]
split = {os.path.basename(f)[:-4]: os.path.basename(os.path.dirname(f)) for f in files}
LAB = np.load(args.labels)['labels'].astype(np.int32)

if args.pairs:
    want = set()
    for p in args.pairs:
        if ':' not in p:
            sys.exit('[X] --pairs 格式是 A:B，收到：%s' % p)
        a, b = p.split(':', 1)
        for x in (a, b):
            if x not in names:
                sys.exit('[X] %s 不在 %s 裡' % (x, PREP))
        want.add((a, b))
    keep = sorted({x for pr in want for x in pr})
    files = [files[names.index(n)] for n in keep]
    names = keep

print('資料集 %s：%d 顆，降取樣 %dx，%d 個標籤'
      % (args.dataset, len(names), stride, len(LAB)))

# ── 讀進來，一次一個標籤做成矩陣 ──────────────────────────────────────
segs = []
for f in files:
    s = np.load(f)['seg']
    segs.append(s[::stride, ::stride, ::stride].astype(np.int32))
shape = segs[0].shape
n = len(segs)
print('每顆 %s = %d voxel' % ('x'.join(map(str, shape)), int(np.prod(shape))))

inter = np.zeros((n, n), dtype=np.float64)   # 各標籤 Dice 的總和
count = np.zeros((n, n), dtype=np.float64)   # 有效標籤數（兩邊都有的才算）

for li, lab in enumerate(LAB):
    M = np.stack([(s == lab).ravel() for s in segs]).astype(np.float32)
    sz = M.sum(1)                                   # 每顆這個標籤的體積
    I = M @ M.T                                     # 所有配對的交集，一次算完
    denom = sz[:, None] + sz[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(denom > 0, 2.0 * I / denom, np.nan)
    ok = ~np.isnan(d)
    inter[ok] += d[ok]
    count[ok] += 1
    del M, I, d
    if (li + 1) % 10 == 0:
        print('  ...%d/%d 標籤' % (li + 1, len(LAB)))

with np.errstate(invalid='ignore'):
    D = inter / count
np.fill_diagonal(D, np.nan)


def verdict(v):
    if v >= SAME_SCAN:
        return '🔴 同一次掃描'
    if v > SUSPICIOUS:
        return '🟡 超出不同人實測上限'
    return '—'


print()
if args.pairs:
    print('%-26s %8s   %s' % ('配對', 'Dice', '判定'))
    for a, b in sorted(want):
        v = D[names.index(a), names.index(b)]
        print('%-26s %8.4f   %s' % ('%s vs %s' % (a, b), v, verdict(v)))
    sys.exit(0)

vals = [(D[i, j], names[i], names[j]) for i, j in itertools.combinations(range(n), 2)]
vals.sort(reverse=True)

print('最像的 %d 對：' % args.top)
print('  %-26s %8s  %-14s %s' % ('配對', 'Dice', '判定', 'split'))
for v, a, b in vals[:args.top]:
    print('  %-26s %8.4f  %-14s %s / %s'
          % ('%s vs %s' % (a, b), v, verdict(v), split[a], split[b]))

arr = np.array([v for v, _, _ in vals])
print()
print('全體 %d 對：中位數 %.4f  第 99 百分位 %.4f  最大 %.4f'
      % (len(vals), np.median(arr), np.percentile(arr, 99), arr.max()))

dup = [(v, a, b) for v, a, b in vals if v >= SAME_SCAN]
odd = [(v, a, b) for v, a, b in vals if SUSPICIOUS < v < SAME_SCAN]
print()
if dup:
    print('🔴 %d 對是同一次掃描存成兩份（>= %.2f）：' % (len(dup), SAME_SCAN))
    for v, a, b in dup:
        flag = '  <-- 橫跨 train/test，會 leakage' if split[a] != split[b] else ''
        print('    %-26s %.4f  (%s / %s)%s' % ('%s vs %s' % (a, b), v, split[a], split[b], flag))
    print()
    print('  處理方式：把重複的那一顆從 --subject-list 移除，或用 --group-map 歸成同一人。')
if odd:
    print('🟡 %d 對超出 ASD 實測的不同人上限（%.4f），但沒到同掃描門檻：'
          % (len(odd), SUSPICIOUS))
    for v, a, b in odd[:10]:
        print('    %-26s %.4f  (%s / %s)' % ('%s vs %s' % (a, b), v, split[a], split[b]))
    print('    僅供人工參考 —— 這個區間與「不同人」完全重疊，不能據此判定同一人。')
if not dup:
    print('[v] 沒有「同一次掃描存成兩份」的情況。')
    print('    ⚠️ 但這支抓不到「同一人掃了兩次」—— 那要靠 DICOM 的 StudyInstanceUID。')
sys.exit(1 if dup else 0)
