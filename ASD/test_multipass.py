"""
改架構之前先試水溫（第 0 步）：不重新訓練，把同一顆模型連跑好幾次，看「重複修正」有沒有用。

做法
----
    第 1 次：model(受試者, atlas)               -> 位移 u1，總位移 U1 = u1（= test_dice.py 的結果）
    第 2 次：model(受試者照 U1 搬好, atlas)      -> 修正 u2
             接起來 U2(x) = u2(x) + U1(x + u2(x))   （先照 u2 走、再照 U1 走）
    第 k 次同理。

⭐ 三個細節
1. 接法跟 SpatialTransformer 的慣例一致：out(x) = in(x + u(x))。
   「搬好的影像再搬一次」＝ 原圖在 x + u2(x) + U1(x + u2(x)) 取值，所以總位移就是上面那條。
   位移場用線性內插（對位移場是對的）；標籤才要最近鄰。
2. 每一次的輸入都從「原始影像 + 目前的總位移」重新內插一次，不是把上一次搬好的影像再搬
   —— 不然影像會越搬越糊。標籤也只用最後的總位移搬一次。
3. 第 1 次的 Dice 要跟 test_dice.py 存的 dice_<epoch>.csv 一樣（自我檢查），對不上就是哪裡接錯了。

⚠️ 這不是真的「串兩顆」：模型沒學過「修正」，第 2 次只是把已經對好大半的影像再丟進同一顆。
   有進步 → 值得花 AI 一天訓練真的串兩顆；沒進步 → 不代表串兩顆沒用（第二顆沒機會學），但要保守看待。

用法
----
    python ASD\\test_multipass.py --model models\\mix_exp6\\0190.pt --test-dir data\\mixed_preprocessed_v2\\test --passes 3

輸出 models/<實驗>/multipass_<epoch>.csv：每人、每一次的 Dice、擠爆比例與點數、
這一次在腦內平均多走了幾 mm、每個結構的 Dice。
"""

import os
import sys
import csv
import glob
import time
import argparse
import numpy as np

os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
os.environ.setdefault('VXM_BACKEND', 'pytorch')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'voxelmorph-code'))

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, help='單一 .pt')
ap.add_argument('--test-dir', required=True, help='例如 data\\mixed_preprocessed_v2\\test')
ap.add_argument('--passes', type=int, default=3, help='連跑幾次（1 = 跟 test_dice.py 一樣）')
ap.add_argument('--limit', type=int, default=0, help='只跑前幾位（試跑用；0 = 全部）')
# 2026-10-06：加寬的模型在筆電（8 GB）上推論會溢位到系統記憶體，慢 10 倍以上；限制記憶體又會 OOM。
# --amp 讓網路用半精度跑（位移場接起來、Jacobian、Dice 照舊用單精度），自我檢查會告訴你跟原本差多少。
ap.add_argument('--amp', action='store_true', help='網路用半精度（float16）跑，省一半顯存')
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--out-csv', default=None)
ap.add_argument('--gpu', default='0')
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
mp = os.path.normpath(args.model)
if not (os.path.isfile(mp) and mp.endswith('.pt')):
    sys.exit('[X] --model 要指向單一 .pt：%s' % mp)
if args.passes < 1:
    sys.exit('[X] --passes 至少 1')

import torch
import voxelmorph as vxm
from scipy.stats import wilcoxon
from arch import load_model    # 串接模型也能再連跑（RCN 的 r×n 測法）

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BAR = '=' * 78

atlas_vol = np.load(os.path.normpath(args.atlas))['vol'].astype(np.float32)
atlas_seg = np.load(os.path.normpath(args.atlas_seg))['seg'].astype(np.int32)
LABELS = np.load(os.path.normpath(args.labels))['labels'].astype(int).tolist()
test_files = sorted(glob.glob(os.path.join(os.path.normpath(args.test_dir), '*.npz')))
if not test_files:
    sys.exit('[X] %s 裡沒有 npz' % args.test_dir)
if args.limit:
    test_files = test_files[:args.limit]

inshape = atlas_vol.shape
atlas_t = torch.from_numpy(atlas_vol)[None, None].to(device)
warp_lin = vxm.torch.layers.SpatialTransformer(inshape, mode='bilinear').to(device)
warp_nn = vxm.torch.layers.SpatialTransformer(inshape, mode='nearest').to(device)
brain = torch.from_numpy(atlas_seg > 0).to(device)          # 位移場定義在 atlas 的格子上


def jacobian_negative(flow):
    """負 Jacobian determinant 的比例與點數（算法同 test_dice.py / batch_test_ixi.py）。"""
    d = [[np.gradient(flow[c], axis=a) for a in range(3)] for c in range(3)]
    j11, j12, j13 = 1 + d[0][0], d[0][1], d[0][2]
    j21, j22, j23 = d[1][0], 1 + d[1][1], d[1][2]
    j31, j32, j33 = d[2][0], d[2][1], 1 + d[2][2]
    det = (j11 * (j22 * j33 - j23 * j32)
           - j12 * (j21 * j33 - j23 * j31)
           + j13 * (j21 * j32 - j22 * j31))
    n = int((det <= 0).sum())
    return n / det.size, n


def dice(a, b, lab):
    x, y = (a == lab), (b == lab)
    s = x.sum() + y.sum()
    if s == 0:
        return np.nan
    return 2.0 * (x & y).sum() / s


print()
print(BAR)
print('  連跑 %d 次（不重新訓練）' % args.passes)
print(BAR)
print('  模型     : %s' % mp)
print('  測試資料 : %s  %d 位' % (args.test_dir, len(test_files)))
print('  裝置     : %s%s' % (device, '（網路用半精度 --amp）' if args.amp else ''))
print()

model = load_model(mp, device)
model.to(device)
model.eval()

rows = []
t0 = time.time()
with torch.no_grad():
    for f in test_files:
        d = np.load(f)
        v = torch.from_numpy(d['vol'].astype(np.float32))[None, None].to(device)
        s = torch.from_numpy(d['seg'].astype(np.float32))[None, None].to(device)
        U = None
        line = []
        for k in range(1, args.passes + 1):
            src = v if U is None else warp_lin(v, U)       # 每次都從原圖重新內插
            with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
                _, u = model(src, atlas_t, registration=True)
            u = u.float()
            U = u if U is None else u + warp_lin(U, u)     # U_k(x) = u_k(x) + U_{k-1}(x + u_k(x))

            seg_w = warp_nn(s, U)[0, 0].cpu().numpy()
            frac = float(np.abs(seg_w - np.round(seg_w)).max())
            if frac > 1e-4:
                sys.exit('[X] 搬完的標籤出現非整數值（%.4g）—— 內插法錯了' % frac)
            seg_w = np.round(seg_w).astype(np.int32)
            per = {lab: dice(seg_w, atlas_seg, lab) for lab in LABELS}
            ratio, npts = jacobian_negative(U[0].cpu().numpy())
            step = float(u[0].norm(dim=0)[brain].mean())
            r = {'file': os.path.basename(f), 'pass': k,
                 'dice_mean': float(np.nanmean([per[l] for l in LABELS])),
                 'jneg_pct': 100 * ratio, 'jneg_n': npts, 'step_mm': step, 'per': per}
            rows.append(r)
            line.append('%d:%.4f/%d點/%.2fmm' % (k, r['dice_mean'], npts, step))
        print('  %-28s %s' % (os.path.basename(f)[:28], '   '.join(line)), flush=True)
print('  耗時 %.1f 秒' % (time.time() - t0))

# ── 自我檢查：第 1 次要跟 test_dice.py 的結果一樣 ─────────────────────────
ep = os.path.basename(mp)[:-3]
split = os.path.basename(os.path.normpath(os.path.abspath(args.test_dir)))
ref_csv = os.path.join(os.path.dirname(mp), 'dice_%s%s.csv' % (ep, '' if split == 'test' else '_' + split))
if os.path.exists(ref_csv):
    with open(ref_csv, encoding='utf-8') as fh:
        ref = {r['file']: float(r['dice_mean']) for r in csv.DictReader(fh)}
    diffs = [abs(r['dice_mean'] - ref[r['file']]) for r in rows if r['pass'] == 1 and r['file'] in ref]
    if not diffs:
        print('[!] %s 裡沒有這批受試者，跳過自我檢查' % ref_csv)
    elif max(diffs) > 1e-3:
        sys.exit('[X] 第 1 次的 Dice 跟 %s 對不上（最大差 %.4g）—— 先別看後面的數字' % (ref_csv, max(diffs)))
    else:
        print('[v] 自我檢查：第 1 次跟 %s 一樣（%d 位，最大差 %.1e）'
              % (os.path.basename(ref_csv), len(diffs), max(diffs)))
else:
    print('[!] 找不到 %s，跳過自我檢查' % ref_csv)

# ── 摘要 ─────────────────────────────────────────────────────────────────
files = [r['file'] for r in rows if r['pass'] == 1]
by = {(r['file'], r['pass']): r for r in rows}
D = np.array([[by[(f, k)]['dice_mean'] for f in files] for k in range(1, args.passes + 1)])
print()
print('  %-6s %8s %10s %12s %10s %12s %22s' % ('次數', 'Dice', '比第1次', '較高人數', 'p', '擠爆(點/人)', '這次在腦內平均走幾 mm'))
for k in range(1, args.passes + 1):
    jn = np.mean([by[(f, k)]['jneg_n'] for f in files])
    st = np.mean([by[(f, k)]['step_mm'] for f in files])
    if k == 1:
        print('  %-6d %8.4f %10s %12s %10s %12.1f %22.2f' % (k, D[0].mean(), '', '', '', jn, st))
        continue
    diff = D[k - 1] - D[0]
    p = wilcoxon(D[k - 1], D[0]).pvalue if len(files) >= 6 and np.any(diff != 0) else float('nan')
    print('  %-6d %8.4f %+10.4f %9d/%-3d %10.2g %12.1f %22.2f'
          % (k, D[k - 1].mean(), diff.mean(), int((diff > 0).sum()), len(files), p, jn, st))

if args.passes >= 2:
    g = {l: np.nanmean([by[(f, 2)]['per'][l] - by[(f, 1)]['per'][l] for f in files]) for l in LABELS}
    srt = sorted(LABELS, key=lambda l: g[l])
    print()
    print('  第 2 次 vs 第 1 次，每個結構（FreeSurfer 編號）：')
    print('    變好最多：' + '  '.join('%d %+.4f' % (l, g[l]) for l in srt[::-1][:5]))
    print('    變差最多：' + '  '.join('%d %+.4f' % (l, g[l]) for l in srt[:5]))

out = args.out_csv or os.path.join(os.path.dirname(mp), 'multipass_%s%s.csv' % (ep, '' if split == 'test' else '_' + split))
if args.limit:
    out = out.replace('.csv', '_limit%d.csv' % args.limit)
if args.amp:
    out = out.replace('.csv', '_amp.csv')
with open(out, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['file', 'pass', 'dice_mean', 'jneg_pct', 'jneg_n', 'step_mm'] + ['label_%d' % l for l in LABELS])
    for r in rows:
        w.writerow([r['file'], r['pass'], '%.6f' % r['dice_mean'], '%.6f' % r['jneg_pct'], r['jneg_n'],
                    '%.4f' % r['step_mm']] + ['%.6f' % r['per'][l] for l in LABELS])
print()
print('  CSV -> %s' % out)
print(BAR)
