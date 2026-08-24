"""
Dice 評估：把受試者的 aseg 用模型的形變場搬到 atlas 空間，跟 atlas 自己的 aseg 比。

這是 VoxelMorph 論文（TMI 2019 §V-A-2）評估配準品質的標準做法，
也是接 FreeSurfer 標籤的真正回報 —— NCC/SSIM 已經飽和，分不出模型好壞。

流程
----
    受試者 npz (vol + seg)
        ↓ model(vol, atlas, registration=True)  -> moved, pos_flow
        ↓ SpatialTransformer(mode='nearest')(seg, pos_flow)
    受試者 seg 在 atlas 空間
        ↓ 跟 atlas seg 逐結構算 Dice
    每個結構一個 Dice

⭐ 三個容易做錯、會安靜給出錯誤數字的地方
------------------------------------------
1. **必須 registration=True**
   networks.py:211 分兩種回傳：訓練時給 preint_flow（未積分、可能半解析度），
   推論時才給 pos_flow（積分過、全解析度）。拿錯的話尺度和解析度都不對。

2. **搬標籤必須 mode='nearest'**
   layers.py:11 的 SpatialTransformer 預設是 'bilinear'，對標籤是錯的
   —— 會在 label 17 和 10 之間插出 13.5 這種不存在的值。

3. **source/target 順序要跟訓練時一致**
   generators.py:118 是 invols = [scan, atlas]，所以 model(受試者, atlas)。
   反過來的話形變場方向相反。

⚠️ 系統性天花板
---------------
MNI152 是 152 顆腦非線性平均，結構本身比個體大（BrainSeg +42%、CerebralWM +59%）。
Affine 已吸收大部分尺寸差（實測受試者被放大約 1.44 倍），但殘餘的邊界模糊仍在。
**即使配準完美，Dice 也達不到 1.0。** 絕對值會低於論文，但方法間的相對比較有效。
報數字時要註明。

用法
----
    # 單一模型
    python ASD\\test_dice.py --model models\\asd_exp1\\0155.pt

    # 掃過多個 epoch（挑最佳用）
    python ASD\\test_dice.py --model-dir models\\asd_exp1 --step 10
"""

import os
import sys
import glob
import time
import argparse
import numpy as np

os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
os.environ.setdefault('VXM_BACKEND', 'pytorch')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'voxelmorph-code'))

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--model', help='單一 .pt')
g.add_argument('--model-dir', help='資料夾，掃過裡面所有 .pt')
g.add_argument('--baseline', action='store_true',
               help='不套任何形變，直接比 Affine 前處理後的 seg 與 atlas seg。'
                    '這是「模型完全沒學到東西」的下限，用來判斷模型到底貢獻了多少。'
                    '論文 Table I 的對應數字是 Affine only = 0.584。')
ap.add_argument('--step', type=int, default=1, help='--model-dir 時每幾個 epoch 評估一次')
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--test-dir', default=os.path.join(ROOT, 'ASD', 'ASD_preprocessed_v1', 'test'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--out-csv', default=None)
ap.add_argument('--gpu', default='0')
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# ── 參數合理性檢查 ───────────────────────────────────────────────────
# --model 要檔案、--model-dir 要資料夾。給錯的話 torch.load 會丟出
# 「PermissionError: Permission denied」，那個訊息完全看不出真正的原因。
if args.model:
    mp = os.path.normpath(args.model)
    if os.path.isdir(mp):
        sys.exit('[X] --model 需要單一 .pt 檔，但 %s 是資料夾。\n'
                 '    要掃過整個資料夾請改用 --model-dir：\n'
                 '        python ASD\\test_dice.py --model-dir %s --step 10'
                 % (mp, args.model.rstrip('\\/')))
    if not os.path.exists(mp):
        sys.exit('[X] 找不到 %s' % mp)
    if not mp.endswith('.pt'):
        sys.exit('[X] --model 應該指向 .pt 檔，收到的是 %s' % mp)

if args.model_dir:
    md = os.path.normpath(args.model_dir)
    if os.path.isfile(md):
        sys.exit('[X] --model-dir 需要資料夾，但 %s 是檔案。\n'
                 '    評估單一模型請改用 --model。' % md)
    if not os.path.isdir(md):
        sys.exit('[X] 找不到資料夾 %s' % md)

if args.step != 1 and not args.model_dir:
    print('[!] --step 只在搭配 --model-dir 時有作用，本次會被忽略。')

import torch
import voxelmorph as vxm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('[!] 沒有 GPU，會很慢')

BAR = '=' * 78

# ── 載入 atlas 與標籤定義 ────────────────────────────────────────────
atlas_vol = np.load(os.path.normpath(args.atlas))['vol'].astype(np.float32)
if not os.path.exists(args.atlas_seg):
    sys.exit('[X] 找不到 atlas 的 seg：%s\n'
             '    要先跑 ASD\\make_atlas_seg.py 產生。' % args.atlas_seg)
atlas_seg = np.load(os.path.normpath(args.atlas_seg))['seg'].astype(np.int32)
LABELS = np.load(os.path.normpath(args.labels))['labels'].astype(int).tolist()

if atlas_vol.shape != atlas_seg.shape:
    sys.exit('[X] atlas vol %s 與 seg %s shape 不一致' % (atlas_vol.shape, atlas_seg.shape))

missing = sorted(set(LABELS) - set(np.unique(atlas_seg).tolist()))
if missing:
    sys.exit('[X] atlas seg 缺少評估用標籤 %s' % missing)

test_files = sorted(glob.glob(os.path.join(os.path.normpath(args.test_dir), '*.npz')))
if not test_files:
    sys.exit('[X] %s 裡沒有 npz' % args.test_dir)

print()
print(BAR)
print('  Dice 評估')
print(BAR)
print('  atlas      : %s  %s' % (os.path.basename(args.atlas), atlas_vol.shape))
print('  atlas seg  : %s  %d 種標籤'
      % (os.path.basename(args.atlas_seg), len(np.unique(atlas_seg))))
print('  評估結構   : %d 個（labels.npz）' % len(LABELS))
print('  測試資料   : %s  %d 筆' % (args.test_dir, len(test_files)))
print('  裝置       : %s' % device)

atlas_t = torch.from_numpy(atlas_vol)[None, None].to(device)
inshape = atlas_vol.shape

# 影像用線性、標籤用最近鄰 —— 兩個不同的 transformer
warp_lin = vxm.torch.layers.SpatialTransformer(inshape, mode='bilinear').to(device)
warp_nn = vxm.torch.layers.SpatialTransformer(inshape, mode='nearest').to(device)


def jacobian_negative_ratio(flow):
    """負 Jacobian determinant 比例（沿用 batch_test_ixi.py 的算法）。"""
    d = [[np.gradient(flow[c], axis=a) for a in range(3)] for c in range(3)]
    j11, j12, j13 = 1 + d[0][0], d[0][1], d[0][2]
    j21, j22, j23 = d[1][0], 1 + d[1][1], d[1][2]
    j31, j32, j33 = d[2][0], d[2][1], 1 + d[2][2]
    det = (j11 * (j22 * j33 - j23 * j32)
           - j12 * (j21 * j33 - j23 * j31)
           + j13 * (j21 * j32 - j22 * j31))
    return float((det <= 0).sum() / det.size)


def dice(a, b, lab):
    """單一結構的 Dice。兩邊都沒有該結構時回傳 nan（不計入平均）。"""
    x, y = (a == lab), (b == lab)
    s = x.sum() + y.sum()
    if s == 0:
        return np.nan
    return 2.0 * (x & y).sum() / s


def evaluate(model_path):
    model = vxm.networks.VxmDense.load(model_path, device)
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for f in test_files:
            d = np.load(f)
            if 'seg' not in d:
                sys.exit('[X] %s 沒有 seg —— 這批 npz 不是 preprocess_fs.py 產生的' % f)
            vol = d['vol'].astype(np.float32)
            seg = d['seg'].astype(np.int32)

            v = torch.from_numpy(vol)[None, None].to(device)
            # ⭐ source=受試者, target=atlas（跟訓練時的 [scan, atlas] 一致）
            # ⭐ registration=True -> 拿積分過、全解析度的 pos_flow
            moved, flow = model(v, atlas_t, registration=True)

            # ⭐ 標籤用最近鄰搬
            s = torch.from_numpy(seg.astype(np.float32))[None, None].to(device)
            seg_w = warp_nn(s, flow)[0, 0].cpu().numpy()
            frac = float(np.abs(seg_w - np.round(seg_w)).max())
            if frac > 1e-4:
                sys.exit('[X] 搬完的標籤出現非整數值（%.4g）—— 內插法錯了' % frac)
            seg_w = np.round(seg_w).astype(np.int32)

            per = {lab: dice(seg_w, atlas_seg, lab) for lab in LABELS}
            vals = np.array([per[l] for l in LABELS], dtype=float)
            fl = flow[0].cpu().numpy()
            rows.append({
                'file': os.path.basename(f),
                'dice_mean': float(np.nanmean(vals)),
                'jneg_pct': 100 * jacobian_negative_ratio(fl),
                'per': per,
            })
    return rows


def summarize(rows, tag):
    dm = np.array([r['dice_mean'] for r in rows])
    jn = np.array([r['jneg_pct'] for r in rows])
    print('  %-14s Dice %.4f ± %.4f   %%|J|<=0 %.4f%%' % (tag, dm.mean(), dm.std(), jn.mean()))
    return dm.mean(), jn.mean()


def evaluate_baseline():
    """不套形變：Affine 前處理之後的 seg 直接跟 atlas seg 比。"""
    rows = []
    for f in test_files:
        d = np.load(f)
        seg = d['seg'].astype(np.int32)
        per = {lab: dice(seg, atlas_seg, lab) for lab in LABELS}
        vals = np.array([per[l] for l in LABELS], dtype=float)
        rows.append({'file': os.path.basename(f),
                     'dice_mean': float(np.nanmean(vals)),
                     'jneg_pct': 0.0,          # 沒有形變場，定義上為 0
                     'per': per})
    return rows


# ── 基準線（不套形變）───────────────────────────────────────────────
if args.baseline:
    print()
    print('  模式 : 基準線（Affine only，不套任何形變）')
    print()
    rows = evaluate_baseline()
    print('  %-38s %8s' % ('受試者', 'Dice'))
    for r in rows:
        print('  %-38s %8.4f' % (r['file'][:38], r['dice_mean']))
    print('  ' + '-' * 50)
    dm = np.array([r['dice_mean'] for r in rows])
    print('  %-38s %8.4f ± %.4f' % ('平均', dm.mean(), dm.std()))
    print()
    print('  這是模型的下限：任何訓練好的模型都應該明顯高於這個數字，')
    print('  否則代表它沒學到東西（或形變場幾乎是零）。')
    print('  論文 Table I 的對應數字：Affine only = 0.584。')

    per = {l: np.nanmean([r['per'][l] for r in rows]) for l in LABELS}
    print()
    print('  逐結構基準線：')
    for i, l in enumerate(LABELS):
        end = '\n' if (i + 1) % 5 == 0 else '   '
        print('    %3d:%.3f' % (l, per[l]), end=end)
    if len(LABELS) % 5:
        print()

    import csv
    out = args.out_csv or os.path.join(os.path.dirname(args.test_dir.rstrip('\\/')),
                                       'dice_baseline.csv')
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'dice_mean'] + ['label_%d' % l for l in LABELS])
        for r in rows:
            w.writerow([r['file'], '%.6f' % r['dice_mean']]
                       + ['%.6f' % r['per'][l] for l in LABELS])
    print()
    print('  CSV -> %s' % out)

# ── 單一模型 ────────────────────────────────────────────────────────
elif args.model:
    mp = os.path.normpath(args.model)
    print()
    print('  模型 : %s' % mp)
    print()
    t0 = time.time()
    rows = evaluate(mp)
    print('  %-38s %8s %10s' % ('受試者', 'Dice', '%|J|<=0'))
    for r in rows:
        print('  %-38s %8.4f %9.4f%%' % (r['file'][:38], r['dice_mean'], r['jneg_pct']))
    print('  ' + '-' * 60)
    dm, jn = summarize(rows, '平均')
    print('  耗時 %.1f 秒' % (time.time() - t0))

    # 逐結構
    print()
    print('  逐結構 Dice（%d 個）：' % len(LABELS))
    per = {l: np.nanmean([r['per'][l] for r in rows]) for l in LABELS}
    for i, l in enumerate(LABELS):
        end = '\n' if (i + 1) % 5 == 0 else '   '
        print('    %3d:%.3f' % (l, per[l]), end=end)
    if len(LABELS) % 5:
        print()

    out = args.out_csv or os.path.join(os.path.dirname(mp),
                                       'dice_%s.csv' % os.path.basename(mp)[:-3])
    import csv
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'dice_mean', 'jneg_pct'] + ['label_%d' % l for l in LABELS])
        for r in rows:
            w.writerow([r['file'], '%.6f' % r['dice_mean'], '%.6f' % r['jneg_pct']]
                       + ['%.6f' % r['per'][l] for l in LABELS])
    print()
    print('  CSV -> %s' % out)

# ── 掃過多個 epoch ──────────────────────────────────────────────────
else:
    md = os.path.normpath(args.model_dir)
    pts = sorted(glob.glob(os.path.join(md, '*.pt')))[::args.step]
    if not pts:
        sys.exit('[X] %s 裡沒有 .pt' % md)
    print()
    print('  掃描 %s，共 %d 個檢查點（--step %d）' % (md, len(pts), args.step))
    print()
    curve = []
    for p in pts:
        ep = int(os.path.basename(p)[:-3])
        rows = evaluate(p)
        dm, jn = summarize(rows, 'epoch %04d' % ep)
        curve.append((ep, dm, jn))

    import csv
    out = args.out_csv or os.path.join(md, 'dice_curve.csv')
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['epoch', 'dice_mean', 'jneg_pct'])
        w.writerows([[e, '%.6f' % d, '%.6f' % j] for e, d, j in curve])
    print()
    print('  CSV -> %s' % out)

    best = max(curve, key=lambda x: x[1])
    print()
    print('  ★ Dice 最高：epoch %d   Dice %.4f   %%|J|<=0 %.4f%%' % best)
    print()
    print('  ⚠️ 挑 epoch 時不要只看 Dice —— 亂折疊也可以把 Dice 衝高。')
    print('     論文 Table I 的 %|J|<=0：VoxelMorph(CC) 0.366%、ANTs SyN 0.185%。')
    print('     0.1% 量級屬正常，2% 以上要避開。')

print()
print(BAR)
print('  ⚠️ 解讀提醒：atlas 是 152 顆腦的非線性平均，結構邊界比個體模糊，')
print('     存在系統性天花板 —— 即使配準完美 Dice 也達不到 1.0。')
print('     絕對值會低於論文（0.75 量級），但方法間的相對比較仍然有效。')
print(BAR)
