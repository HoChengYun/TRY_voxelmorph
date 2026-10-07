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
    # 基準線（只做線性對位，不套模型）
    python ASD\\test_dice.py --baseline --test-dir data\\mixed_preprocessed_v1\\test --exp-name mix_exp1

    # 單一模型
    python ASD\\test_dice.py --model models\\mix_exp1\\0230.pt --test-dir data\\mixed_preprocessed_v1\\test

    # 掃過多個 epoch（挑最佳用）
    python ASD\\test_dice.py --model-dir models\\mix_exp1 --step 10 --test-dir data\\mixed_preprocessed_v1\\test

    # tigerbx 組：atlas 分割也要換
    python ASD\\test_dice.py --model-dir models\\tiger_exp1 --step 10 ^
        --test-dir data\\tigerbx_preprocessed_v1\\test --atlas-seg IXI\\atlas_mni152_09c_v3_seg_tigerbx.npz

--test-dir 必填，直接給路徑（2026-09-13 改）：原本的 --dataset ASD 會自己組出
data/ASD_preprocessed_v1，指令上看不出用的是哪一版；有了 v2 之後還會安靜地拿 v1 去跑。

--surface（2026-10-07 加）：另外算 HD95 與 SDlogJ（Learn2Reg 的報告方式），寫到 surface_<epoch>.csv
    HD95  ：每個結構，兩個標籤的邊界點到對方邊界的最短距離，兩個方向各取第 95 百分位、取較大者
            （同 MONAI、DeepMind surface-distance 的 robust Hausdorff）；1 voxel = 1 mm，單位 mm，越小越好
    SDlogJ：log|J| 的標準差，在 atlas 腦遮罩（seg > 0）內計算；|J| ≤ 0 先截到 1e-9 再取 log。越小 = 形變越平滑
    ⚠️ 已有 dice_<epoch>.csv 時不會覆蓋（那是正式結果），只拿重算的 Dice 跟它對一次，確認推論可重現。
    --amp：網路用半精度（2× width 模型在 8 GB 筆電上要加），輸出檔名多 _amp。
    只支援 --model 與 --baseline（--model-dir 掃 epoch 時太慢）。

    python ASD\\test_dice.py --model models\\mix_exp6\\0190.pt --test-dir data\\mixed_preprocessed_v2\\test --surface
    python ASD\\test_dice.py --baseline --test-dir data\\mixed_preprocessed_v2\\test --exp-name mix_exp2 --surface
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
ap.add_argument('--test-dir', required=True,
                help='test 資料夾，例如 data\\mixed_preprocessed_v1\\test')
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--exp-name', default=None,
                help='--baseline 時把結果寫到 models/<實驗名>/。基準線只跟「test 集 + atlas」'
                     '有關、跟模型無關，但放進實驗資料夾能讓每個實驗自成一體，'
                     '之後翻舊實驗不用另外找對照。不給則寫到 models/<test 上一層的資料夾名>_baseline/')
ap.add_argument('--out-csv', default=None)
ap.add_argument('--gpu', default='0')
ap.add_argument('--surface', action='store_true',
                help='另外算 HD95（mm）與 SDlogJ，寫到 surface_<epoch>.csv（見檔頭）')
ap.add_argument('--amp', action='store_true', help='網路用半精度（float16）跑，省顯存；標籤搬移與指標照舊單精度')
args = ap.parse_args()
if args.surface and args.model_dir:
    sys.exit('[X] --surface 只支援 --model 與 --baseline（掃 epoch 時 HD95 太慢）')

args.test_dir = os.path.normpath(args.test_dir)
if not os.path.isdir(args.test_dir):
    sys.exit('[X] 找不到 test 資料夾：%s' % args.test_dir)

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
    if not mp.endswith(('.pt', '.h5')):
        sys.exit('[X] --model 應該指向 .pt（或作者的 .h5）檔，收到的是 %s' % mp)

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
from scipy import ndimage
from arch import load_model    # 2026-10-06：新架構（串接等）也讀得到；舊模型照舊是 VxmDense.load

# Windows 主控台預設 cp950，印到 emoji 會 UnicodeEncodeError 直接中斷程式。
# 不要求使用者記得設 PYTHONIOENCODING —— 忘一次就白跑一輪。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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
# 🔴 2026-09-16 加：同一顆模型會在 val（挑 epoch）和 test（最後一次）各評估一次。
#    兩次都寫 dice_curve.csv 的話，後跑的會安靜蓋掉先跑的 —— 而且看檔名分不出
#    這條曲線是哪一份資料算的。資料夾名不是 test 就自動加後綴。
def _split_suffix(test_dir):
    b = os.path.basename(os.path.normpath(os.path.abspath(test_dir)))
    return '' if b == 'test' else '_' + b

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


def jacobian_det(flow):
    """每個 voxel 的 Jacobian determinant det(I + ∇u)（中央差分，沿用 batch_test_ixi.py 的算法）。"""
    d = [[np.gradient(flow[c], axis=a) for a in range(3)] for c in range(3)]
    j11, j12, j13 = 1 + d[0][0], d[0][1], d[0][2]
    j21, j22, j23 = d[1][0], 1 + d[1][1], d[1][2]
    j31, j32, j33 = d[2][0], d[2][1], 1 + d[2][2]
    return (j11 * (j22 * j33 - j23 * j32)
            - j12 * (j21 * j33 - j23 * j31)
            + j13 * (j21 * j32 - j22 * j31))


def jacobian_negative_ratio(flow, det=None):
    """負 Jacobian determinant 比例，分母是整個影像（舊定義，jneg_pct 欄位；保留是為了跟以前的 CSV 對得上）。"""
    det = jacobian_det(flow) if det is None else det
    return float((det <= 0).sum() / det.size)


def folding_fg(det):
    """負 Jacobian determinant 比例，分母是 atlas 非背景 voxel（jneg_fg_pct 欄位）。
    2026-10-07 起報告一律用這個（使用者：「folding 比例的分母我想和 VoxelMorph 一樣」）。"""
    return float((det[FG] <= 0).mean())


# SDlogJ 與 jneg_fg_pct 的計算範圍：fixed image（atlas）的非背景 voxel（1,867,705 個，約占全影像 22.6%）。
# 🔴 2026-10-07：VoxelMorph 論文文字是「count all non-background voxels for which |J| ≤ 0」（§V-A-2），但 Table I 標題寫明
#    百分比的分母是固定的 5.2 M voxel（他們的「腦內」）。我們的分母（非背景 1.87 M、整個影像 8.26 M）都不同，
#    百分比不能跟論文的 0.366% 直接比 → 跟論文比較一律用 folding voxel 數（論文 VoxelMorph (CC) 19,077）。
FG = atlas_vol > 0


def sdlogj(det):
    """log|J| 的標準差（atlas 非背景內）。|J| ≤ 0 的地方 log 無定義，先截到 1e-9（同 Learn2Reg 的做法）。"""
    return float(np.log(np.clip(det[FG], 1e-9, 1e9)).std())


def hd95(x, y):
    """兩個二值標籤的 95% Hausdorff distance（voxel = 1 mm）。
    邊界 = 標籤內、至少一個 6 鄰居不在標籤內的 voxel；兩個方向各取第 95 百分位，回傳較大者
    （同 MONAI compute_hausdorff_distance(percentile=95)、DeepMind surface-distance 的 robust Hausdorff）。
    為了快，只在兩個標籤的聯集外框（外擴 2 voxel）內算距離轉換；邊界都在框內，距離不受影響。"""
    if not x.any() or not y.any():
        return np.nan
    idx = np.argwhere(x | y)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    x, y = x[sl], y[sl]
    bx = x & ~ndimage.binary_erosion(x)
    by = y & ~ndimage.binary_erosion(y)
    to_x = ndimage.distance_transform_edt(~bx)       # 每個 voxel 到 x 邊界的距離
    to_y = ndimage.distance_transform_edt(~by)
    return float(max(np.percentile(to_y[bx], 95), np.percentile(to_x[by], 95)))


def dice(a, b, lab):
    """單一結構的 Dice。兩邊都沒有該結構時回傳 nan（不計入平均）。"""
    x, y = (a == lab), (b == lab)
    s = x.sum() + y.sum()
    if s == 0:
        return np.nan
    return 2.0 * (x & y).sum() / s


def evaluate(model_path):
    if model_path.endswith('.h5'):
        # 作者釋出的 Keras 模型：這台的 TF 載不起來，搬進 PyTorch 用（見 author_model.py）
        from author_model import load_author_h5
        model = load_author_h5(model_path, device)
    else:
        model = load_model(model_path, device)
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
            with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
                moved, flow = model(v, atlas_t, registration=True)
            flow = flow.float()

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
            det = jacobian_det(fl)
            r = {
                'file': os.path.basename(f),
                'dice_mean': float(np.nanmean(vals)),
                'jneg_pct': 100 * jacobian_negative_ratio(fl, det),      # 舊定義（分母：整個影像）
                'jneg_fg_pct': 100 * folding_fg(det),                    # 分母：atlas 非背景 ← 報告用這個（比我們自己的模型）
                'per': per,
            }
            if args.surface:
                add_surface(r, seg_w, det)
            rows.append(r)
    return rows


def add_surface(r, seg_w, det):
    """--surface：每個結構的 HD95＋這位受試者的 SDlogJ（det 為 None = 沒有形變，SDlogJ 定義為 0）。"""
    r['hd'] = {lab: hd95(seg_w == lab, atlas_seg == lab) for lab in LABELS}
    r['hd95_mean'] = float(np.nanmean([r['hd'][l] for l in LABELS]))
    r['sdlogj'] = 0.0 if det is None else sdlogj(det)
    r['jneg_fg_pct'] = 0.0 if det is None else 100 * folding_fg(det)                 # 分母：atlas 非背景
    print('    %-30s Dice %.4f  HD95 %.2f mm  SDlogJ %.4f  folding（非背景）%.4f%%'
          % (r['file'][:30], r['dice_mean'], r['hd95_mean'], r['sdlogj'], r['jneg_fg_pct']), flush=True)


def write_surface(rows, out):
    import csv
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'dice_mean', 'jneg_pct', 'jneg_fg_pct', 'hd95_mean', 'sdlogj']
                   + ['hd95_label_%d' % l for l in LABELS])
        for r in rows:
            w.writerow([r['file'], '%.6f' % r['dice_mean'], '%.6f' % r['jneg_pct'], '%.6f' % r['jneg_fg_pct'],
                        '%.4f' % r['hd95_mean'], '%.6f' % r['sdlogj']] + ['%.4f' % r['hd'][l] for l in LABELS])
    hd = np.array([r['hd95_mean'] for r in rows])
    sj = np.array([r['sdlogj'] for r in rows])
    print()
    print('  HD95   %.3f ± %.3f mm（30 個結構平均，再平均 %d 位）' % (hd.mean(), hd.std(), len(rows)))
    print('  SDlogJ %.4f ± %.4f' % (sj.mean(), sj.std()))
    print('  folding：分母整個影像 %.4f%%；分母 atlas 非背景 %.4f%%'
          % (np.mean([r['jneg_pct'] for r in rows]), np.mean([r['jneg_fg_pct'] for r in rows])))
    print('  CSV -> %s' % out)


def keep_existing(out, rows):
    """--surface 或 --amp 時，已有的 dice CSV 是正式結果，不覆蓋；拿重算的 Dice 跟它對一次（推論可重現的檢查）。
    回傳 True = 保留舊檔、不要寫。"""
    if not ((args.surface or args.amp) and os.path.exists(out)):
        return False
    import csv
    with open(out, encoding='utf-8') as fh:
        old = {r['file']: float(r['dice_mean']) for r in csv.DictReader(fh)}
    diff = [abs(old[r['file']] - r['dice_mean']) for r in rows if r['file'] in old]
    print()
    print('  [i] 保留既有的 %s（正式結果，不覆蓋）' % os.path.basename(out))
    print('      重算的 Dice 與它比：%d 位，最大差 %.2e、平均差 %.2e%s'
          % (len(diff), max(diff), float(np.mean(diff)), '（--amp 半精度，有小差異屬正常）' if args.amp else ''))
    return True


def summarize(rows, tag):
    dm = np.array([r['dice_mean'] for r in rows])
    jn = np.array([r['jneg_pct'] for r in rows])
    jf = np.array([r['jneg_fg_pct'] for r in rows])
    print('  %-14s Dice %.4f ± %.4f   folding（非背景）%.4f%%' % (tag, dm.mean(), dm.std(), jf.mean()))
    return dm.mean(), jn.mean(), jf.mean()


def evaluate_baseline():
    """不套形變：Affine 前處理之後的 seg 直接跟 atlas seg 比。"""
    rows = []
    for f in test_files:
        d = np.load(f)
        seg = d['seg'].astype(np.int32)
        per = {lab: dice(seg, atlas_seg, lab) for lab in LABELS}
        vals = np.array([per[l] for l in LABELS], dtype=float)
        r = {'file': os.path.basename(f),
             'dice_mean': float(np.nanmean(vals)),
             'jneg_pct': 0.0,          # 沒有形變場，定義上為 0
             'jneg_fg_pct': 0.0,
             'per': per}
        if args.surface:
            add_surface(r, seg, None)
        rows.append(r)
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
    # 🔴 2026-09-08 改：原本寫到 data/<資料集>_preprocessed_v1/ 底下。
    #    data/ 應該只放資料，衍生結果全部進 models/ —— 同一份資料集會被很多實驗用到，
    #    把結果混進去之後很難分辨哪個檔案是輸入、哪個是產出。
    # 沒給 --exp-name 就用資料夾名（含版本），例如 models/mixed_preprocessed_v1_baseline/
    prep_name = os.path.basename(os.path.dirname(os.path.abspath(args.test_dir)))
    out = args.out_csv or os.path.join(
        ROOT, 'models', args.exp_name or (prep_name + '_baseline'),
        'dice_baseline%s.csv' % _split_suffix(args.test_dir))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not keep_existing(out, rows):
        with open(out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'dice_mean'] + ['label_%d' % l for l in LABELS])
            for r in rows:
                w.writerow([r['file'], '%.6f' % r['dice_mean']]
                           + ['%.6f' % r['per'][l] for l in LABELS])
        print()
        print('  CSV -> %s' % out)
    if args.surface:
        write_surface(rows, os.path.join(os.path.dirname(out), 'surface_baseline%s.csv' % _split_suffix(args.test_dir)))

# ── 單一模型 ────────────────────────────────────────────────────────
elif args.model:
    mp = os.path.normpath(args.model)
    print()
    print('  模型 : %s' % mp)
    print()
    t0 = time.time()
    rows = evaluate(mp)
    print('  %-38s %8s %10s' % ('受試者', 'Dice', 'folding'))
    for r in rows:
        print('  %-38s %8.4f %9.4f%%' % (r['file'][:38], r['dice_mean'], r['jneg_fg_pct']))
    print('  ' + '-' * 60)
    dm, jn, jf = summarize(rows, '平均')
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

    out = args.out_csv or os.path.join(
        os.path.dirname(mp),
        'dice_%s%s.csv' % (os.path.basename(mp)[:-3], _split_suffix(args.test_dir)))
    import csv
    if not keep_existing(out, rows):
        with open(out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'dice_mean', 'jneg_pct', 'jneg_fg_pct'] + ['label_%d' % l for l in LABELS])
            for r in rows:
                w.writerow([r['file'], '%.6f' % r['dice_mean'], '%.6f' % r['jneg_pct'], '%.6f' % r['jneg_fg_pct']]
                           + ['%.6f' % r['per'][l] for l in LABELS])
        print()
        print('  CSV -> %s' % out)
    if args.surface:
        write_surface(rows, os.path.join(os.path.dirname(out), 'surface_%s%s%s.csv'
                                         % (os.path.basename(mp)[:-3], _split_suffix(args.test_dir),
                                            '_amp' if args.amp else '')))

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
        dm, jn, jf = summarize(rows, 'epoch %04d' % ep)
        curve.append((ep, dm, jn, jf))

    import csv
    out = args.out_csv or os.path.join(md, 'dice_curve%s.csv' % _split_suffix(args.test_dir))
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['epoch', 'dice_mean', 'jneg_pct', 'jneg_fg_pct'])
        w.writerows([[e, '%.6f' % d, '%.6f' % j, '%.6f' % jf] for e, d, j, jf in curve])
    print()
    print('  CSV -> %s' % out)

    best = max(curve, key=lambda x: x[1])
    print()
    print('  ★ Dice 最高：epoch %d   Dice %.4f   folding（非背景）%.4f%%' % (best[0], best[1], best[3]))
    print()
    print('  ⚠️ 挑 epoch 時不要只看 Dice —— 亂折疊也可以把 Dice 衝高。')
    print('     跟論文比較請用 folding voxel 數（論文 Table I 的百分比分母是 5.2 M voxel，跟這裡不同）：VoxelMorph(CC) 19,077、ANTs SyN 9,662。')

print()
print(BAR)
print('  ⚠️ 解讀提醒：atlas 是 152 顆腦的非線性平均，結構邊界比個體模糊，')
print('     存在系統性天花板 —— 即使配準完美 Dice 也達不到 1.0。')
print('     絕對值會低於論文（0.75 量級），但方法間的相對比較仍然有效。')
print(BAR)
