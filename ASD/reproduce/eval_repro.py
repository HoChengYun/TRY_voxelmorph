# -*- coding: utf-8 -*-
"""照原作重現的模型（DWP、RCN）用我們的評估。算法逐項同 ASD/test_dice.py（那支是腳本、import 不進來，這裡照抄）：
  Dice：30 個結構（voxelmorph-code/data/labels.npz），受試者的 seg 用最近鄰搬到 atlas 空間
  folding：|J| ≤ 0（中央差分），jneg_pct 分母整個影像、jneg_fg_pct 分母 atlas 非背景
  影像項 ncc_loss：VoxelMorph 的局部 NCC（窗格 9），NCC(atlas, 搬好的受試者)
  --surface：HD95（每個結構，兩方向第 95 百分位取大）、SDlogJ（atlas 非背景內）
CSV 的格式、檔名也同 test_dice.py（dice_<名>.csv、dice_curve.csv、surface_<名>.csv；資料夾不是 test 時加 _<資料夾名>），
之後的畫圖、簡報程式照樣讀。

各方法的推論照原作：
  dwp：模型目錄的 train_configs.py（train_dwp.py 存的）蓋出 VXMPlus_DualWarpPy＋RegistrationHead；
       混合精度（同原作 val_iter）→ 最細那層的位移放大到全尺寸 → 位移、搬好的受試者。要用 venv_dwp 跑。
  rcn：rcn_torch.RecursiveCascadedNetworks(fixed = atlas, moving = 受試者)，最後的合成位移（推論不加隨機形變）。

用法
  python ASD\\reproduce\\eval_repro.py --method rcn --model-dir runs\\rcn_vtn3 --step 2 --test-dir data\\mixed_preprocessed_v2\\val
  python ASD\\reproduce\\eval_repro.py --method rcn --model runs\\rcn_vtn3\\100000.pt --test-dir data\\mixed_preprocessed_v2\\test --surface
  D:\\MyHome\\MRI\\orig_repos\\venv_dwp\\Scripts\\python.exe ASD\\reproduce\\eval_repro.py --method dwp --model-dir runs\\dwp_mixv2 ^
      --step 5 --test-dir data\\mixed_preprocessed_v2\\val
"""
import os
import re
import sys
import csv
import glob
import time
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--method', required=True, choices=['dwp', 'rcn'])
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--model', help='單一檢查點（dwp：NNNN.pth；rcn：NNNNNN.pt）')
g.add_argument('--model-dir', help='資料夾，掃過裡面的檢查點')
ap.add_argument('--step', type=int, default=1, help='--model-dir 時每幾個檢查點評估一次（另外一定包含最後一個）')
ap.add_argument('--test-dir', required=True)
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--orig-repo', default=r'D:\MyHome\MRI\orig_repos\rethink-reg', help='dwp：原作 repo')
ap.add_argument('--out-csv', default=None)
ap.add_argument('--surface', action='store_true', help='另算 HD95、SDlogJ（只給 --model）')
ap.add_argument('--max-subjects', type=int, default=0, help='測試用：只評估前幾位')
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
if args.surface and args.model_dir:
    sys.exit('[X] --surface 只支援 --model（掃檢查點時 HD95 太慢）')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch                                    # noqa: E402
import torch.nn.functional as F                 # noqa: E402
from scipy import ndimage                       # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
LABELS = np.load(args.labels)['labels'].astype(int).tolist()
FG = atlas_vol > 0
files = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))
if args.max_subjects:
    files = files[:args.max_subjects]
if not files:
    sys.exit('[X] %s 裡沒有 npz' % args.test_dir)
atlas_t = torch.from_numpy(atlas_vol)[None, None].to(device)
SUFFIX = '' if os.path.basename(os.path.normpath(os.path.abspath(args.test_dir))) == 'test' \
    else '_' + os.path.basename(os.path.normpath(os.path.abspath(args.test_dir)))


# ── 模型 ─────────────────────────────────────────────────────────────

def load_dwp(path):
    sys.path.insert(0, args.orig_repo)
    from mmengine import Config
    from models import build_flow_estimator, build_registration_head
    cfg = Config.fromfile(os.path.join(os.path.dirname(path), 'train_configs.py'))
    model = build_flow_estimator(cfg.model_cfg)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    head = build_registration_head(cfg.registration_cfg).to(device)
    amp = getattr(torch, cfg.amp_dtype)

    def register(v):
        with torch.autocast(device_type='cuda', dtype=amp, enabled=cfg.use_amp and device == 'cuda'):
            fwd, _ = model(v, atlas_t)                      # source = 受試者、target = atlas（同訓練）
            flow, moved, _ = head(fwd[-1], v)
        return moved.float(), flow.float()
    return register


def load_rcn(path):
    sys.path.insert(0, HERE)
    import rcn_torch as R
    ck = torch.load(path, map_location=device)
    a = ck['args']
    model = R.RecursiveCascadedNetworks(a['base_network'], a['n_cascades'], a['image_size'], rep=a['rep'],
                                        **a.get('net_args', {}))
    model.load_state_dict(ck['state_dict'])
    model.to(device).eval()

    def register(v):
        out = model(atlas_t, v)                             # fixed = atlas、moving = 受試者（同訓練）
        return out['warped'], out['flow']
    return register


# ── 指標（同 test_dice.py）──────────────────────────────────────────────

def warp_nearest(seg, flow):
    """同 VoxelMorph SpatialTransformer(mode='nearest')：grid + flow、align_corners=True、超出範圍補 0。"""
    size = flow.shape[2:]
    r = [torch.arange(n, device=flow.device, dtype=torch.float32) for n in size]
    loc = torch.stack(torch.meshgrid(*r, indexing='ij'), 0)[None] + flow
    loc = torch.stack([2 * (loc[:, i] / (n - 1) - 0.5) for i, n in enumerate(size)], -1)
    return F.grid_sample(seg, loc.flip(-1), mode='nearest', align_corners=True)


def jacobian_det(flow):
    d = [[np.gradient(flow[c], axis=a) for a in range(3)] for c in range(3)]
    j11, j12, j13 = 1 + d[0][0], d[0][1], d[0][2]
    j21, j22, j23 = d[1][0], 1 + d[1][1], d[1][2]
    j31, j32, j33 = d[2][0], d[2][1], 1 + d[2][2]
    return (j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31) + j13 * (j21 * j32 - j22 * j31))


def hd95(x, y):
    if not x.any() or not y.any():
        return np.nan
    idx = np.argwhere(x | y)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    x, y = x[sl], y[sl]
    bx = x & ~ndimage.binary_erosion(x)
    by = y & ~ndimage.binary_erosion(y)
    to_x = ndimage.distance_transform_edt(~bx)
    to_y = ndimage.distance_transform_edt(~by)
    return float(max(np.percentile(to_y[bx], 95), np.percentile(to_x[by], 95)))


def dice(a, b, lab):
    x, y = (a == lab), (b == lab)
    s = x.sum() + y.sum()
    return np.nan if s == 0 else 2.0 * (x & y).sum() / s


def ncc_loss(I, J, win=9):
    sum_filt = torch.ones([1, 1, win, win, win], device=I.device)
    pad = win // 2
    I2, J2, IJ = I * I, J * J, I * J
    I_sum, J_sum, I2_sum, J2_sum, IJ_sum = (F.conv3d(t, sum_filt, stride=1, padding=pad) for t in (I, J, I2, J2, IJ))
    n = float(win ** 3)
    u_I, u_J = I_sum / n, J_sum / n
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * n
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * n
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * n
    return float(-torch.mean(cross * cross / (I_var * J_var + 1e-5)))


def evaluate(path, surface=False):
    register = (load_dwp if args.method == 'dwp' else load_rcn)(path)
    rows = []
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            v = torch.from_numpy(d['vol'].astype(np.float32))[None, None].to(device)
            moved, flow = register(v)
            s = torch.from_numpy(d['seg'].astype(np.float32))[None, None].to(device)
            seg_w = warp_nearest(s, flow)[0, 0].cpu().numpy()
            if np.abs(seg_w - np.round(seg_w)).max() > 1e-4:
                sys.exit('[X] 搬完的標籤出現非整數值 —— 內插法錯了')
            seg_w = np.round(seg_w).astype(np.int32)
            per = {lab: dice(seg_w, atlas_seg, lab) for lab in LABELS}
            fl = flow[0].cpu().numpy()
            det = jacobian_det(fl)
            r = {'file': os.path.basename(f), 'per': per,
                 'dice_mean': float(np.nanmean([per[l] for l in LABELS])),
                 'jneg_pct': 100 * float((det <= 0).sum() / det.size),
                 'jneg_fg_pct': 100 * float((det[FG] <= 0).mean()),
                 'fold_fg': int((det[FG] <= 0).sum()),
                 'ncc_loss': ncc_loss(atlas_t, moved.float())}
            if surface:
                r['hd'] = {lab: hd95(seg_w == lab, atlas_seg == lab) for lab in LABELS}
                r['hd95_mean'] = float(np.nanmean([r['hd'][l] for l in LABELS]))
                r['sdlogj'] = float(np.log(np.clip(det[FG], 1e-9, 1e9)).std())
            rows.append(r)
    return rows


def ckpt_number(p):
    m = re.fullmatch(r'(\d+)\.pth?', os.path.basename(p))
    return int(m.group(1)) if m else None


print('方法 %s；%s 共 %d 位；裝置 %s' % (args.method, args.test_dir, len(files), device))
if args.model:
    t0 = time.time()
    rows = evaluate(args.model, args.surface)
    dm = np.array([r['dice_mean'] for r in rows])
    print('Dice %.4f ± %.4f；folding（腦內）每位 %.1f 個 voxel、%.4f%%；影像項 %.6f；%.0f 秒'
          % (dm.mean(), dm.std(), np.mean([r['fold_fg'] for r in rows]), np.mean([r['jneg_fg_pct'] for r in rows]),
             np.mean([r['ncc_loss'] for r in rows]), time.time() - t0))
    name = os.path.splitext(os.path.basename(args.model))[0]
    out = args.out_csv or os.path.join(os.path.dirname(args.model), 'dice_%s%s.csv' % (name, SUFFIX))
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'dice_mean', 'jneg_pct', 'jneg_fg_pct'] + ['label_%d' % l for l in LABELS])
        for r in rows:
            w.writerow([r['file'], '%.6f' % r['dice_mean'], '%.6f' % r['jneg_pct'], '%.6f' % r['jneg_fg_pct']]
                       + ['%.6f' % r['per'][l] for l in LABELS])
    print('CSV -> %s' % out)
    if args.surface:
        out = os.path.join(os.path.dirname(out), 'surface_%s%s.csv' % (name, SUFFIX))
        with open(out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'dice_mean', 'jneg_pct', 'jneg_fg_pct', 'hd95_mean', 'sdlogj']
                       + ['hd95_label_%d' % l for l in LABELS])
            for r in rows:
                w.writerow([r['file'], '%.6f' % r['dice_mean'], '%.6f' % r['jneg_pct'], '%.6f' % r['jneg_fg_pct'],
                            '%.4f' % r['hd95_mean'], '%.6f' % r['sdlogj']] + ['%.4f' % r['hd'][l] for l in LABELS])
        print('HD95 %.3f mm；SDlogJ %.4f；CSV -> %s'
              % (np.mean([r['hd95_mean'] for r in rows]), np.mean([r['sdlogj'] for r in rows]), out))
else:
    pts = sorted([p for p in glob.glob(os.path.join(args.model_dir, '*.pt*')) if ckpt_number(p) is not None],
                 key=ckpt_number)
    if not pts:
        sys.exit('[X] %s 裡沒有檢查點' % args.model_dir)
    pts = sorted(set(pts[::args.step] + [pts[-1]]), key=ckpt_number)
    curve = []
    for p in pts:
        t0 = time.time()
        rows = evaluate(p)
        e = ckpt_number(p)
        c = (e, np.mean([r['dice_mean'] for r in rows]), np.mean([r['jneg_pct'] for r in rows]),
             np.mean([r['jneg_fg_pct'] for r in rows]), np.mean([r['ncc_loss'] for r in rows]))
        curve.append(c)
        print('  %s  Dice %.4f  folding（腦內）%.4f%%  影像項 %.6f  %.0f 秒'
              % (os.path.basename(p), c[1], c[3], c[4], time.time() - t0), flush=True)
    out = args.out_csv or os.path.join(args.model_dir, 'dice_curve%s.csv' % SUFFIX)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['epoch', 'dice_mean', 'jneg_pct', 'jneg_fg_pct', 'ncc_loss'])
        w.writerows([[e, '%.6f' % d, '%.6f' % j, '%.6f' % jf, '%.6f' % nc] for e, d, j, jf, nc in curve])
    best = max(curve, key=lambda x: x[1])
    print('CSV -> %s；Dice 最高：%d（%.4f）' % (out, best[0], best[1]))
