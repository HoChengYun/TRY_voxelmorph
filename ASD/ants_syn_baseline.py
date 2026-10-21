# -*- coding: utf-8 -*-
"""傳統方法對照：ANTs SyN 對 test 51 位做「受試者 → atlas」，用跟 test_dice.py 一模一樣的 Dice、folding 評估。

參數照 VoxelMorph 論文（Balakrishnan et al., TMI 2019）§V-A-3 的 ANTs SyN 設定（作者在多個資料集上掃參數挑出來的）：
  「SyN step size of 0.25, Gaussian parameters (9, 0.2), at three scales with at most 201 iterations each」，相似度 cross-correlation
→ ANTsPy：type_of_transform='SyNOnly'、grad_step=0.25、flow_sigma=9、total_sigma=0.2、syn_metric='CC'、
  syn_sampling=4（CC 半徑 4 = 窗格 9，同 VoxelMorph 的 NCC）、reg_iterations=(201, 201, 201)（縮小 4、2、1 倍；平滑 2、1、0）
  CC 半徑論文沒寫，用 ANTs 慣用的 4。

🔴 initial_transform='Identity'：ANTsPy 0.6.3 的 SyNOnly 不給也是 Identity（Affine 等其他類型才會先對齊重心
   -r [fixed,moving,1]），明寫是保險：我們的資料已經做過 affine，SyN 只做非線性的部分，跟 VoxelMorph 拿到的輸入一樣。

評估（同 ASD/test_dice.py）：
  - 受試者的 seg 用 SyN 的形變、最近鄰搬到 atlas 空間，跟 atlas seg 比 30 個結構的 Dice
  - folding：讀 SyN 的位移場（定義在 atlas 格子上，單位 = voxel，因為影像間距 1、方向單位矩陣），
    用跟 test_dice.py 一樣的中央差分算 Jacobian；jneg_pct 分母整個影像、jneg_fg_pct 分母 atlas 非背景（atlas_vol > 0）
  - --check：第一位另外驗證位移場的方向與座標慣例（自己用位移場搬標籤，要跟 ANTs 搬的一樣；Jacobian 要跟 ANTs 算的一樣）

輸出 models\\ants_syn\\dice_syn_<資料夾名>.csv（file, dice_mean, jneg_pct, jneg_fg_pct, sec, label_<編號>…），
每做完一位就寫一列，中斷後再跑會跳過已經有的。

視覺化用（2026-10-18 加）：--subjects 只做指定的幾位、--save-dir 把結果存成 <save-dir>\\<受試者>.npz
（warped：搬過去的影像（線性）、seg_warped：搬過去的標籤（最近鄰）、flow：位移場 (3, X, Y, Z)、det：Jacobian、sec）。
已經在 CSV 裡的受試者，給 --save-dir 時照樣重跑（只為了存結果），不會再寫一列進 CSV，會印出跟 CSV 那列的 Dice 差多少。

用法
  python ASD\\ants_syn_baseline.py --test-dir data\\mixed_preprocessed_v2\\test --max-subjects 1 --check   # 先試一位
  python ASD\\ants_syn_baseline.py --test-dir data\\mixed_preprocessed_v2\\test                            # 51 位
  python ASD\\ants_syn_baseline.py --test-dir data\\mixed_preprocessed_v2\\test --subjects T054 D031 VNT045 sub-0043 ^
      --save-dir models\\ants_syn\\vis                                                                       # 視覺化用
"""
import os
import sys
import csv
import glob
import time
import shutil
import argparse
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--test-dir', required=True)
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--out-dir', default=os.path.join(ROOT, 'models', 'ants_syn'))
ap.add_argument('--threads', type=int, default=os.cpu_count(), help='ITK 執行緒數（預設全部核心）')
ap.add_argument('--iterations', type=int, nargs='+', default=[201, 201, 201], help='每個尺度最多幾次（論文 201 201 201）')
ap.add_argument('--max-subjects', type=int, default=0)
ap.add_argument('--check', action='store_true', help='第一位驗證位移場慣例')
ap.add_argument('--subjects', nargs='+', help='只做這幾位（檔名不含 .npz）')
ap.add_argument('--save-dir', help='把搬過去的影像、標籤、位移場、Jacobian 存成 <save-dir>/<受試者>.npz')
args = ap.parse_args()
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(args.threads)

import ants                                                    # noqa: E402
from scipy import ndimage                                       # noqa: E402

LABELS = np.load(args.labels)['labels'].astype(int).tolist()
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
FG = atlas_vol > 0
fixed = ants.from_numpy(atlas_vol)                              # 間距 1、原點 0、方向單位矩陣 → 物理座標 = voxel 座標


def jacobian_det(flow):
    """det(I + ∇u)，中央差分（同 test_dice.py）。flow: (3, X, Y, Z)，單位 voxel。"""
    d = [[np.gradient(flow[c], axis=a) for a in range(3)] for c in range(3)]
    j11, j12, j13 = 1 + d[0][0], d[0][1], d[0][2]
    j21, j22, j23 = d[1][0], 1 + d[1][1], d[1][2]
    j31, j32, j33 = d[2][0], d[2][1], 1 + d[2][2]
    return (j11 * (j22 * j33 - j23 * j32)
            - j12 * (j21 * j33 - j23 * j31)
            + j13 * (j21 * j32 - j22 * j31))


def dice(a, b, lab):
    x, y = (a == lab), (b == lab)
    s = x.sum() + y.sum()
    return np.nan if s == 0 else 2.0 * (x & y).sum() / s


files = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))
if args.subjects:
    want = set(args.subjects)
    files = [f for f in files if os.path.basename(f)[:-4] in want]
    missing = want - {os.path.basename(f)[:-4] for f in files}
    if missing:
        sys.exit('[X] %s 裡找不到：%s' % (args.test_dir, ' '.join(sorted(missing))))
if args.max_subjects:
    files = files[:args.max_subjects]
os.makedirs(args.out_dir, exist_ok=True)
split = os.path.basename(os.path.normpath(args.test_dir))
out = os.path.join(args.out_dir, 'dice_syn_%s.csv' % split)
done, done_dice = set(), {}
if os.path.exists(out):
    with open(out, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            done.add(row['file'])
            done_dice[row['file']] = float(row['dice_mean'])
if args.save_dir:
    os.makedirs(args.save_dir, exist_ok=True)
fields = ['file', 'dice_mean', 'jneg_pct', 'jneg_fg_pct', 'sec'] + ['label_%d' % l for l in LABELS]
new_file = not os.path.exists(out)
print('ANTs SyN（SyNOnly、CC 半徑 4、step 0.25、Gaussian (9, 0.2)、迭代 %s）；%d 個執行緒；%d 位（已完成 %d）'
      % ('x'.join(map(str, args.iterations)), args.threads, len(files), len(done & {os.path.basename(f) for f in files})),
      flush=True)

for k, f in enumerate(files):
    name = os.path.basename(f)
    rerun = name in done                      # 已經在 CSV 裡：只有要存結果時才重跑，而且不再寫進 CSV
    if rerun and not args.save_dir:
        continue
    d = np.load(f)
    vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
    moving = ants.from_numpy(vol)
    tmp = tempfile.mkdtemp(prefix='syn_')
    t0 = time.time()
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyNOnly', initial_transform='Identity',
                            grad_step=0.25, flow_sigma=9, total_sigma=0.2, syn_metric='CC', syn_sampling=4,
                            reg_iterations=tuple(args.iterations), outprefix=os.path.join(tmp, 'syn_'))
    sec = time.time() - t0
    seg_w = ants.apply_transforms(fixed=fixed, moving=ants.from_numpy(seg.astype(np.float32)),
                                  transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor').numpy()
    seg_w = np.round(seg_w).astype(np.int32)
    warp = [t for t in reg['fwdtransforms'] if t.endswith('Warp.nii.gz')]
    assert len(warp) == 1, reg['fwdtransforms']
    flow = np.moveaxis(ants.image_read(warp[0]).numpy(), -1, 0)     # (3, X, Y, Z)
    det = jacobian_det(flow)
    if args.check and k == 0:
        # 1. 自己用位移場（最近鄰）搬標籤，要跟 ANTs 搬的一樣 → 位移場的方向（定義在 atlas 格子上、指向受試者）與軸順序都對
        grid = np.stack(np.meshgrid(*[np.arange(n) for n in seg.shape], indexing='ij'))
        mine = ndimage.map_coordinates(seg, grid + flow, order=0, mode='nearest')
        same = (mine == seg_w).mean()
        # 2. Jacobian 跟 ANTs 自己算的比（ANTs 用的差分方式不同，看相關與 ≤ 0 的點數）
        jd = ants.create_jacobian_determinant_image(fixed, warp[0], do_log=False).numpy()
        print('  [check] 自己搬的標籤跟 ANTs 搬的相同比例 %.5f；Jacobian 相關 %.4f；|J| ≤ 0：我們 %d、ANTs %d；'
              '仿射檔 %s' % (same, np.corrcoef(det[FG], jd[FG])[0, 1], int((det <= 0).sum()), int((jd <= 0).sum()),
                         [os.path.basename(t) for t in reg['fwdtransforms'] if not t.endswith('Warp.nii.gz')]), flush=True)
    per = {lab: dice(seg_w, atlas_seg, lab) for lab in LABELS}
    r = {'file': name, 'dice_mean': float(np.nanmean([per[l] for l in LABELS])),
         'jneg_pct': 100 * float((det <= 0).mean()), 'jneg_fg_pct': 100 * float((det[FG] <= 0).mean()), 'sec': sec}
    r.update({'label_%d' % l: per[l] for l in LABELS})
    if args.save_dir:
        warped = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'],
                                       interpolator='linear').numpy().astype(np.float32)
        np.savez_compressed(os.path.join(args.save_dir, name), warped=warped, seg_warped=seg_w.astype(np.int16),
                            flow=flow.astype(np.float32), det=det.astype(np.float32), sec=sec)
        print('  存 -> %s' % os.path.join(args.save_dir, name), flush=True)
    if rerun:
        print('  （已在 CSV：這次 Dice %.4f，CSV 那列 %.4f，差 %+.4f；不再寫進 CSV）'
              % (r['dice_mean'], done_dice[name], r['dice_mean'] - done_dice[name]), flush=True)
    else:
        with open(out, 'a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow({k2: ('%.6f' % v if isinstance(v, float) else v) for k2, v in r.items()})
    shutil.rmtree(tmp, ignore_errors=True)
    print('  [%d/%d] %-22s Dice %.4f  folding（非背景）%d 個  %.0f 秒'
          % (k + 1, len(files), name, r['dice_mean'], int((det[FG] <= 0).sum()), sec), flush=True)
print('  CSV ->', out)
