# -*- coding: utf-8 -*-
"""同一個人，清掉去顱骨的殘渣前後各跑一次，看非線性對位差多少。

為什麼要這樣做
--------------
「殘渣多的人 Dice 比較低嗎」用人跟人比會被頭大小、影像品質汙染。
這支改成配對比較：同一顆腦、同一顆模型，唯一差別是那塊殘渣在不在。

殘渣的定義：影像還亮著、但離 FreeSurfer 標到的腦組織 10mm 以外。
10mm 以外不可能是腦或緊貼腦的硬腦膜，所以清掉它不會動到皮質
（這跟「把腦罩往內侵蝕」不同，侵蝕會切進皮質）。標籤完全不動。

⚠️ 線性對位是在殘渣還在的時候算好的，這支不重跑 affine，
   量到的是「殘渣對非線性那一步的影響」，不含它對 affine 的影響。
"""
import os
import sys
import csv
import glob
import argparse
import numpy as np
import torch
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'voxelmorph-code'))

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--test-dir', required=True)
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--far-mm', type=float, default=10.0, help='離腦組織幾 mm 以外算殘渣')
ap.add_argument('--out-csv', required=True)
ap.add_argument('--gpu', default='0')
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = np.load(args.labels)['labels'] if 'labels' in np.load(args.labels) else np.load(args.labels)[np.load(args.labels).files[0]]
LABELS = np.asarray(LABELS).astype(int).tolist()
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
atlas_t = torch.from_numpy(atlas_vol)[None, None].to(device)

warp_nn = vxm.torch.layers.SpatialTransformer(atlas_vol.shape, mode='nearest').to(device)
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device).eval()


def dice_mean(seg_w):
    v = []
    for lab in LABELS:
        x, y = (seg_w == lab), (atlas_seg == lab)
        s = x.sum() + y.sum()
        v.append(np.nan if s == 0 else 2.0 * (x & y).sum() / s)
    return float(np.nanmean(np.array(v, dtype=float)))


def run(vol, seg):
    v = torch.from_numpy(vol)[None, None].to(device)
    _, flow = model(v, atlas_t, registration=True)
    s = torch.from_numpy(seg.astype(np.float32))[None, None].to(device)
    w = warp_nn(s, flow)[0, 0].cpu().numpy()
    if float(np.abs(w - np.round(w)).max()) > 1e-4:
        sys.exit('[X] 搬完的標籤出現非整數值 —— 內插法錯了')
    return dice_mean(np.round(w).astype(np.int32))


files = sorted(glob.glob(os.path.join(os.path.normpath(args.test_dir), '*.npz')))
print('模型 %s｜%d 筆｜殘渣定義 >%.0fmm｜裝置 %s'
      % (os.path.basename(args.model), len(files), args.far_mm, device))

rows = []
with torch.no_grad():
    for f in files:
        d = np.load(f)
        vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
        junk = (vol > 0.02) & (ndi.distance_transform_edt(seg == 0) > args.far_mm)
        clean = vol.copy()
        clean[junk] = 0.0
        a, b = run(vol, seg), run(clean, seg)
        rows.append(dict(subject=os.path.basename(f)[:-4], junk_vox=int(junk.sum()),
                         dice_dirty=round(a, 6), dice_clean=round(b, 6), delta=round(b - a, 6)))
        print('  %-10s 殘渣 %6d 顆｜留著 %.4f → 清掉 %.4f｜%+.4f'
              % (rows[-1]['subject'], rows[-1]['junk_vox'], a, b, b - a), flush=True)

os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
with open(args.out_csv, 'w', newline='', encoding='utf-8') as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

dl = np.array([r['delta'] for r in rows])
jk = np.array([r['junk_vox'] for r in rows])
print('\n%d 位：清掉殘渣後 Dice 平均 %+.4f（中位數 %+.4f），變好的 %d／%d'
      % (len(rows), dl.mean(), np.median(dl), int((dl > 0).sum()), len(dl)))
print('殘渣越多、清掉的效果越大嗎：相關 %+.3f' % np.corrcoef(jk, dl)[0, 1])
print('->', args.out_csv)
