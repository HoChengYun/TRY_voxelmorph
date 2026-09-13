"""
把 neurite-oasis 的一位受試者轉成 ASD/visualize_dice.py 吃的 npz，拿來看作者模型在作者資料上的樣子。

oasis_npz/ 裡其實有 seg，但那是 neurite-oasis 的 seg35：0–35 的連續編號，
不是 FreeSurfer 的編號（17 = 左海馬迴那種）。repo 的 atlas.npz 和 labels.npz 用的是
FreeSurfer 編號，直接比會對錯結構。這裡照 scripts/torch/test_oasis.py 的 SEG35_TO_FS 換過去。

seg35 沒有 CSF（FreeSurfer 24），但 atlas.npz 有。評估清單不拿掉 24 的話，
CSF 會因為受試者那邊永遠是空的而被算成 0 分，平均被拉低 —— 所以另存一份不含 CSF 的清單。

用法（--subject 可以給原始資料夾，也可以給 oasis_npz 裡的 npz —— 兩者內容相同）
    python oasis\\prepare_author_check.py --subject oasis\\OASIS_OAS1_0050_MR1 --out-dir oasis\\author_check
    python oasis\\prepare_author_check.py --subject oasis\\oasis_npz\\test\\OASIS_OAS1_0277_MR1.npz --out-dir oasis\\author_check

輸出
    <out-dir>/<受試者>.npz          vol = aligned_norm、seg = aligned_seg35 換成 FreeSurfer 編號
    <out-dir>/labels_eval.npz       評估用的結構（labels.npz 裡、受試者也有的那些）
"""
import os
import sys
import argparse

import numpy as np
import nibabel as nib

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 照抄 voxelmorph-code/scripts/torch/test_oasis.py
SEG35_TO_FS = {
     0:  0,   1:  2,   2:  3,   3:  4,   4:  5,
     5:  7,   6:  8,   7: 10,   8: 11,   9: 12,
    10: 13,  11: 14,  12: 15,  13: 16,  14: 17,
    15: 18,  16: 26,  17: 28,  18: 30,  19: 31,
    20: 41,  21: 42,  22: 43,  23: 44,  24: 46,
    25: 47,  26: 49,  27: 50,  28: 51,  29: 52,
    30: 53,  31: 54,  32: 58,  33: 60,  34: 62,
    35: 63,
}

ap = argparse.ArgumentParser()
ap.add_argument('--subject', '--subject-dir', dest='subject', required=True,
                help='neurite-oasis 的受試者資料夾（有 aligned_norm / aligned_seg35），'
                     '或 oasis_npz 裡的 npz（vol = aligned_norm、seg = aligned_seg35）')
ap.add_argument('--out-dir', required=True)
ap.add_argument('--atlas', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'atlas.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
args = ap.parse_args()

sd = os.path.normpath(args.subject)
if sd.endswith('.npz'):
    name = os.path.basename(sd)[:-4]
    _d = np.load(sd)
    vol, s35 = _d['vol'].astype(np.float32), _d['seg'].astype(np.int32)
else:
    name = os.path.basename(sd.rstrip('\\/'))
    vol = np.asarray(nib.load(os.path.join(sd, 'aligned_norm.nii.gz')).dataobj).astype(np.float32)
    s35 = np.asarray(nib.load(os.path.join(sd, 'aligned_seg35.nii.gz')).dataobj).astype(np.int32)
if s35.max() > 35:
    sys.exit('[X] seg 最大值 %d > 35：這不是 seg35（可能已經換過編號了），不要重複換' % s35.max())

extra = sorted(set(np.unique(s35).tolist()) - set(SEG35_TO_FS))
if extra:
    sys.exit('[X] seg35 有對照表沒有的值 %s' % extra)
seg = np.zeros_like(s35, dtype=np.int16)
for a, b in SEG35_TO_FS.items():
    seg[s35 == a] = b

atlas = np.load(args.atlas)
if vol.shape != atlas['vol'].shape:
    sys.exit('[X] 大小 %s 跟 atlas %s 不同' % (vol.shape, atlas['vol'].shape))
m1, m2 = vol > 0.05, atlas['vol'] > 0.05
print('受試者 %s  %s  值域 [%.3f, %.3f]' % (name, vol.shape, vol.min(), vol.max()))
print('跟 atlas 的腦遮罩重疊 %.3f（高代表已在同一空間，不用再對位）' % (2 * (m1 & m2).sum() / (m1.sum() + m2.sum())))

labels = np.load(args.labels)['labels'].astype(int).tolist()
have = set(np.unique(seg).tolist())
keep = [l for l in labels if l in have]
drop = [l for l in labels if l not in have]
print('評估結構 %d 個；受試者沒有而拿掉的：%s' % (len(keep), drop or '無'))

os.makedirs(args.out_dir, exist_ok=True)
out = os.path.join(args.out_dir, name + '.npz')
np.savez_compressed(out, vol=vol, seg=seg)
np.savez(os.path.join(args.out_dir, 'labels_eval.npz'), labels=np.array(keep))
print('[v] %s' % out)
print('[v] %s' % os.path.join(args.out_dir, 'labels_eval.npz'))
