"""
確認 training sample 與 atlas 的 vol shape 是否一致
用法：python oasis/check_input_atlas_size.py
"""

import numpy as np
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLE_PATH = os.path.join(_HERE, "oasis_npz", "train", "OASIS_OAS1_0007_MR1.npz")
ATLAS_PATH  = os.path.join(_HERE, "..", "voxelmorph-code", "data", "atlas.npz")

def load_and_report(label, path):
    path = os.path.normpath(path)
    if not os.path.exists(path):
        print(f"[{label}] ❌ 找不到檔案：{path}")
        return None
    f    = np.load(path)
    keys = list(f.keys())
    vol  = f["vol"]
    print(f"[{label}]")
    print(f"  路徑  : {path}")
    print(f"  keys  : {keys}")
    print(f"  shape : {vol.shape}")
    print(f"  dtype : {vol.dtype}")
    print(f"  min   : {vol.min():.4f}  max : {vol.max():.4f}")
    return vol

print("=" * 55)
sample = load_and_report("Training Sample", SAMPLE_PATH)
print()
atlas  = load_and_report("Atlas",           ATLAS_PATH)
print("=" * 55)

if sample is not None and atlas is not None:
    if sample.shape == atlas.shape:
        print(f"✅ Shape 相同：{sample.shape}")
    else:
        print(f"❌ Shape 不同！")
        print(f"   Sample : {sample.shape}")
        print(f"   Atlas  : {atlas.shape}")
