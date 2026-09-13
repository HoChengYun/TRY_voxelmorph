# -*- coding: utf-8 -*-
"""量我們模型的推論時間，對照論文 Table I 的 GPU sec / CPU sec。

計時範圍跟論文一致：前處理之後、單一 pair 的 model(scan, atlas)（含積分），不含讀檔。
用法：python bench.py [cpu 線程數，預設 1] [CPU 量幾顆，預設 3]
"""
import os
import sys
import json
import glob
import time
import platform

import numpy as np

os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
os.environ.setdefault('VXM_BACKEND', 'pytorch')
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))   # 專案根目錄
sys.path.insert(0, os.path.join(ROOT, 'voxelmorph-code'))
import torch
import voxelmorph as vxm

HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_CPU = int(sys.argv[2]) if len(sys.argv) > 2 else 3

MODEL = os.path.join(ROOT, 'models', 'mix_exp1', '0230.pt')
atlas = np.load(os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
files = sorted(glob.glob(os.path.join(ROOT, 'data', 'mixed_preprocessed_v1', 'test', '*.npz')))
vols = [np.load(f)['vol'].astype(np.float32) for f in files]


def run(device, vs, warm):
    model = vxm.networks.VxmDense.load(MODEL, device).to(device).eval()
    a = torch.from_numpy(atlas)[None, None].to(device)
    ts = []
    with torch.no_grad():
        for i, v in enumerate(vs):
            x = torch.from_numpy(v)[None, None].to(device)
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x, a, registration=True)
            if device == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= warm:
                ts.append(dt)
            print('  %s %2d  %.3f s%s' % (device, i, dt, '  (暖機，不計)' if i < warm else ''), flush=True)
    return ts


out = {'model': 'mix_exp1/0230.pt', 'n_test': len(vols)}
cfg = torch.load(MODEL, map_location='cpu', weights_only=False)['config']
out['config'] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in cfg.items()}

if torch.cuda.is_available():
    g = run('cuda', vols[:2] + vols, warm=2)
    out['gpu'] = {'name': torch.cuda.get_device_name(0), 'n': len(g),
                  'mean': float(np.mean(g)), 'sd': float(np.std(g, ddof=1))}

torch.set_num_threads(THREADS)
c = run('cpu', vols[:1] + vols[:N_CPU], warm=1)
out['cpu'] = {'name': platform.processor(), 'threads': THREADS, 'n': len(c),
              'mean': float(np.mean(c)), 'sd': float(np.std(c, ddof=1)) if len(c) > 1 else 0.0}

json.dump(out, open(os.path.join(HERE, 'bench.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
