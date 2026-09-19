# -*- coding: utf-8 -*-
"""把這份簡報要用的數字從原始 CSV 算出來 -> deck_data.json。投影片上不手打任何數字。

讀：
    data/mixed_preprocessed_v2/mixed_manifest.json          切分與來源
    models/{mix,tiger}_exp{2,3}/dice_<epoch>.csv            test 結果
    models/{mix,tiger}_exp{2,3}/dice_curve_val.csv          val 曲線（挑 epoch）
    models/mix_exp2/dice_baseline*.csv                      FreeSurfer 組起點
    models/tiger_exp2/dice_baseline*.csv                    tigerbx 組起點
    models/mix_exp2/cross_mix_tiger_exp2_exp3/*.csv         交叉評估
"""
import os
import csv
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(HERE, 'deck_data.json')

EPOCH = {'mix_exp2': '0240', 'mix_exp3': '0240', 'tiger_exp2': '0250', 'tiger_exp3': '0210'}


def rd(p):
    with open(os.path.join(ROOT, p), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return ({r['file'][:-4]: float(r['dice_mean']) for r in rows},
            {r['file'][:-4]: float(r.get('jneg_pct') or 0) for r in rows})


def curve(p):
    with open(os.path.join(ROOT, p), encoding='utf-8') as f:
        rows = sorted([(int(r['epoch']), float(r['dice_mean']), float(r['jneg_pct']))
                       for r in csv.DictReader(f)])
    best = max(rows, key=lambda r: r[1])
    return {'n': len(rows), 'best_epoch': best[0], 'best': best[1], 'jneg': best[2],
            'epoch0': rows[0][1]}


D = {}

# ── 資料與切分 ───────────────────────────────────────────────────────
man = json.load(open(os.path.join(ROOT, 'data/mixed_preprocessed_v2/mixed_manifest.json'),
                     encoding='utf-8'))['members']
split, ds = {}, {}
for k, v in man.items():
    split[v['split']] = split.get(v['split'], 0) + 1
    ds.setdefault(v['dataset'], {'train': 0, 'val': 0, 'test': 0})[v['split']] += 1
D['split'] = split
D['by_ds'] = ds
D['n_total'] = sum(split.values())

# ── 起點與四顆模型 ───────────────────────────────────────────────────
b_fs, _ = rd('models/mix_exp2/dice_baseline.csv')
b_tg, _ = rd('models/tiger_exp2/dice_baseline.csv')
K = sorted(b_fs)
D['n_test'] = len(K)
D['baseline'] = {'fs': float(np.mean([b_fs[k] for k in K])),
                 'tg': float(np.mean([b_tg[k] for k in K]))}

models = {}
for name in EPOCH:
    d, j = rd('models/%s/dice_%s.csv' % (name, EPOCH[name]))
    base = b_fs if name.startswith('mix') else b_tg
    a = np.array([d[k] for k in K])
    jj = np.array([j[k] for k in K])
    c = curve('models/%s/dice_curve_val.csv' % name)
    models[name] = {
        'epoch': EPOCH[name], 'mean': float(a.mean()), 'sd': float(a.std(ddof=1)),
        'jneg': float(jj.mean()), 'jneg_max': float(jj.max()),
        'gain': float(np.mean([d[k] - base[k] for k in K])),
        'val_best_epoch': c['best_epoch'], 'val_best': c['best'], 'val_epoch0': c['epoch0'],
        'worst_id': min(K, key=lambda x: d[x]), 'worst': float(min(d[k] for k in K)),
        'per_subject': {k: d[k] for k in K},
    }
D['models'] = models


def paired(a, b):
    v = np.array([a[k] - b[k] for k in K])
    return {'mean': float(v.mean()), 'se': float(v.std(ddof=1) / len(v) ** .5),
            'win': int((v > 0).sum()), 'n': len(v)}


ps = {}
ps['fs_v3_minus_v2'] = paired(models['mix_exp3']['per_subject'], models['mix_exp2']['per_subject'])
ps['tg_v3_minus_v2'] = paired(models['tiger_exp3']['per_subject'], models['tiger_exp2']['per_subject'])
for n2, n3 in (('mix_exp2', 'tiger_exp2'), ('mix_exp3', 'tiger_exp3')):
    g_m = {k: models[n2]['per_subject'][k] - b_fs[k] for k in K}
    g_t = {k: models[n3]['per_subject'][k] - b_tg[k] for k in K}
    ps['gain_%s_minus_%s' % (n3, n2)] = paired(g_t, g_m)
D['paired'] = ps

# ── 交叉評估 ─────────────────────────────────────────────────────────
X = 'models/mix_exp2/cross_mix_tiger_exp2_exp3/'
cross = {}
for f, tag in (('tiger_exp2_on_freesurfer.csv', 'tiger_exp2_on_fs'),
               ('tiger_exp3_on_freesurfer.csv', 'tiger_exp3_on_fs'),
               ('mix_exp2_on_tigerbx.csv', 'mix_exp2_on_tg'),
               ('mix_exp3_on_tigerbx.csv', 'mix_exp3_on_tg')):
    d, j = rd(X + f)
    cross[tag] = {'mean': float(np.mean([d[k] for k in K])),
                  'jneg': float(np.mean([j[k] for k in K])), 'per_subject': d}
D['cross'] = cross
D['cross_paired'] = {
    'fs_v2': paired(models['mix_exp2']['per_subject'], cross['tiger_exp2_on_fs']['per_subject']),
    'fs_v3': paired(models['mix_exp3']['per_subject'], cross['tiger_exp3_on_fs']['per_subject']),
    'tg_v2': paired(models['tiger_exp2']['per_subject'], cross['mix_exp2_on_tg']['per_subject']),
    'tg_v3': paired(models['tiger_exp3']['per_subject'], cross['mix_exp3_on_tg']['per_subject']),
}

for m in D['models'].values():
    m.pop('per_subject')
for c in D['cross'].values():
    c.pop('per_subject')

json.dump(D, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('ok ->', OUT)
print('切分', D['split'], '｜起點 FS %.4f / TG %.4f' % (D['baseline']['fs'], D['baseline']['tg']))
for n, m in D['models'].items():
    print('  %-11s val 最佳 %d -> test %.4f（貢獻 +%.3f，折疊 %.3f%%）'
          % (n, m['val_best_epoch'], m['mean'], m['gain'], m['jneg']))
