# -*- coding: utf-8 -*-
"""把簡報要用的數字全部從原始 CSV 算出來，存成 deck_data.json。

投影片裡的每個數字都從這裡來，不手打 —— 避免簡報跟實際結果對不上。
"""
import io
import os
import csv
import json
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))   # 專案根目錄
OUT = os.path.join(HERE, 'deck_data.json')


def rows(p):
    return list(csv.DictReader(io.open(ROOT + '\\' + p, encoding='utf-8')))


def bykey(p):
    return {r['file']: r for r in rows(p)}


man = json.load(io.open(ROOT + r'\data\mixed_preprocessed_v1\mixed_manifest.json', encoding='utf-8'))['members']

dem = {}
for ds in ('ASD', 'DGM', 'VNT'):
    hdr = None
    for l in io.open(ROOT + r'\data\%s_data\fs_stats\demographics.tsv' % ds, encoding='utf-8'):
        if l.startswith('#') or not l.strip():
            continue
        p = l.rstrip('\n').split('\t')
        if p[0] == 'subject':
            hdr = p
            continue
        dem[p[0]] = dict(zip(hdr, p))

D = {}

# ── 曲線 ──────────────────────────────────────────────────────────────
for exp in ('mix_exp1', 'tiger_exp1', 'asd_exp1'):
    r = rows(r'models\%s\dice_curve.csv' % exp)
    D['curve_' + exp] = [[int(x['epoch']), float(x['dice_mean']), float(x['jneg_pct'])] for x in r]

# ── 逐顆 ──────────────────────────────────────────────────────────────
fb = bykey(r'models\mix_exp1\dice_baseline.csv')
fa = bykey(r'models\mix_exp1\dice_0230.csv')
tb = bykey(r'models\tiger_exp1\dice_baseline.csv')
ta = bykey(r'models\tiger_exp1\dice_0240.csv')
ks = sorted(fa)
assert set(ks) == set(ta) == set(fb) == set(tb)

subj = []
for k in ks:
    s = k[:-4]
    subj.append({
        'id': s, 'ds': man[k]['dataset'],
        'age': float(dem[s]['age']) if s in dem else None,
        'fs_base': float(fb[k]['dice_mean']), 'fs_after': float(fa[k]['dice_mean']),
        'tg_base': float(tb[k]['dice_mean']), 'tg_after': float(ta[k]['dice_mean']),
    })
D['subjects'] = subj

FB = np.array([x['fs_base'] for x in subj]); FA = np.array([x['fs_after'] for x in subj])
TB = np.array([x['tg_base'] for x in subj]); TA = np.array([x['tg_after'] for x in subj])
dg = (TA - TB) - (FA - FB)
D['summary'] = {
    'n': len(subj),
    'fs_base': FB.mean(), 'fs_after': FA.mean(), 'fs_gain': (FA - FB).mean(),
    'tg_base': TB.mean(), 'tg_after': TA.mean(), 'tg_gain': (TA - TB).mean(),
    'gain_diff': dg.mean(), 'gain_diff_sem': dg.std(ddof=1) / len(dg) ** 0.5,
    'gain_diff_tg_better': int((dg > 0).sum()),
    'fs_base_sd': FB.std(ddof=1), 'fs_after_sd': FA.std(ddof=1),
    'tg_base_sd': TB.std(ddof=1), 'tg_after_sd': TA.std(ddof=1),
    'r_gain_vs_base_fs': float(np.corrcoef(FB, FA - FB)[0, 1]),
    'sem_fs_base': FB.std(ddof=1) / len(FB) ** 0.5,
}

# 標準差用論文 Table I 的定義：「across structures and subjects」
# = 28 位 × 30 個結構的所有 Dice 攤平後的標準差（不是 28 個受試者平均值的標準差）
def pooled_sd(d):
    v = [float(d[k][c]) for k in ks for c in d[k]
         if c.startswith('label_') and d[k][c] not in ('', 'nan')]
    return float(np.std(v, ddof=1)), len(v)


D['sd_pooled'] = {}
for tag, d in (('fs_base', fb), ('fs_after', fa), ('tg_base', tb), ('tg_after', ta)):
    D['sd_pooled'][tag], D['sd_pooled']['n_' + tag] = pooled_sd(d)

# 折疊 voxel 數：jneg_pct 是佔整個 192×224×192 的百分比，全部 0 就是 0 個
for tag, d in (('fs', fa), ('tg', ta)):
    D['jneg_max_' + tag] = max(float(d[k]['jneg_pct']) for k in ks)

# 推論時間（bench.py 量的；沒量過就不放）
try:
    D['bench'] = json.load(io.open(OUT.replace('deck_data.json', 'bench.json'), encoding='utf-8'))
except FileNotFoundError:
    D['bench'] = None

# 作者的預訓練模型跑作者的資料（OASIS 4 位，手冊 §19）：test_dice.py 算的 CSV
try:
    ab = bykey(r'models\author_exp1\dice_baseline.csv')
    aa = bykey(r'models\author_exp1\dice_vxm_dense_brain_T1_3D_mse.csv')
    au = [{'id': k[:-4], 'base': float(ab[k]['dice_mean']), 'after': float(aa[k]['dice_mean']),
           'jneg': float(aa[k]['jneg_pct'])} for k in sorted(aa)]
    D['author_oasis'] = {'subjects': au,
                         'base': float(np.mean([x['base'] for x in au])),
                         'after': float(np.mean([x['after'] for x in au])),
                         'n_labels': sum(1 for c in aa[sorted(aa)[0]] if c.startswith('label_'))}
except FileNotFoundError:
    D['author_oasis'] = None

# 高原
for exp in ('mix_exp1', 'tiger_exp1'):
    c = D['curve_' + exp]
    pl = [d for e, d, _ in c if e >= 90]
    best = max(c, key=lambda x: x[1])
    D['plateau_' + exp] = {'mean': float(np.mean(pl)), 'range': max(pl) - min(pl),
                           'best_epoch': best[0], 'best': best[1]}

# ── 分資料集 ─────────────────────────────────────────────────────────
D['by_ds'] = {}
for ds in ('ASD', 'DGM', 'VNT'):
    idx = [i for i, x in enumerate(subj) if x['ds'] == ds]
    D['by_ds'][ds] = {'n': len(idx),
                      'fs_base': FB[idx].mean(), 'fs_after': FA[idx].mean(),
                      'tg_base': TB[idx].mean(), 'tg_after': TA[idx].mean(),
                      'gain_diff': dg[idx].mean()}

# ── 分結構 ───────────────────────────────────────────────────────────
FSN = {2: '左大腦白質', 3: '左大腦皮質', 4: '左側腦室', 7: '左小腦白質', 8: '左小腦皮質',
       10: '左視丘', 11: '左尾狀核', 12: '左殼核', 13: '左蒼白球', 14: '第三腦室',
       15: '第四腦室', 16: '腦幹', 17: '左海馬迴', 18: '左杏仁核', 24: 'CSF',
       28: '左腹側間腦', 31: '左脈絡叢', 41: '右大腦白質', 42: '右大腦皮質', 43: '右側腦室',
       46: '右小腦白質', 47: '右小腦皮質', 49: '右視丘', 50: '右尾狀核', 51: '右殼核',
       52: '右蒼白球', 53: '右海馬迴', 54: '右杏仁核', 60: '右腹側間腦', 63: '右脈絡叢'}
labs = [int(c[6:]) for c in fa[ks[0]] if c.startswith('label_')]


def per(d, l, excl=None):
    v = [float(d[k]['label_%d' % l]) for k in ks
         if k != excl and d[k]['label_%d' % l] not in ('', 'nan')]
    return float(np.mean(v))


D['per_struct'] = [{'label': l, 'name': FSN.get(l, str(l)),
                    'fs_base': per(fb, l), 'fs_after': per(fa, l),
                    'tg_base': per(tb, l), 'tg_after': per(ta, l)} for l in labs]

# A0131
D['a0131'] = {}
for l in (18, 54, 12, 53):
    D['a0131'][FSN[l]] = {'fs': float(fa['A0131.npz']['label_%d' % l]),
                         'tg': float(ta['A0131.npz']['label_%d' % l]),
                         'fs_others': per(fa, l, 'A0131.npz'),
                         'tg_others': per(ta, l, 'A0131.npz')}

# ── 同人偵測（先前實測，數字直接記錄）───────────────────────────────
D['dup_dist'] = {
    'ASD': {'pairs': 13366, 'median': 0.6631, 'p99': 0.7208, 'max': 0.7473},
    'DGM': {'pairs': 1431, 'median': 0.6563, 'p99': 0.7218, 'max': 0.8561},
    'VNT': {'pairs': 2278, 'median': 0.6533, 'p99': 0.7191, 'max': 0.7371},
}
D['dup_pairs'] = [
    ['A0131 / YT13', 0.9793, '同一次掃描（重複匯出）'],
    ['D015 / D037', 0.8561, '同一人，相隔 3 個月'],
    ['D038 / DGM002', 0.8528, '同一人，相隔 9 個月'],
    ['A0131 / A0132', 0.7318, '同一人（5 歲），相隔 23 天'],
    ['D018 / DGM001', 0.6995, '不同人（DICOM 定案）'],
    ['VNT027 / VNT028', 0.6609, '不同人（複製登錄）'],
]

# 人口學
D['demo'] = {}
for ds in ('ASD', 'DGM', 'VNT'):
    ages = [float(v['age']) for k, v in dem.items()
            if any(m.get('dataset') == ds and kk[:-4] == k for kk, m in man.items())]
    D['demo'][ds] = {'n': len(ages), 'age_median': float(np.median(ages)),
                     'age_min': min(ages), 'age_max': max(ages),
                     'under18': int(sum(a < 18 for a in ages))}

# 舊模型 asd_exp1（167 顆清單、17 顆 test）—— 第 4 頁「洩漏有沒有灌水」用
r17 = bykey(r'models\asd_exp1\dice_0190.csv')
v17 = {k: float(x['dice_mean']) for k, x in r17.items()}
a17 = np.array(list(v17.values()))
D['old17'] = {'n': len(a17), 'mean': a17.mean(), 'sem': a17.std(ddof=1) / len(a17) ** 0.5,
              'a0131': v17['A0131.npz'],
              'mean_excl': float(np.mean([x for k, x in v17.items() if k != 'A0131.npz']))}

json.dump(D, io.open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1,
          default=float)
print('ok ->', OUT)
s = D['summary']
print('FS  %.4f -> %.4f (%+.4f)' % (s['fs_base'], s['fs_after'], s['fs_gain']))
print('TG  %.4f -> %.4f (%+.4f)' % (s['tg_base'], s['tg_after'], s['tg_gain']))
print('diff %+.4f  sem %.4f  tg better %d/28' % (s['gain_diff'], s['gain_diff_sem'], s['gain_diff_tg_better']))
print('demo', D['demo'])
