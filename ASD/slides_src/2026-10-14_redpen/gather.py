# -*- coding: utf-8 -*-
"""10/14 meeting 簡報的數字，全部從原始 CSV 算 -> deck_data.json。投影片上不手打數字。

讀：
    models/<exp>/dice_<epoch>.csv                 test 結果（每個實驗資料夾只有一份 4 位數 epoch 的 CSV）
    models/mix_exp2/dice_baseline.csv             起點（只做線性對位）
    models/folding_check/                         ① 擠爆的位置（ASD/check_folding.py 產生）
    models/skullstrip_check/                      ③④ 殘留（skullstrip_label_dice*.py、check_top_residue.py 產生）

還沒跑完的實驗（mix_exp6、mix_exp7、mix_wide_vel）沒有 test CSV，標成 pending，簡報上顯示「跑中」。
結果帶回來、放進 models/<exp>/ 之後重跑這支就會補上。
"""
import os
import sys
import csv
import glob
import json
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(HERE, 'deck_data.json')
J = lambda *p: os.path.join(ROOT, *p)
sys.path.insert(0, J('ASD'))

# 每顆的設定（照操作單，不是從 CSV 來）：版本、實際平滑權重（λ × int_downsize）、寬度、積分解析度
CFG = {
    'mix_exp2': ('速度場', 2.0, '預設', '半解析度'),
    'mix_exp5': ('速度場', 2.0, '預設', '全尺寸'),
    'mix_exp6': ('速度場', 1.0, '預設', '全尺寸'),
    'mix_exp7': ('速度場', 0.5, '預設', '全尺寸'),
    'mix_exp4': ('位移場', 2.0, '預設', '全尺寸'),
    'mix_exp3': ('位移場', 1.0, '預設', '全尺寸'),
    'mix_wide': ('位移場', 1.0, '加寬 2 倍', '全尺寸'),
    'mix_wide_vel': ('速度場', 1.0, '加寬 2 倍', '全尺寸'),
}


def rd(p):
    with open(p, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return ({r['file'][:-4]: float(r['dice_mean']) for r in rows},
            {r['file'][:-4]: float(r.get('jneg_pct') or 0) for r in rows})


def test_csv(exp):
    """models/<exp>/ 底下的 test 結果。只有一份就用它；有好幾份就用驗證集最好的那個 epoch。"""
    fs = sorted(glob.glob(J('models', exp, 'dice_[0-9][0-9][0-9][0-9].csv')))
    if len(fs) <= 1:
        return fs[0] if fs else None
    with open(J('models', exp, 'dice_curve_val.csv'), encoding='utf-8') as f:
        best = max(csv.DictReader(f), key=lambda r: float(r['dice_mean']))
    p = J('models', exp, 'dice_%04d.csv' % int(best['epoch']))
    if p not in fs:
        raise SystemExit('[X] %s 有好幾份 test CSV，但沒有驗證集最好的 epoch %s' % (exp, best['epoch']))
    return p


D = {}

# ── 模型 ─────────────────────────────────────────────────────────────
base, _ = rd(J('models', 'mix_exp2', 'dice_baseline.csv'))
K = sorted(base)
D['n_test'] = len(K)
D['baseline'] = float(np.mean([base[k] for k in K]))
models, per = {}, {}
for e, (ver, wt, width, res) in CFG.items():
    m = {'version': ver, 'weight': wt, 'width': width, 'res': res}
    p = test_csv(e)
    if p is None:
        m['status'] = 'pending'
    else:
        d, j = rd(p)
        a = np.array([d[k] for k in K])
        m.update({'status': 'done', 'epoch': os.path.basename(p)[5:9], 'mean': float(a.mean()),
                  'jneg': float(np.mean([j[k] for k in K])),
                  'gain': float(np.mean([d[k] - base[k] for k in K]))})
        per[e] = d
    models[e] = m
D['models'] = models


def paired(a, b):
    if a not in per or b not in per:
        return None
    v = np.array([per[a][k] - per[b][k] for k in K])
    return {'mean': float(v.mean()), 'win': int((v > 0).sum()), 'n': len(v)}


# 一次只差一件事的配對（51 位逐人相減）
D['paired'] = {
    'res': paired('mix_exp5', 'mix_exp2'),        # 只差積分解析度
    'version': paired('mix_exp4', 'mix_exp5'),    # 只差版本（平滑權重 2）
    'lam_disp': paired('mix_exp3', 'mix_exp4'),   # 位移場：平滑權重 2 -> 1
    'lam_vel_1': paired('mix_exp6', 'mix_exp5'),  # 速度場：2 -> 1
    'lam_vel_05': paired('mix_exp7', 'mix_exp6'),  # 速度場：1 -> 0.5
    'version_w1': paired('mix_exp3', 'mix_exp6'),  # 只差版本（平滑權重 1）
    'width_disp': paired('mix_wide', 'mix_exp3'),  # 位移場：加寬
    'width_vel': paired('mix_wide_vel', 'mix_exp6'),  # 速度場：加寬
    'version_wide': paired('mix_wide_vel', 'mix_wide'),  # 加寬後只差版本
}

# ── ① 擠爆的位置（三顆位移場；速度場版都不擠爆）───────────────────────────
FC = J('models', 'folding_check')
with open(os.path.join(FC, 'folding_by_subject.csv'), encoding='utf-8') as f:
    fsub = list(csv.DictReader(f))
atlas = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol']
aseg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg']
brain = ndimage.binary_fill_holes((atlas > 0.01) | (aseg > 0))     # 跟 check_folding.py 同一個定義
fold = {}
for e in ('mix_exp4', 'mix_exp3', 'mix_wide'):
    rr = [r for r in fsub if r['exp'] == e]
    z = np.load(os.path.join(FC, 'depth_%s.npz' % e))
    cnt = z['region_counts'].astype(float)
    share = {str(n): float(100 * c / cnt.sum()) for n, c in zip(z['region_names'], cnt)}
    heat = np.load(os.path.join(FC, 'heat_%s.npz' % e))['heat']
    fold[e] = {
        'n_subjects': len(rr),
        'points_med': float(np.median([int(r['n_folded']) for r in rr])),
        'clusters_med': float(np.median([int(r['n_clusters']) for r in rr])),
        'largest_med': float(np.median([int(r['largest_cluster']) for r in rr])),
        'largest_max': int(max(int(r['largest_cluster']) for r in rr)),
        'depth_med': float(np.median([float(r['median_depth_mm']) for r in rr])),
        'share': share,
        'ctx_wm': share['大腦皮質'] + share['大腦白質'],
        # 每人擠的地方一不一樣：腦內的點，有幾 % 至少 1／5／10 位在那裡擠爆過
        'any1': float(100 * (heat[brain] >= 1).mean()),
        'any5': float(100 * (heat[brain] >= 5).mean()),
        'any10': float(100 * (heat[brain] >= 10).mean()),
    }
D['folding'] = fold

# ── ③④ 殘留旁邊的結構 ──────────────────────────────────────────────────
SK = J('models', 'skullstrip_check')
with open(os.path.join(SK, 'label_dice_summary.csv'), encoding='utf-8') as f:
    t50 = {r['metric']: r for r in csv.DictReader(f)}
with open(os.path.join(SK, 'label_dice_pooled_summary.csv'), encoding='utf-8') as f:
    pool = {r['metric']: r for r in csv.DictReader(f)}
REG = {'top': 'top_vertex_mm', 'base': 'base_blob10', 'back': 'back_occ_mm'}
num = lambda r, k: float(r[k])
res = {}
for k, metric in REG.items():
    a, b = t50[metric], pool[metric]
    res[k] = {
        'labels': a['label_names'], 'near_shares': a['near_shares'],
        'labels_pooled': b['labels'], 'near_shares_pooled': b['near_shares'],
        'test_r': num(a, 'spearman_gain'), 'test_p': num(a, 'p_gain'),
        'pooled_n': int(b['pooled_n']), 'pooled_r': num(b, 'pooled_r'), 'pooled_p': num(b, 'pooled_p'),
        'cohort_r': {c: num(b, c + '_r') for c in ('test', 'val', 'MRS')},
        'cohort_p': {c: num(b, c + '_p') for c in ('test', 'val', 'MRS')},
        'dirty_n': int(b['dirty_n']), 'clean_n': int(b['clean_n']),
        'dirty_before': num(b, 'dirty_before'), 'dirty_after': num(b, 'dirty_after'),
        'clean_before': num(b, 'clean_before'), 'clean_after': num(b, 'clean_after'),
        'dirty_gain': num(b, 'dirty_gain'), 'clean_gain': num(b, 'clean_gain'), 'mw_p': num(b, 'mw_p'),
    }
D['residue'] = res

# 30 個結構全部平均 vs 只平均旁邊的結構（同樣 170 人、同樣批內百分位與批內置中）
from skullstrip_label_dice_pooled import load, COHORTS
rows = load()
ctx = lambda row: float(np.mean([float(row['label_%d' % l]) for l in (3, 42)]))
for r in rows:
    r['x'] = float(r['res']['top_vertex_mm'])
    r['g_all'] = float(r['after']['dice_mean']) - float(r['before']['dice_mean'])
    r['g_ctx'] = ctx(r['after']) - ctx(r['before'])
for cn in [c[0] for c in COHORTS]:
    rr = [r for r in rows if r['cohort'] == cn]
    xs = np.array([r['x'] for r in rr])
    ma, mc = np.mean([r['g_all'] for r in rr]), np.mean([r['g_ctx'] for r in rr])
    for r in rr:
        r['pct'] = (np.sum(xs < r['x']) + 0.5 * (np.sum(xs == r['x']) - 1)) / (len(xs) - 1)
        r['gc_all'], r['gc_ctx'] = r['g_all'] - ma, r['g_ctx'] - mc
r_all, p_all = spearmanr([r['pct'] for r in rows], [r['gc_all'] for r in rows])
r_ctx, p_ctx = spearmanr([r['pct'] for r in rows], [r['gc_ctx'] for r in rows])
D['dilution'] = {'n': len(rows), 'r_all': float(r_all), 'p_all': float(p_all),
                 'r_ctx': float(r_ctx), 'p_ctx': float(p_ctx)}
# 順帶：mix_exp3 對第五包 MRS（從沒看過的研究）
mrs = [r for r in rows if r['cohort'] == 'MRS']
D['mrs'] = {'n': len(mrs), 'after': float(np.mean([float(r['after']['dice_mean']) for r in mrs])),
            'before': float(np.mean([float(r['before']['dice_mean']) for r in mrs]))}

# 頭頂那層是不是 FreeSurfer 把腦畫太小（check_top_residue.py）
with open(os.path.join(SK, 'top_residue_split.csv'), encoding='utf-8') as f:
    ts = list(csv.DictReader(f))
hv = [r for r in ts if float(r['total_mm_pct']) >= 0.75]
cl = [r for r in ts if float(r['total_mm_pct']) <= 0.25]
chk = {'n': len(ts), 'dirty_n': len(hv), 'clean_n': len(cl)}
for k in ('top_ctx', 'ctx_thick', 'cont_int'):
    a = [float(r[k]) for r in hv if r[k] != 'nan']
    b = [float(r[k]) for r in cl if r[k] != 'nan']
    chk[k] = {'dirty': float(np.mean(a)), 'clean': float(np.mean(b)), 'p': float(mannwhitneyu(a, b).pvalue)}
chk['top_ctx_min'] = float(min(float(r['top_ctx']) for r in ts))
D['top_check'] = chk

# 後腦杓：test 那批的分布（skullstrip_all520.csv）
with open(os.path.join(SK, 'skullstrip_all520.csv'), encoding='utf-8') as f:
    back = [float(r['back_occ_mm']) for r in csv.DictReader(f) if r['split'] == 'test' and r['subject'] != 'A0131']
D['back'] = {'n': len(back), 'median': float(np.median(back)), 'min': float(min(back)), 'max': float(max(back))}

json.dump(D, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('ok ->', OUT)
for e, m in models.items():
    if m['status'] == 'done':
        print('  %-13s test %.4f（貢獻 +%.3f，擠爆 %.3f%%，epoch %s）' % (e, m['mean'], m['gain'], m['jneg'], m['epoch']))
    else:
        print('  %-13s 跑中' % e)
print('  殘留：頭頂 170 人 r=%+.2f；30 個結構一起平均 r=%+.2f' % (r_ctx, r_all))
for e, v in fold.items():
    print('  擠爆 %-9s 皮質＋白質 %.0f%%，離腦表面 %.1f mm，同一點 5 人以上 %.2f%%'
          % (e, v['ctx_wm'], v['depth_med'], v['any5']))
