# -*- coding: utf-8 -*-
"""10/14 meeting 簡報的數字，全部從原始 CSV 算 -> deck_data.json。投影片上不手打數字。

讀：
    models/<exp>/dice_<epoch>.csv                 test 結果（每個實驗資料夾只有一份 4 位數 epoch 的 CSV）
    models/mix_exp2/dice_baseline.csv             起點（只做線性對位）
    models/folding_check/                         ① 擠爆的位置（ASD/check_folding.py 產生）
    models/skullstrip_check/                      ③④ 殘留（skullstrip_label_dice*.py、check_top_residue.py 產生）
    log/mix_wide.txt、log/mix_wide_vel.txt        ⑤ 每步時間（沒加／有加顯存設定）

還沒跑完的實驗沒有 test CSV，標成 pending，簡報上顯示「跑中」（mix_exp6、7 10-04、mix_wide_vel 10-05 帶回來了）。
結果帶回來、放進 models/<exp>/ 之後重跑這支就會補上。
"""
import os
import sys
import csv
import glob
import json
import re
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
    # ⑥ 改架構（2026-10-06 加；都跟 mix_exp6 一樣是速度場、全尺寸、權重 1，只差架構）。還沒在 AI 上跑 → pending
    'mix_cascade': ('速度場', 1.0, '串兩顆', '全尺寸'),
    'mix_pyramid': ('速度場', 1.0, '由粗到細', '全尺寸'),
    'mix_cascade_pyramid': ('速度場', 1.0, '串兩顆＋由粗到細', '全尺寸'),
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
                  'points': float(np.mean([j[k] for k in K]) / 100 * 192 * 224 * 192),   # 每人平均幾個擠爆點
                  'points_max': float(max(j[k] for k in K) / 100 * 192 * 224 * 192),
                  'n_any': int(sum(j[k] > 0 for k in K)),
                  'gain': float(np.mean([d[k] - base[k] for k in K]))})
        per[e] = d
    models[e] = m
D['models'] = models


def paired(a, b):
    """51 位逐人相減。p 是 Wilcoxon signed-rank（2026-10-07 加：投影片改正式用語，「沒差」改寫成有無顯著差異）。"""
    if a not in per or b not in per:
        return None
    from scipy.stats import wilcoxon
    v = np.array([per[a][k] - per[b][k] for k in K])
    return {'mean': float(v.mean()), 'win': int((v > 0).sum()), 'n': len(v), 'p': float(wilcoxon(v).pvalue)}


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
    'version_wide': paired('mix_wide_vel', 'mix_wide'),  # 加寬後只差版本（速度場 - 位移場）
    'version_w1_vel': paired('mix_exp6', 'mix_exp3'),    # 預設寬度只差版本（速度場 - 位移場，跟上一行同方向）
}

# ── ② 平滑權重對各結構的影響（左右平均；30 個評估結構併成 17 種）─────────────────
PAIRS = {'大腦皮質': (3, 42), '大腦白質': (2, 41), '側腦室': (4, 43), '小腦白質': (7, 46), '小腦皮質': (8, 47),
         '視丘': (10, 49), '尾狀核': (11, 50), '殼核': (12, 51), '蒼白球': (13, 52), '海馬迴': (17, 53),
         '杏仁核': (18, 54), '腹側間腦': (28, 60), '脈絡叢': (31, 63), '第三腦室': (14,), '第四腦室': (15,),
         '腦幹': (16,), '腦脊髓液': (24,)}
struct = {}
for e in ('mix_exp5', 'mix_exp6', 'mix_exp7', 'mix_exp3', 'mix_wide', 'mix_wide_vel'):      # 加寬兩顆給 ⑤ 用（10-06）
    p = test_csv(e)
    if p is None:
        continue
    with open(p, encoding='utf-8') as f:
        rr = {r['file'][:-4]: r for r in csv.DictReader(f)}
    struct[e] = {n: float(np.mean([np.nanmean([float(rr[k]['label_%d' % l]) for k in K]) for l in ls]))
                 for n, ls in PAIRS.items()}
D['struct'] = struct

# ── ⑤ 加寬：越難對的人幫越多（2026-10-06 使用者要放進簡報；09-20 那份第 25 頁是位移場的版本）─────────
# 分組、相關都用「起點 Dice」（只做線性對位，兩顆模型都沒碰過）；用其中一顆模型自己的分數會有回歸平均的假象（CLAUDE.md 第 7 點）。
# 相關用 Pearson，跟 09-20 那張（2026-09-20_cross/gather.py 的 r_base）一樣
from scipy.stats import pearsonr
wd = {}
for key, a, b in (('vel', 'mix_wide_vel', 'mix_exp6'), ('disp', 'mix_wide', 'mix_exp3')):
    if a not in per or b not in per:
        continue
    bs = np.array([base[k] for k in K])
    g = np.array([per[a][k] - per[b][k] for k in K])
    o = np.argsort(bs)
    rho, pv = pearsonr(bs, g)
    wd[key] = {'subjects': K, 'base': bs.tolist(), 'gain': g.tolist(), 'r': float(rho), 'p': float(pv),
               'hard10': float(g[o[:10]].mean()), 'mid': float(g[o[10:-10]].mean()), 'easy10': float(g[o[-10:]].mean())}
D['wide_diff'] = wd

# ── ⑤ 訓練夠不夠久：驗證集第 100 輪之後的範圍、上下晃的大小、趨勢（每 100 輪變多少）──────────────
plateau = {}
for e in ('mix_wide_vel', 'mix_wide'):
    with open(J('models', e, 'dice_curve_val.csv'), encoding='utf-8') as f:
        cv = [(int(r['epoch']), float(r['dice_mean'])) for r in csv.DictReader(f)]
    eps = np.array([x for x, _ in cv if x >= 100])
    vs = np.array([y for x, y in cv if x >= 100])
    plateau[e] = {'lo': float(vs.min()), 'hi': float(vs.max()), 'sd': float(vs.std()), 'n': int(len(vs)),
                  'slope100': float(np.polyfit(eps, vs, 1)[0] * 100)}
D['plateau'] = plateau

# ── ⑤ 訓練 loss：最後一個 epoch 的影像項、平滑項 ─────────────────────────────────────────────
# 讀法同 ASD/plot_loss_curve.py（那支 import 時就解析命令列參數，所以不能直接 import，正規表示式照抄）
LOSS_LINE = re.compile(r'epoch:\s*(\d+)\s+step:\s*(\d+)/(\d+).*?loss:\s*(-?[\d.eE+-]+)\s+'
                       r'\((-?[\d.eE+-]+),\s*(-?[\d.eE+-]+)(?:,\s*(-?[\d.eE+-]+))?\)')


def last_epoch_loss(e):
    p = J('log', e + '.txt')
    if not os.path.exists(p):
        return None
    raw = open(p, 'rb').read()
    for enc in ('utf-16', 'utf-8', 'cp950'):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if '\ufffd' not in t and 'epoch' in t:
            break
    else:
        return None
    acc = {}
    for mm in LOSS_LINE.finditer(t):
        acc.setdefault(int(mm.group(1)), []).append((float(mm.group(5)), float(mm.group(6))))
    last = np.array(acc[max(acc)])
    return {'image': float(last[:, 0].mean()), 'smooth': float(last[:, 1].mean())}


D['loss_final'] = {e: last_epoch_loss(e) for e in ('mix_exp3', 'mix_wide', 'mix_exp6', 'mix_wide_vel')}

# ── ① 擠爆的位置（三顆位移場；速度場版都不擠爆）───────────────────────────
FC = J('models', 'folding_check')
with open(os.path.join(FC, 'folding_by_subject.csv'), encoding='utf-8') as f:
    fsub = list(csv.DictReader(f))
atlas = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol']
aseg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg']
brain = ndimage.binary_fill_holes((atlas > 0.01) | (aseg > 0))     # 跟 check_folding.py 同一個定義
fold = {}
for e in ('mix_exp4', 'mix_exp3', 'mix_wide', 'mix_exp6', 'mix_exp7', 'mix_wide_vel'):
    rr = [r for r in fsub if r['exp'] == e]
    if not rr or not os.path.exists(os.path.join(FC, 'heat_%s.npz' % e)):
        continue                                   # 速度場 exp6、7 10-04 補算，mix_wide_vel 10-05
    if sum(int(r['n_folded']) for r in rr) == 0:
        fold[e] = {'n_subjects': len(rr), 'points_mean': 0.0, 'points_max': 0, 'n_any': 0}
        continue
    z = np.load(os.path.join(FC, 'depth_%s.npz' % e))
    cnt = z['region_counts'].astype(float)
    share = {str(n): float(100 * c / cnt.sum()) for n, c in zip(z['region_names'], cnt)}
    heat = np.load(os.path.join(FC, 'heat_%s.npz' % e))['heat']
    fold[e] = {
        'n_subjects': len(rr),
        'points_med': float(np.median([int(r['n_folded']) for r in rr])),
        'points_mean': float(np.mean([int(r['n_folded']) for r in rr])),
        'points_max': int(max(int(r['n_folded']) for r in rr)),     # 擠爆點最多的那位有幾點
        'n_any': int(sum(int(r['n_folded']) > 0 for r in rr)),      # 有幾位至少有 1 個擠爆點
        'clusters_med': float(np.median([int(r['n_clusters']) for r in rr])),
        'largest_med': float(np.median([int(r['largest_cluster']) for r in rr])),
        'largest_max': int(max(int(r['largest_cluster']) for r in rr)),
        'depth_med': float(np.nanmedian([float(r['median_depth_mm']) for r in rr])),   # 沒有擠爆點的人是 nan
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
# 2026-10-05 使用者要標 Dice 數值：殘留最多／最少 1/4（批內百分位，同 skullstrip_label_dice_pooled.py）的起點 → 配準後。
# 只寫配準後會誤導：30 個結構一起平均時，殘留多的人起點就比較高，配準後也比較高
quart = {}
for nm, sel in (('dirty', [r for r in rows if r['pct'] >= 0.75]), ('clean', [r for r in rows if r['pct'] <= 0.25])):
    quart[nm] = {'n': len(sel),
                 'all_b': float(np.mean([float(r['before']['dice_mean']) for r in sel])),
                 'all_a': float(np.mean([float(r['after']['dice_mean']) for r in sel])),
                 'ctx_b': float(np.mean([ctx(r['before']) for r in sel])),
                 'ctx_a': float(np.mean([ctx(r['after']) for r in sel]))}
D['dilution']['quart'] = quart

# ── 2026-10-05：③④ 改用原始數值，170 人直接合在一起（使用者問「最少 → 最多」有沒有單位）──────────────────
# 橫軸：頭頂、後腦杓是殘留厚度（mm），顱底是最大一坨殘留的體積（mm³）；縱軸：只算殘留旁邊結構的 Dice 進步（不扣平均）。
# 三批的殘留分布差不多，跟批內百分位版（上面的 residue、dilution）比：頭頂 r −0.401 vs −0.395、顱底 −0.092 vs −0.087、
# 後腦杓 +0.039 vs +0.025。分組也直接用 mm 切（170 人的 1/4、3/4 分位數），老師比較好想像
from skullstrip_label_dice import NAME as LNAME
LID = {v: k for k, v in LNAME.items()}
# FreeSurferColorLUT 的正式名稱：第 14 頁表格寫「參考了哪些 FreeSurfer 結構」（2026-10-06 使用者要的）
FSNAME = {2: 'Left-Cerebral-White-Matter', 3: 'Left-Cerebral-Cortex', 4: 'Left-Lateral-Ventricle',
          7: 'Left-Cerebellum-White-Matter', 8: 'Left-Cerebellum-Cortex', 10: 'Left-Thalamus', 11: 'Left-Caudate',
          12: 'Left-Putamen', 13: 'Left-Pallidum', 14: '3rd-Ventricle', 15: '4th-Ventricle', 16: 'Brain-Stem',
          17: 'Left-Hippocampus', 18: 'Left-Amygdala', 24: 'CSF', 28: 'Left-VentralDC', 31: 'Left-choroid-plexus',
          41: 'Right-Cerebral-White-Matter', 42: 'Right-Cerebral-Cortex', 43: 'Right-Lateral-Ventricle',
          46: 'Right-Cerebellum-White-Matter', 47: 'Right-Cerebellum-Cortex', 49: 'Right-Thalamus', 50: 'Right-Caudate',
          51: 'Right-Putamen', 52: 'Right-Pallidum', 53: 'Right-Hippocampus', 54: 'Right-Amygdala',
          60: 'Right-VentralDC', 63: 'Right-choroid-plexus', 85: 'Optic-Chiasm'}
EVAL30 = {l for ls in PAIRS.values() for l in ls}                          # 30 個評估結構（labels.npz）


def fs_merged(ls):
    """[3, 42] -> 'Cerebral-Cortex'（左右合併時拿掉 Left-／Right-）；單一標籤照 LUT 原名"""
    side, _, rest = FSNAME[ls[0]].partition('-')
    return rest if len(ls) == 2 and side in ('Left', 'Right') else FSNAME[ls[0]]


# 2026-10-07 使用者：結構名稱要附英文（FreeSurferColorLUT 名稱）→ 第 9、24 頁的圖、第 11 頁的表
D['struct_en'] = {n: fs_merged(ls) for n, ls in PAIRS.items()}


def fs_pairs(labs):
    """[3, 42, 16, 47, 8] -> [['大腦皮質', 'Left/Right-Cerebral-Cortex（3、42）'], ['腦幹', 'Brain-Stem（16）'],
    ['小腦皮質', 'Left/Right-Cerebellum-Cortex（8、47）']]：左右合併，一個結構一列（第 14 頁表格一列一行）"""
    groups = {}
    for l in labs:
        side, _, rest = FSNAME[l].partition('-')
        groups.setdefault(rest if side in ('Left', 'Right') else FSNAME[l], []).append(l)
    return [[LNAME[ls[0]].lstrip('左右'),
             '%s（%s）' % ('Left/Right-' + key if len(ls) == 2 else FSNAME[ls[0]], '、'.join(str(l) for l in sorted(ls)))]
            for key, ls in groups.items()]


def near_groups(shares, included):
    """第 11 頁表格（2026-10-07 改成一個結構一列、附 FreeSurfer 名稱）：
    '左大腦皮質 52.0%、右大腦皮質 47.9%' -> [{zh, fs, labels, shares, included, note}]，左右合併、照原本順序"""
    inc = {LID[n] for n in included.split('、')}
    out, idx = [], {}
    for item in shares.split('、'):
        name, pct = item.rsplit(' ', 1)
        l = LID[name]
        side, _, rest = FSNAME[l].partition('-')
        key = rest if side in ('Left', 'Right') else FSNAME[l]
        if key not in idx:
            idx[key] = len(out)
            out.append({'zh': LNAME[l].lstrip('左右'), 'items': []})
        out[idx[key]]['items'].append((l, float(pct.rstrip('%'))))
    for g in out:
        g['items'].sort()
        ls = [l for l, _ in g['items']]
        g['labels'] = ls
        g['shares'] = [s for _, s in g['items']]
        g['fs'] = ('Left/Right-' + fs_merged(ls) if len(ls) == 2 else FSNAME[ls[0]]) + '（%s）' % '、'.join(map(str, ls))
        g['included'] = all(l in inc for l in ls)
        g['note'] = ('' if g['included'] else
                     '非 30 個評估結構' if not any(l in EVAL30 for l in ls) else '占比 < 5%')
        del g['items']
    return out


D['residue_near'] = {k: near_groups(res[k]['near_shares_pooled'], res[k]['labels_pooled']) for k in REG}

mm = {}
for k, metric in REG.items():
    labs = [LID[n] for n in pool[metric]['labels'].split('、')]      # 殘留旁邊的結構（同 skullstrip_label_dice_pooled.py 挑的）
    av = lambda row: float(np.nanmean([float(row['label_%d' % l]) for l in labs]))
    xs = np.array([float(r['res'][metric]) for r in rows])
    b = np.array([av(r['before']) for r in rows])
    a = np.array([av(r['after']) for r in rows])
    lo, hi = np.percentile(xs, [25, 75])
    dd, cc = xs >= hi, xs <= lo
    rho, pv = spearmanr(xs, a - b)
    ent = {'labs': labs, 'fs_pairs': fs_pairs(labs), 'n': len(rows), 'r': float(rho), 'p': float(pv), 'lo': float(lo), 'hi': float(hi),
           'dirty_n': int(dd.sum()), 'clean_n': int(cc.sum()),
           'dirty_b': float(b[dd].mean()), 'dirty_a': float(a[dd].mean()),
           'clean_b': float(b[cc].mean()), 'clean_a': float(a[cc].mean()),
           'mw_p': float(mannwhitneyu((a - b)[dd], (a - b)[cc]).pvalue)}
    if k == 'top':                                   # 第 12 頁：30 個結構一起平均當對照
        ab = np.array([float(r['before']['dice_mean']) for r in rows])
        aa = np.array([float(r['after']['dice_mean']) for r in rows])
        r2, p2 = spearmanr(xs, aa - ab)
        ent.update({'all_r': float(r2), 'all_p': float(p2),
                    'all_dirty_b': float(ab[dd].mean()), 'all_dirty_a': float(aa[dd].mean()),
                    'all_clean_b': float(ab[cc].mean()), 'all_clean_a': float(aa[cc].mean())})
    mm[k] = ent
D['residue_mm'] = mm
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
# ⑤ 每步時間：mix_wide 沒加顯存設定、mix_wide_vel 有加（log 每步印的 time:）
def step_time(e):
    p = J('log', e + '.txt')
    if not os.path.exists(p):
        return None
    raw = open(p, 'rb').read()
    for enc in ('utf-16', 'utf-8', 'cp950'):       # log 混用兩種編碼（CLAUDE.md「log/ 實際內容」）
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if '\ufffd' not in t and 'epoch' in t:
            break
    else:
        return None
    x = [float(v) for v in re.findall(r'time: ([\d.]+) sec', t)]
    return {'sec': float(np.mean(x)), 'hours': float(np.sum(x) / 3600), 'steps': len(x)} if x else None


D['train_time'] = {e: step_time(e) for e in ('mix_wide', 'mix_wide_vel')}

# ── ⑥ 改架構（老師紅字之外；2026-10-06 使用者決定先做，CLAUDE.md 待辦 5、手冊 §25）──────────────────────
# 第 0 步：不訓練，現成模型連跑 2～3 次（ASD/test_multipass.py 的 CSV；mix_wide_vel 是筆電半精度跑的 _amp 版）
from scipy.stats import wilcoxon
MP = {'mix_exp6': 'multipass_0190.csv', 'mix_exp3': 'multipass_0240.csv', 'mix_wide_vel': 'multipass_0240_amp.csv'}
mp = {}
for e, fn in MP.items():
    p = J('models', e, fn)
    if not os.path.exists(p):
        continue
    with open(p, encoding='utf-8') as f:
        rr = list(csv.DictReader(f))
    by = {(r['file'][:-4], int(r['pass'])): r for r in rr}
    npass = max(int(r['pass']) for r in rr)
    Dp = np.array([[float(by[(k, i)]['dice_mean']) for k in K] for i in range(1, npass + 1)])
    Jp = np.array([[int(by[(k, i)]['jneg_n']) for k in K] for i in range(1, npass + 1)])
    Sp = np.array([[float(by[(k, i)]['step_mm']) for k in K] for i in range(1, npass + 1)])
    ent = {'passes': [{'mean': float(Dp[i].mean()), 'points': float(Jp[i].mean()), 'points_max': int(Jp[i].max()),
                       'n_any': int((Jp[i] > 0).sum()), 'step_mm': float(Sp[i].mean())} for i in range(npass)]}
    for i in range(1, npass):                         # 第 i+1 次 vs 第 1 次（逐人配對）
        g = Dp[i] - Dp[0]
        ent['gain%d' % (i + 1)] = {'mean': float(g.mean()), 'win': int((g > 0).sum()), 'n': len(g),
                                   'p': float(wilcoxon(Dp[i], Dp[0]).pvalue)}
    if npass >= 3:
        g = Dp[2] - Dp[1]
        ent['p3_vs_p2'] = {'mean': float(g.mean()), 'win': int((g > 0).sum()), 'n': len(g)}
    # 越難對的人幫越多？用起點 Dice 分組（同 wide_diff）
    bs = np.array([base[k] for k in K])
    g = Dp[1] - Dp[0]
    o = np.argsort(bs)
    rho, pv = pearsonr(bs, g)
    ent['diff'] = {'r': float(rho), 'p': float(pv), 'hard10': float(g[o[:10]].mean()),
                   'mid': float(g[o[10:-10]].mean()), 'easy10': float(g[o[-10:]].mean())}
    # 每個結構（17 種，左右平均）：第 2 次 − 第 1 次
    ent['struct'] = {n: float(np.mean([np.nanmean([float(by[(k, 2)]['label_%d' % l]) - float(by[(k, 1)]['label_%d' % l])
                                                   for k in K]) for l in ls])) for n, ls in PAIRS.items()}
    mp[e] = ent
D['multipass'] = mp
# mix_exp6 連跑兩次 vs mix_wide_vel（正式的 test CSV）
if 'mix_exp6' in mp and 'mix_wide_vel' in per:
    with open(J('models', 'mix_exp6', MP['mix_exp6']), encoding='utf-8') as f:
        two = {r['file'][:-4]: float(r['dice_mean']) for r in csv.DictReader(f) if r['pass'] == '2'}
    a_ = np.array([two[k] for k in K])
    b_ = np.array([per['mix_wide_vel'][k] for k in K])
    D['multipass_vs_wide'] = {'mean': float((a_ - b_).mean()), 'win': int((a_ > b_).sum()), 'n': len(K),
                              'p': float(wilcoxon(a_, b_).pvalue), 'two': float(a_.mean()), 'wide': float(b_.mean())}

# 第 1、2 步與疊在一起（程式寫好、驗證通過，等 AI 跑）。參數直接從網路數（跟影像大小無關，用小影像蓋）；
# 每步倍數、顯存、AI 時數是筆電實測＋外插（手冊 §25.3、§25.4），不是 CSV —— 寫死在這裡
os.environ.setdefault('VXM_BACKEND', 'pytorch')
os.environ.setdefault('NEURITE_BACKEND', 'pytorch')
from arch import VxmCascade, VxmPyramid                                   # noqa: E402
import voxelmorph as vxm                                                    # noqa: E402
_s = (32, 32, 32)
npar = lambda net: int(sum(p.numel() for p in net.parameters()))
D['arch'] = [
    # 實驗, 改什麼, 參數, 每步倍數, 顯存 GB（實際）, AI 上幾小時
    {'exp': 'mix_exp6', 'what': '原本的 VoxelMorph', 'params': npar(vxm.networks.VxmDense(_s, int_downsize=1)),
     'tmul': 1.00, 'mem': 8.8, 'hours': None},
    {'exp': 'mix_cascade', 'what': '串兩顆', 'params': npar(VxmCascade(_s, n_cascades=2, int_downsize=1)),
     'tmul': 1.89, 'mem': 16.9, 'hours': 25},
    {'exp': 'mix_pyramid', 'what': '由粗到細', 'params': npar(VxmPyramid(_s)), 'tmul': 1.49, 'mem': 9.7, 'hours': 19},
    {'exp': 'mix_cascade_pyramid', 'what': '兩個疊在一起',
     'params': npar(VxmCascade(_s, n_cascades=2, int_downsize=1, stage='pyramid')), 'tmul': 2.86, 'mem': 19.0, 'hours': 37},
]


def no_nan(o):
    """JSON 不認得 NaN（build.js 的 JSON.parse 會直接報錯），一律換成 null。"""
    if isinstance(o, dict):
        return {k: no_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [no_nan(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


json.dump(no_nan(D), open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('ok ->', OUT)
for e, m in models.items():
    if m['status'] == 'done':
        print('  %-13s test %.4f（貢獻 +%.3f，擠爆 %.3f%%，epoch %s）' % (e, m['mean'], m['gain'], m['jneg'], m['epoch']))
    else:
        print('  %-13s 跑中' % e)
for e, t in D['train_time'].items():
    if t:
        print('  %-13s 每步 %.2f 秒、共 %.1f 小時（%d 步）' % (e, t['sec'], t['hours'], t['steps']))
print('  殘留：頭頂 170 人 r=%+.2f；30 個結構一起平均 r=%+.2f' % (r_ctx, r_all))
for e, v in fold.items():
    if 'ctx_wm' not in v:
        print('  擠爆 %-9s 沒有任何擠爆點' % e)
        continue
    print('  擠爆 %-9s 每人平均 %.1f 點，皮質＋白質 %.0f%%，同一點 5 人以上 %.2f%%'
          % (e, v['points_mean'], v['ctx_wm'], v['any5']))
