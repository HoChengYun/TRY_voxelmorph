# -*- coding: utf-8 -*-
"""10/28 簡報：⑥ 架構修改之結果（2 × 2 完成、延長訓練）、⑧ 與傳統方法（ANTs SyN）之比較、附錄（影像邊界之 folding）。
2026-10-21 使用者：「開做」（AI 帶回 mix_cascade_pyramid、mix_pyramid_500；ANTs SyN 51 位跑完）。

數字 → deck_data_arch.json（build.js 讀；投影片上的數字一律從這裡來，不手打）
圖   → models\\deck_charts\\1028_*.png
  1028_arch_res.png     五個模型 × 五項指標（Dice、未參與訓練之結構、HD95、SDlogJ、腦內 folding）
  1028_arch_2x2.png     每位受試者之 Dice（mix_exp6、2× width、cascade、coarse-to-fine、兩者合併；10-22 起不畫 ΔDice）
  1028_train_curves.png 訓練過程：四個模型各一欄，上 訓練損失之相似度項、下 val Dice（10-22）
  1028_p500.png         coarse-to-fine 延長至 500 epochs：val Dice、訓練之相似度項
  1028_ants_paper.png   VoxelMorph 論文 Table I 與本研究：affine、ANTs SyN、VoxelMorph
  1028_bgfold.png       附錄：影像邊界 folding 之成因（一維示意）與只改推論之 padding 後之結果
ANTs 與 VoxelMorph 並排之受試者圖另見 make_ants_vis.py（要等 models\\ants_syn\\vis 跑完）。

資料來源
  test Dice           models\\<實驗>\\dice_<epoch>.csv（AI 上 test_dice.py 的正式結果）；affine = models\\mix_exp2\\dice_baseline.csv
  HD95、SDlogJ、folding  models\\<實驗>\\surface_<epoch>.csv（筆電 test_dice.py --surface；腦內 folding = jneg_fg_pct × atlas 非背景 voxel 數）
  未參與訓練之結構、NCC  models\\<實驗>\\semisup_check_<epoch>.csv（ASD/check_semisup.py）
  ANTs SyN            models\\ants_syn\\dice_syn_test.csv（ASD/ants_syn_baseline.py，論文 §V-A-3 之參數）
  影像邊界 folding     models\\folding_check\\bg\\bg_border_test.csv、bg_stats.csv、bg_levels.csv（ASD/check_bg_folding.py，5 位）
  val 曲線、訓練 loss  models\\mix_pyramid(_500)\\dice_curve_val.csv、log\\mix_pyramid(_500).txt
  TransMorph-diff（10-28）models\\tm_diff_mixv2\\：dice_0075.csv、dice_curve_val.csv 從 TransMorph 專案 runs\\tm_diff_mixv2\\ 複製；
                      surface_0075.csv、semisup_check_0075.csv 是筆電上 eval_mixv2.py --surface --heldout 算的（訓練 log 沒帶回）
  TransMorph（10-23）  models\\tm_mixv2\\：dice_0205.csv、dice_curve_val.csv、train_log.txt 是 AI 帶回來的
                      （TransMorph 專案 runs\\tm_mixv2\\，複製過來）；surface_0205.csv、semisup_check_0205.csv 是筆電上
                      TransMorph 專案的 mixv2\\eval_mixv2.py --surface --heldout 算的（算法同 test_dice.py、check_semisup.py）
"""
import os
import re
import csv
import json

import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
J = lambda *p: os.path.join(ROOT, *p)
OUT = J('models', 'deck_charts')
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#D9D9D2'
TEAL, TEAL_L, BLUE, AMBER, RED, GRAY, PURPLE = '#0E7C7B', '#5BB8B6', '#2F7FD0', '#D08C2B', '#B03A2E', '#9A9A93', '#7A4FA0'
SLATE = '#4B5D73'                    # TransMorph
SLATE_L = '#8A9BB0'                  # TransMorph-diff（10-28）
PLUM, PLUM_L = '#8E5B8A', '#B98AB4'    # 半監督 γ = 0.5、5（10-28）
W = 12.13
FG_VOX = 1867705                     # atlas 非背景 voxel 數（jneg_fg_pct 之分母）
ALL_VOX = 192 * 224 * 192

# 論文 Table I（Balakrishnan et al., IEEE TMI 2019）：Dice、folding voxel 數、CPU／GPU 秒數
TAB1 = {'affine': 0.584, 'ants': 0.749, 'vxm': 0.753, 'ants_fold': 9662, 'vxm_fold': 19077, 'ants_sec': 9059, 'vxm_gpu_sec': 0.45}

# 三個架構修改模型讀 deck_models.json（XC／XP／XM；換模型只改那個檔，見 換模型.md）；
# LEGACY（10-06 版三顆＋mix_pyramid_500）也照讀：延長訓練、腦外 folding 兩頁（build.js 已關）講的是它們
from deck_models import ARCH, XC, XP, XM, LEGACY
MODELS = [('mix_exp6', '0190'), ('mix_wide_vel', '0240')] + list(ARCH.values()) + [('mix_exp8', '0250'), ('mix_exp9', '0245'),
                                                                       ('mix_exp3', '0240'), ('tm_mixv2', '0205'),
                                                                       ('tm_diff_mixv2', '0075')]
MODELS += [m for m in LEGACY if m[0] not in dict(MODELS)]
EP = dict(MODELS)


def rd(p):
    with open(p, encoding='utf-8') as fh:
        return {r['file']: r for r in csv.DictReader(fh)}


base = rd(J('models', 'mix_exp2', 'dice_baseline.csv'))
FILES = sorted(base)
col = lambda tab, k, scale=1.0: np.array([float(tab[f][k]) * scale for f in FILES])

V = {'affine': {'dice': col(base, 'dice_mean'), 'held': col(rd(J('models', 'semisup_check_affine.csv')), 'heldout_mean'),
                'ncc': col(rd(J('models', 'semisup_check_affine.csv')), 'ncc'),
                'hd95': col(rd(J('models', 'mix_exp2', 'surface_baseline.csv')), 'hd95_mean')}}
for e, ep in MODELS:
    d, s = rd(J('models', e, 'dice_%s.csv' % ep)), rd(J('models', e, 'surface_%s.csv' % ep))
    v = {'dice': col(d, 'dice_mean'), 'hd95': col(s, 'hd95_mean'), 'sdlogj': col(s, 'sdlogj'),
         'fold': col(s, 'jneg_fg_pct', FG_VOX / 100), 'fold_all': col(s, 'jneg_pct', ALL_VOX / 100)}
    p = J('models', e, 'semisup_check_%s.csv' % ep)
    if os.path.exists(p):
        q = rd(p)
        v['held'], v['ncc'] = col(q, 'heldout_mean'), col(q, 'ncc')
    V[e] = v
syn = rd(J('models', 'ants_syn', 'dice_syn_test.csv'))
assert set(syn) == set(FILES), 'ANTs SyN 的受試者跟 test 對不上'
V['ants'] = {'dice': col(syn, 'dice_mean'), 'fold': col(syn, 'jneg_fg_pct', FG_VOX / 100), 'sec': col(syn, 'sec')}


def params(e):
    import torch
    ck = torch.load(J('models', e, '%s.pt' % EP[e]), map_location='cpu', weights_only=False)
    return int(sum(t.numel() for k, t in ck['model_state'].items() if not k.endswith('.grid')))


def paired(a, b, k='dice', lower_better=False):
    x, y = V[a][k], V[b][k]
    d = x - y
    win = int((d < 0).sum() if lower_better else (d > 0).sum())
    return {'mean': float(d.mean()), 'win': win, 'n': len(d), 'p': float(wilcoxon(x, y).pvalue)}


OUT_D = {'n': len(FILES), 'paper': TAB1}
OUT_D['m'] = {e: {k: float(v.mean()) for k, v in V[e].items()} for e in V}
OUT_D['params'] = {e: params(e) for e in ['mix_exp6', 'mix_wide_vel', XC, XP, XM, 'mix_exp3']}
# TransMorph 的 .pt 格式不同（TransMorph 專案的 common.save_model），參數量讀訓練 log 第二行「共 N 個參數」
OUT_D['params']['tm_mixv2'] = int(re.search(r'共 (\d+) 個參數', open(J('models', 'tm_mixv2', 'train_log.txt'), encoding='utf-8').read()).group(1))
# TransMorph-diff（10-28）：訓練 log 沒帶回來，參數量直接數 .pt（TransMorph 專案 runs\tm_diff_mixv2\0075.pt；不含 relative_position_index 這種 buffer）
def _tm_params(p):
    import torch
    sd = torch.load(p, map_location='cpu', weights_only=False)['state_dict']
    return int(sum(t.numel() for k, t in sd.items() if 'relative_position_index' not in k))


OUT_D['params']['tm_diff_mixv2'] = _tm_params(os.path.join('D:' + os.sep, 'MyHome', 'MRI', 'Transmorph_NewArchitecture', 'runs',
                                                           'tm_diff_mixv2', '0075.pt'))
P = {}
for e in ['mix_wide_vel', XC, XP, XM]:
    P[e + '_vs_exp6'] = {k: paired(e, 'mix_exp6', k, k in ('hd95', 'sdlogj')) for k in ('dice', 'held', 'ncc', 'hd95')}
P['cp_vs_cascade'] = paired(XM, XC)
P['cp_vs_pyramid'] = paired(XM, XP)
P['pyramid_vs_cascade'] = paired(XP, XC)
P['p500_vs_pyramid'] = paired('mix_pyramid_500', 'mix_pyramid')        # 以下三行：延長訓練那頁（舊模型，見 LEGACY）
P['cp_vs_p500'] = paired('mix_cascade_pyramid', 'mix_pyramid_500')
P['cp_vs_p500_held'] = paired('mix_cascade_pyramid', 'mix_pyramid_500', 'held')
for e in dict.fromkeys(['mix_exp3', 'mix_exp6', XC, XP, 'mix_pyramid_500', XM, 'mix_cascade_pyramid', 'tm_mixv2', 'tm_diff_mixv2']):
    P[e + '_vs_ants'] = paired(e, 'ants')
for e in ['mix_exp3', 'mix_exp6', 'mix_wide_vel', XC, XP, XM]:   # TransMorph（10-23）
    P['tm_vs_' + e] = paired('tm_mixv2', e)
    if e != 'mix_exp3':                                                    # mix_exp3 沒有 semisup_check
        P['tm_vs_' + e + '_m'] = {k: paired('tm_mixv2', e, k, k == 'hd95') for k in ('held', 'ncc', 'hd95')}
# TransMorph-diff（10-28）
for e in ['affine', 'tm_mixv2', 'mix_exp6', 'mix_exp3']:
    P['tmd_vs_' + e] = paired('tm_diff_mixv2', e)
# 照原作版 vs 舊版（10-28；舊版三顆在 LEGACY，換模型後 XC／XP／XM 是照原作版）
OLD = {'cascade': 'mix_cascade', 'c2f': 'mix_pyramid', 'merged': 'mix_cascade_pyramid'}
if XM != OLD['merged']:
    P['new_vs_old'] = {r: paired(ARCH[r][0], OLD[r]) for r in OLD}
    P['new_vs_old_m'] = {r: {k: paired(ARCH[r][0], OLD[r], k, k == 'hd95') for k in ('held', 'hd95')} for r in OLD}
    OUT_D['old'] = OLD
OUT_D['p'] = P
g1, g2, g12 = (P[e + '_vs_exp6']['dice']['mean'] for e in (XC, XP, XM))
OUT_D['additive'] = {'cascade': g1, 'pyramid': g2, 'both': g12, 'ratio': g12 / (g1 + g2)}
order = np.argsort(V['affine']['dice'])
OUT_D['hard'] = {e: {'hard10': float((V[e]['dice'] - V['mix_exp6']['dice'])[order[:10]].mean()),
                     'easy10': float((V[e]['dice'] - V['mix_exp6']['dice'])[order[-10:]].mean()),
                     'hard10_dice': float(V[e]['dice'][order[:10]].mean()),       # Dice 本身（10-22 起投影片只寫 Dice）
                     'easy10_dice': float(V[e]['dice'][order[-10:]].mean())}
                 for e in ['mix_exp6', XC, XP, XM]}

# ── 延長訓練：val 曲線（0～500）、訓練之相似度項 ─────────────────────────────
def val_curve(p):
    with open(p, encoding='utf-8') as fh:
        return {int(r['epoch']): float(r['dice_mean']) for r in csv.DictReader(fh)}


VC = val_curve(J('models', 'mix_pyramid', 'dice_curve_val.csv'))
VC.update({k: v for k, v in val_curve(J('models', 'mix_pyramid_500', 'dice_curve_val.csv')).items() if k > 250})
VCP = val_curve(J('models', 'mix_cascade_pyramid', 'dice_curve_val.csv'))
_vt = val_curve(J('models', 'tm_diff_mixv2', 'dice_curve_val.csv'))           # TransMorph-diff 那頁（10-28）：val 選定之 epoch
OUT_D['train_tmd_best'] = max(_vt, key=_vt.get)
win = lambda C, a, b: float(np.mean([v for k, v in C.items() if a <= k <= b]))
OUT_D['p500'] = {'win': {'%d-%d' % (a, b): win(VC, a, b) for a, b in [(150, 200), (200, 250), (255, 300), (300, 400), (400, 500)]},
                 'best250': max(((k, v) for k, v in VC.items() if k <= 250), key=lambda t: t[1]),
                 'best500': max(VC.items(), key=lambda t: t[1]),
                 'cp_win': {'%d-%d' % (a, b): win(VCP, a, b) for a, b in [(150, 200), (200, 250)]}}

# 10-28：半監督（train_semisup.py）的括號裡多一項標籤項 →（影像項, 平滑項, 標籤項），第三項可有可無
PAT = re.compile(r'epoch: (\d+)\s+step: \d+/\d+\s+time: ([\d.]+) sec\s+loss: (-?[\d.]+)\s+\((-?[\d.]+), (-?[\d.]+)(?:, -?[\d.]+)?\)')


def loss_by_epoch(name):
    raw = open(J('models', 'tm_mixv2', 'train_log.txt') if name == 'tm_mixv2' else J('log', name + '.txt'), 'rb').read()
    t = None
    for enc in ('utf-8', 'utf-16', 'cp950'):
        try:
            c = raw.decode(enc)
        except Exception:
            continue
        if '�' not in c and 'epoch' in c:
            t = c
            break
    E = {}
    for m in PAT.finditer(t):
        E.setdefault(int(m.group(1)), []).append((float(m.group(2)), float(m.group(4))))
    return {e: (float(np.mean([x[1] for x in v])), float(np.sum([x[0] for x in v]))) for e, v in E.items()}


LP = loss_by_epoch('mix_pyramid')
LP.update(loss_by_epoch('mix_pyramid_500'))
LCP = loss_by_epoch('mix_cascade_pyramid')
OUT_D['p500']['img_loss'] = {'250': LP[250][0], '500': LP[500][0]}
OUT_D['hours'] = {'mix_pyramid_500': sum(LP[e][1] for e in LP if e > 250) / 3600,
                  'mix_cascade_pyramid': sum(v[1] for v in LCP.values()) / 3600}

# ── ANTs SyN ───────────────────────────────────────────────────────────
OUT_D['ants'] = {'sec_mean': float(V['ants']['sec'].mean()), 'sec_min': float(V['ants']['sec'].min()),
                 'sec_max': float(V['ants']['sec'].max()), 'threads': 20}

# ── 附錄：影像邊界之 folding ────────────────────────────────────────────
BT = list(csv.DictReader(open(J('models', 'folding_check', 'bg', 'bg_border_test.csv'), encoding='utf-8')))
BS = list(csv.DictReader(open(J('models', 'folding_check', 'bg', 'bg_stats.csv'), encoding='utf-8')))
bg = {}
for e in ['mix_pyramid', 'mix_pyramid_500', 'mix_cascade_pyramid']:
    b = [r for r in BT if r['model'] == e]
    s = [r for r in BS if r['model'] == e]
    near = sum(int(r['bg_border_0_0']) + int(r['bg_border_1_1']) + int(r['bg_border_2_4']) for r in s)
    bg[e] = {'n': len(b), 'bg_zeros': float(np.mean([float(r['bg_zeros']) for r in b])),
             'bg_border': float(np.mean([float(r['bg_border']) for r in b])),
             'fg_zeros': float(np.mean([float(r['fg_zeros']) for r in b])),
             'fg_border': float(np.mean([float(r['fg_border']) for r in b])),
             'dice_maxdiff': float(max(abs(float(r['dice_border']) - float(r['dice_zeros'])) for r in b)),
             'within4': near / max(1, sum(int(r['fold_bg']) for r in s)),
             'disp_border': float(np.mean([float(r['disp_border_mean']) for r in s])),
             'bg_test51': float((V[e]['fold_all'] - V[e]['fold']).mean())}
s6 = [r for r in BS if r['model'] == 'mix_exp6']
bg['mix_exp6'] = {'disp_border': float(np.mean([float(r['disp_border_mean']) for r in s6]))}
OUT_D['bgfold'] = bg

with open(os.path.join(HERE, 'deck_data_arch.json'), 'w', encoding='utf-8') as fh:
    json.dump(OUT_D, fh, ensure_ascii=False, indent=1)
print('->', os.path.join(HERE, 'deck_data_arch.json'))


# ══ 圖 ═══════════════════════════════════════════════════════════════
def save(fig, name, dpi=130):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def clean(ax, axis='x'):
    ax.grid(alpha=.3, axis=axis)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def dot_chart(name, rows, panels, h):
    """列 = 模型、欄 = 指標（同 2026-10-14_redpen/make_semisup.py 的 dot_chart）。"""
    fig, axes = plt.subplots(1, len(panels), figsize=(W, h), facecolor=PAPER, sharey=True)
    fig.subplots_adjust(left=0.17 if max(len(r[0]) for r in rows) <= 20 else 0.195,   # 10-28：mix_cascade_pyramid_rcn_dwp 名稱較長
                         right=0.985, top=1 - 0.62 / h, bottom=0.3 / h, wspace=0.14)
    yv = -np.arange(len(rows), dtype=float)
    for ax, (title, note, k, kind, xl, fmt) in zip(axes, panels):
        span = xl[1] - xl[0]
        for yy, (lab, cfg, c, val) in zip(yv, rows):
            v = val.get(k)
            if v is None:
                ax.text(xl[0] + 0.02 * span, yy, '—', va='center', fontsize=10, color=MUTED)
                continue
            if kind == 'dot':
                if v < xl[0] or v > xl[1]:      # 超出範圍（10-28，TransMorph-diff）：點畫在邊界、數字加箭頭
                    lo_side = v < xl[0]
                    ax.plot(xl[0] if lo_side else xl[1], yy, 'o', color=c, ms=8, clip_on=False)
                    ax.text(xl[0] + 0.045 * span if lo_side else xl[1] - 0.045 * span, yy, ('← ' + fmt(v)) if lo_side else (fmt(v) + ' →'),
                            va='center', ha='left' if lo_side else 'right', fontsize=10.5, color=INK)
                    continue
                ax.plot([xl[0], v], [yy, yy], color=c, lw=1, alpha=0.25)
                ax.plot(v, yy, 'o', color=c, ms=8)
            else:
                if v > xl[1]:                    # 超出範圍：長條畫到邊界、數字寫在條內（10-23，TransMorph 的 folding、SDlogJ）
                    ax.barh(yy, xl[1], height=0.6, color=c)
                    ax.text(xl[1] - 0.015 * span, yy, fmt(v) + ' →', va='center', ha='right', fontsize=10.5, color='white')
                    continue
                ax.barh(yy, v, height=0.6, color=c)
            inside = kind == 'bar' and v > 0.62 * span
            ax.text(v + (-0.015 if inside else (0.045 if kind == 'dot' else 0.02)) * span, yy, fmt(v), va='center',
                    ha='right' if inside else 'left', fontsize=10.5, color='white' if inside else INK)
        ax.set_xlim(*xl)
        ax.text(0, 1.0 + 0.36 / h, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=11.5, fontweight='bold', color=INK)
        if note:
            ax.text(0, 1.0 + 0.06 / h, note, transform=ax.transAxes, ha='left', va='bottom', fontsize=9.5, color=MUTED)
        ax.tick_params(axis='x', labelsize=9.5)
        ax.tick_params(axis='y', length=0)
        clean(ax)
    tr = axes[0].get_yaxis_transform()
    for yy, (lab, cfg, c, _) in zip(yv, rows):
        dy = 0.17 if len(rows) <= 7 else 0.21           # 10-28：9 列時兩行字會擠在一起，拉開一點
        axes[0].text(-0.04, yy + dy, lab, transform=tr, ha='right', va='center', fontsize=11, fontweight='bold', color=INK)
        axes[0].text(-0.04, yy - dy - 0.04, cfg, transform=tr, ha='right', va='center', fontsize=9.5, color=c)
    axes[0].set_yticks([])
    axes[0].set_ylim(yv[-1] - 0.75, yv[0] + 0.55)
    save(fig, name)


f2, f3 = (lambda v: '%.2f' % v), (lambda v: '%.3f' % v)
fN = lambda v: '0' if v == 0 else ('%.1f' % v if v < 10 else '{:,.0f}'.format(v))
M = OUT_D['m']

# ── 1. 六個模型 × 五項指標（10-23 加 TransMorph：放在 2× width 後面，同屬「加大網路」）──────────────────
ROWS5 = [('mix_exp6', 'VoxelMorph（SVF）', TEAL), ('mix_wide_vel', '2× width', TEAL_L),
         ('tm_mixv2', 'TransMorph（disp., 46.8 M）', SLATE), (XC, 'Cascade（2 stages）', BLUE),
         (XP, 'Coarse-to-fine', AMBER), (XM, 'Cascade + coarse-to-fine', RED)]
# 10-28 使用者：「很多一起比較的圖沒有補上 transmorph diff」→ 多一列；它的 Dice、HD95 遠在其他模型之外，範圍不跟著放大（點畫在邊界、加箭頭）
TMD_ROW = ('tm_diff_mixv2', 'TransMorph-diff（SVF, MSE）', SLATE_L)
# 10-28 使用者：「就拿論文裡 30 個結構就好」「把半監督的數字加進來看」→ 拿掉「未參與訓練之結構」那一格；最後加兩列半監督（標 *：標籤項即為評估之 30 結構）
SEMI_ROWS = [('mix_exp8', 'Semi-sup. γ = 0.5 *', PLUM), ('mix_exp9', 'Semi-sup. γ = 5 *', PLUM_L)]
NAME = {'tm_mixv2': 'TransMorph', 'tm_diff_mixv2': 'TransMorph-diff'}
rows = [(NAME.get(e, e), cfg, c, {k: M[e][k] for k in ('dice', 'hd95', 'sdlogj', 'fold')})
        for e, cfg, c in ROWS5[:3] + [TMD_ROW] + ROWS5[3:] + SEMI_ROWS]
OUTSIDE = ('TransMorph-diff', 'mix_exp8', 'mix_exp9')
# 範圍只用非監督、正常訓練的模型定：TransMorph-diff 與半監督的 Dice、HD95 遠在外面，TransMorph 系列的 SDlogJ、folding 大一兩個數量級 → 超出的畫在邊界加箭頭
SV = [r for r in rows if not r[0].startswith('TransMorph') and r[0] not in OUTSIDE]
RR = [r for r in rows if r[0] not in OUTSIDE]
lo = lambda k, pad: min(r[3][k] for r in RR) - pad
hi = lambda k, pad: max(r[3][k] for r in RR) + pad
dot_chart('1028_arch_res.png', rows, [
    ('(a) Dice ↑', '30 結構；affine %.3f' % M['affine']['dice'], 'dice', 'dot', (lo('dice', 0.004), hi('dice', 0.006)), f3),
    ('(b) HD95（mm）↓', '30 結構；affine %.2f' % M['affine']['hd95'], 'hd95', 'dot', (lo('hd95', 0.03), hi('hd95', 0.05)), f2),
    ('(c) SDlogJ ↓', 'log|J| 之標準差（腦內）', 'sdlogj', 'bar', (0, max(r[3]['sdlogj'] for r in SV) * 1.45), f2),
    ('(d) Folding voxels ↓', '腦內，每位平均', 'fold', 'bar', (0, max(r[3]['fold'] for r in SV) * 1.45), fN),
], 4.9)

# ── 2. 每位受試者之 Dice（2026-10-22 使用者：「簡報不用再放 Dice 變化了，希望用 Dice 來比較」）──
# 原本畫相對 mix_exp6 之 ΔDice；改成直接畫 Dice，同一位受試者以細線相連（看得出每一位都往上）
# 10-28：加 TransMorph-diff 與半監督（使用者：「p11 可以加上半監督」），一度各開一軸（縱軸另計）；
# 同日使用者：「右邊的不能整合在一塊嗎」→ 三組之範圍首尾相接（TransMorph-diff 0.64～0.74、其他 0.73～0.84、半監督 0.84～0.94），
# 併回同一個縱軸。細線只連原本六欄（2 × 2、2× width、TransMorph）與半監督兩欄；TransMorph-diff 單獨一欄；半監督加淡紫底色（*）
COLS = ROWS5 + [('tm_diff_mixv2', None, SLATE_L), ('mix_exp8', None, PLUM), ('mix_exp9', None, PLUM_L)]
NM = len(ROWS5)
XS = np.array(list(range(NM)) + [NM + 0.4, NM + 1.75, NM + 2.75])     # 組與組之間多空一點
LAB = {'mix_exp6': 'VoxelMorph\n（mix_exp6）', 'mix_wide_vel': '2× width', 'tm_mixv2': 'TransMorph\n（disp.）',
       XC: 'Cascade\n（2 stages）', XP: 'Coarse-\nto-fine', XM: 'Cascade\n+ coarse-to-fine', 'tm_diff_mixv2': 'TransMorph\n-diff',
       'mix_exp8': 'Semi-sup.\nγ = 0.5 *', 'mix_exp9': 'Semi-sup.\nγ = 5 *'}
fig, ax = plt.subplots(figsize=(7.9, 4.75), facecolor=PAPER)
fig.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.17)
rng = np.random.default_rng(0)
jit = rng.uniform(-0.12, 0.12, len(FILES))
Y = np.array([V[e]['dice'] for e, _, _ in COLS])                        # 欄數 × 51
for j in range(Y.shape[1]):
    ax.plot(XS[:NM] + jit[j], Y[:NM, j], color=GRAY, lw=0.6, alpha=0.35, zorder=1)
    ax.plot(XS[-2:] + jit[j], Y[-2:, j], color=GRAY, lw=0.6, alpha=0.35, zorder=1)
ax.axvspan(XS[-2] - 0.5, XS[-1] + 0.5, color=PLUM, alpha=0.07, lw=0, zorder=0)
for i, (e, _, c) in enumerate(COLS):
    ax.scatter(XS[i] + jit, Y[i], s=13, color=c, alpha=0.75, edgecolors='none', zorder=2)
    ax.plot([XS[i] - 0.3, XS[i] + 0.3], [Y[i].mean()] * 2, color=INK, lw=2, zorder=3)
for g in (list(range(NM)), [NM], [NM + 1, NM + 2]):                     # 平均標在該組最高點之上，同組對齊
    top = Y[g].max() + 0.005
    for i in g:
        ax.text(XS[i], top, '%.3f' % Y[i].mean(), ha='center', va='bottom', fontsize=10, fontweight='bold', color=INK)
IC = [e for e, _, _ in ROWS5].index(XM)
add = OUT_D['additive']
exp_add = V['mix_exp6']['dice'].mean() + add['cascade'] + add['pyramid']
ax.plot([IC - 0.32, IC + 0.32], [exp_add] * 2, color=MUTED, lw=1.4, ls='--', zorder=3)
ax.text(IC + 0.36, exp_add, '若完全相加\n%.3f' % exp_add, ha='left', va='center', fontsize=8.5, color=MUTED)
ax.set_xticks(XS); ax.set_xticklabels([LAB[e] for e, _, _ in COLS], fontsize=8.5)
ax.set_xlim(-0.55, XS[-1] + 0.55)
ax.set_ylim(Y.min() - 0.008, Y.max() + 0.03)
ax.set_ylabel('Dice（30 結構平均）', fontsize=10)
ax.set_title('每位受試者之 Dice（test %d 位；橫線＝平均，細線連接同一位）' % len(FILES), fontsize=10, loc='left')
clean(ax, 'y')
save(fig, '1028_arch_2x2.png')

# ── 2b. 訓練過程：每個模型一欄，上 訓練損失之相似度項、下 val Dice ─────────────────
# 2026-10-22 使用者：「最近幾次實驗 loss 和 train 的圖可以放簡報」「分開放，train 和 val 放一起」「這次 meeting 先放舊的」
# → val loss 還沒有（test_dice.py 10-22 才加 ncc_loss 欄），這次用現有的：訓練 log 之影像項（每 epoch 100 步平均）＋ dice_curve_val.csv
#   相似度項跨模型可以比（總 loss 不行：cascade 每一顆都罰平滑），所以上排共用縱軸
TRAIN4 = [('mix_exp6', 'VoxelMorph\n（mix_exp6）', TEAL), ('tm_mixv2', 'TransMorph\n（disp.，46.8 M）', SLATE),
          (XC, 'Cascade', BLUE), (XP, 'Coarse-to-fine', AMBER), (XM, 'Cascade +\ncoarse-to-fine', RED)]   # 10-23 加 TransMorph
# 10-28：多一欄 TransMorph-diff。訓練 log 沒帶回、相似度是 MSE（跟 NCC 不能比）→ 上排只寫說明；val Dice 約 0.70，下排另開縱軸
# 10-28 使用者：「p14 也可以補上半監督」「（縱軸另計）保留」→ 再加兩欄半監督：上排相似度項同一縱軸（同為 −NCC，可比）；
# 下排 val Dice 約 0.90～0.93，兩欄共用另一縱軸。另計縱軸都是同樣的刻度間距；它們前面留空白放刻度，免得跟左邊一欄疊在一起
SEMI2 = [('mix_exp8', 'Semi-sup. *\nγ = 0.5', PLUM), ('mix_exp9', 'Semi-sup. *\nγ = 5', PLUM_L)]
NT = len(TRAIN4)
WR = [1] * NT + [0.3, 1, 0.3, 1, 1]                                   # 0.3 = 放刻度的空白
COL = {e: j for j, (e, _, _) in enumerate(TRAIN4)}
COL.update({'tm_diff_mixv2': NT + 1, 'mix_exp8': NT + 3, 'mix_exp9': NT + 4})
fig = plt.figure(figsize=(W, 5.4), facecolor=PAPER)
gs = fig.add_gridspec(2, len(WR), left=0.07, right=0.995, top=0.85, bottom=0.1, hspace=0.12, wspace=0.08, width_ratios=WR)
TOP, BOT = {}, {}
for e, j in COL.items():                  # 上排共用 mix_exp6 的縱軸；下排三組：原本五欄、TransMorph-diff、半監督兩欄
    TOP[e] = fig.add_subplot(gs[0, j], sharex=TOP.get('mix_exp6'), sharey=None if e == 'tm_diff_mixv2' else TOP.get('mix_exp6'))
    by = BOT.get('mix_exp6') if j < NT else (BOT.get('mix_exp8') if e == 'mix_exp9' else None)
    BOT[e] = fig.add_subplot(gs[1, j], sharex=TOP['mix_exp6'], sharey=by)
    TOP[e].tick_params(labelbottom=False, labelleft=(e == 'mix_exp6'))
    BOT[e].tick_params(labelleft=(e in ('mix_exp6', 'tm_diff_mixv2', 'mix_exp8')))
OUT_D['train'] = {}
for e, nm, c in TRAIN4 + SEMI2:
    L = loss_by_epoch(e)
    Vv = val_curve(J('models', e, 'dice_curve_val.csv'))
    le, kv = sorted(L), sorted(Vv)
    top, bot = TOP[e], BOT[e]
    top.plot(le, [L[k][0] for k in le], color=c, lw=1.3)
    top.set_title(nm, fontsize=9.5, fontweight='bold', color=c)
    bot.plot(kv, [Vv[k] for k in kv], 'o-', color=c, ms=2.8, lw=1.1)
    be = int(EP[e])
    bot.plot(be, Vv[be], '*', ms=14, color=c, markeredgecolor=INK, zorder=5)
    bot.text(be + (8 if be >= 230 else 0), Vv[be] + 0.0025, 'epoch %d' % be, ha='right' if be >= 230 else 'center', fontsize=8.5, color=INK)   # 最右邊一欄靠右，免得被切掉
    w_ = lambda a, b: float(np.mean([v for k, v in Vv.items() if a <= k <= b]))
    OUT_D['train'][e] = {'img': {str(k): L[k][0] for k in (50, 150, 200, 250)},
                         'val_win': {'150-200': w_(150, 200), '200-250': w_(200, 250)}, 'best': [be, Vv[be]]}
    for ax in (top, bot):
        ax.tick_params(labelsize=9)
        clean(ax, 'both')
    bot.set_xlabel('epoch', fontsize=10)
TOP['mix_exp6'].set_ylim(-0.285, -0.15)
BOT['mix_exp6'].set_ylim(0.76, 0.825)
b0 = 0.825 - 0.76                                                     # 另計縱軸：跟其他欄同樣的刻度間距
sb = max(OUT_D['train'][e]['best'][1] for e, _, _ in SEMI2)
BOT['mix_exp8'].set_ylim(sb + 0.012 - b0, sb + 0.012)
BOT['mix_exp9'].text(0.97, 0.04, '縱軸另計', transform=BOT['mix_exp9'].transAxes, ha='right', fontsize=8.5, color=MUTED)
# TransMorph-diff 一欄
top, bot = TOP['tm_diff_mixv2'], BOT['tm_diff_mixv2']
Vt = val_curve(J('models', 'tm_diff_mixv2', 'dice_curve_val.csv'))
kt = sorted(Vt)
bot.plot(kt, [Vt[k] for k in kt], 'o-', color=SLATE_L, ms=2.8, lw=1.1)
bt = OUT_D['train_tmd_best']
bot.plot(bt, Vt[bt], '*', ms=14, color=SLATE_L, markeredgecolor=INK, zorder=5)
bot.text(bt, Vt[bt] + 0.0025, 'epoch %d' % bt, ha='center', fontsize=8.5, color=INK)
bot.set_ylim(Vt[bt] + 0.012 - b0, Vt[bt] + 0.012)
bot.set_xlabel('epoch', fontsize=10)
bot.tick_params(labelsize=9)
bot.text(0.97, 0.04, '縱軸另計', transform=bot.transAxes, ha='right', fontsize=8.5, color=MUTED)
top.set_title('TransMorph-diff\n（SVF，MSE）', fontsize=9.5, fontweight='bold', color=SLATE_L)
top.text(0.5, 0.5, '訓練 log 未帶回\n相似度為 MSE，\n不可與 NCC 比較', transform=top.transAxes, ha='center', va='center', fontsize=9, color=MUTED)
top.set_yticks([])
for ax in (top, bot):
    clean(ax, 'both')
wt = lambda a, b: float(np.mean([v for k, v in Vt.items() if a <= k <= b]))
OUT_D['train']['tm_diff_mixv2'] = {'val_win': {'150-200': wt(150, 200), '200-250': wt(200, 250)}, 'best': [bt, Vt[bt]]}
TOP['mix_exp6'].set_ylabel('訓練損失之相似度項\n（−NCC，訓練集；↓）', fontsize=10)
BOT['mix_exp6'].set_ylabel('val Dice（30 結構；↑）\n★ = 選定之 epoch', fontsize=10)
fig.suptitle('上：訓練集（每 epoch 100 步之平均）　下：validation 51 位（每 5 epochs；縱軸自 0.76 起，TransMorph-diff、半監督另計）',
             fontsize=10, color=MUTED, x=0.07, ha='left')
save(fig, '1028_train_curves.png')
with open(os.path.join(HERE, 'deck_data_arch.json'), 'w', encoding='utf-8') as fh:    # OUT_D['train'] 是畫圖時才算的，再寫一次
    json.dump(OUT_D, fh, ensure_ascii=False, indent=1)

# ── 3. 延長訓練：val Dice 與訓練之相似度項 ────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.9), facecolor=PAPER)
fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.15, wspace=0.2)
ks = sorted(k for k in VC if k > 0)
a1.plot([k for k in ks if k <= 250], [VC[k] for k in ks if k <= 250], 'o-', ms=3, lw=1.2, color=AMBER, label='mix_pyramid（1～250）')
a1.plot([k for k in ks if k >= 250], [VC[k] for k in ks if k >= 250], 'o-', ms=3, lw=1.2, color=RED,
        label='mix_pyramid_500（自 epoch 250 接續）')
for (a, b) in [(150, 200), (200, 250), (300, 400), (400, 500)]:
    w_ = win(VC, a, b)
    a1.plot([a, b], [w_, w_], color=INK, lw=2.2)
    a1.text(b, w_ + 0.0004, '%.4f' % w_, ha='right', va='bottom', fontsize=8.5, color=INK, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85), zorder=6)
for k, c in [(OUT_D['p500']['best250'][0], AMBER), (OUT_D['p500']['best500'][0], RED)]:
    a1.plot(k, VC[k], '*', ms=15, color=c, markeredgecolor=INK, zorder=5)
a1.axvline(250, color=MUTED, ls=':', lw=1)
a1.set_ylim(0.800, 0.8195)
a1.set_xlabel('epoch'); a1.set_ylabel('val Dice（30 結構）')
a1.set_title('(a) val Dice（每 5 epochs；粗線 = 區間平均；★ = 選定之 epoch）', fontsize=10.5, loc='left')
a1.legend(fontsize=8.5, loc='lower right')
clean(a1, 'both')
le = sorted(LP)
a2.plot(le, [LP[e][0] for e in le], color=AMBER, lw=1.2, label='mix_pyramid／mix_pyramid_500')
a2.plot(sorted(LCP), [LCP[e][0] for e in sorted(LCP)], color=PURPLE, lw=1.2, label='mix_cascade_pyramid（250 epochs）')
a2.axvline(250, color=MUTED, ls=':', lw=1)
a2.set_ylim(-0.275, -0.235)
a2.set_xlabel('epoch'); a2.set_ylabel('相似度項（−NCC，訓練集）')
a2.set_title('(b) 訓練損失之相似度項（每 epoch 平均；越低越好）', fontsize=10.5, loc='left')
a2.legend(fontsize=8.5, loc='upper right')
clean(a2, 'both')
save(fig, '1028_p500.png')

# ── 4. 論文 Table I 與本研究 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor=PAPER)
fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.2)
groups = [('VoxelMorph 論文（Table I）\n8 個資料集，test 250 位', [('Affine', TAB1['affine'], GRAY), ('ANTs SyN', TAB1['ants'], PURPLE),
                                                         ('VoxelMorph', TAB1['vxm'], TEAL)]),
          ('本研究（mixed_v2）\ntest %d 位' % len(FILES), [('Affine', M['affine']['dice'], GRAY), ('ANTs SyN', M['ants']['dice'], PURPLE),
                                                     ('VoxelMorph', M['mix_exp3']['dice'], TEAL),
                                                     ('TransMorph', M['tm_mixv2']['dice'], SLATE),      # 10-23
                                                     ('TransMorph-diff', M['tm_diff_mixv2']['dice'], SLATE_L),   # 10-28
                                                     ('Cascade +\ncoarse-to-fine', M[XM]['dice'], RED)])]
x0, ticks, tl = 0, [], []
fig.subplots_adjust(bottom=0.3)
for gname, bars in groups:
    for j, (lab, v, c) in enumerate(bars):
        ax.bar(x0 + j, v, width=0.78, color=c)
        ax.text(x0 + j, v + 0.005, '%.3f' % v, ha='center', va='bottom', fontsize=9.5, color=INK)
        # 10-23 加 TransMorph 後一組 5 條，標籤改兩行，免得互相重疊
        ticks.append(x0 + j); tl.append({'ANTs SyN': 'ANTs\nSyN', 'VoxelMorph': 'Voxel\nMorph', 'TransMorph': 'Trans\nMorph',
                                         'TransMorph-diff': 'Trans\nMorph\n-diff',
                                         'Cascade +\ncoarse-to-fine': 'Cascade\n+ C2F'}.get(lab, lab))
    ax.text(x0 + (len(bars) - 1) / 2, -0.2, gname, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9.5,
            color=INK, fontweight='bold')
    x0 += len(bars) + 0.8
ax.set_ylim(0.5, 0.86)
ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=8.5)
ax.set_ylabel('Dice')
ax.set_title('Affine、ANTs SyN、VoxelMorph（displacement field，λ = 1）之 Dice', fontsize=10.5, loc='left')
clean(ax, 'y')
save(fig, '1028_ants_paper.png')

# ── 5. 附錄：影像邊界之 folding ─────────────────────────────────────────
N, Uv, uv = 192, 2.1, 0.6
x = np.arange(N - 7, N)


def sample(U, p, mode):
    q = np.clip(p, 0, N - 1) if mode == 'border' else p
    i0 = np.floor(q).astype(int)
    w = q - i0
    get = lambda i: np.where((i >= 0) & (i <= N - 1), U[np.clip(i, 0, N - 1)], 0.0)
    return (1 - w) * get(i0) + w * get(i0 + 1)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.6), facecolor=PAPER, gridspec_kw={'width_ratios': [1, 1.15]})
fig.subplots_adjust(left=0.06, right=0.985, top=0.85, bottom=0.17, wspace=0.28)
for mode, c, lab in [('zeros', RED, "padding_mode = 'zeros'（訓練與評估之設定）"), ('border', BLUE, "padding_mode = 'border'")]:
    U = uv + sample(np.full(N, Uv), x + uv, mode)
    phi = x + U
    a1.plot(x, phi, 'o-', color=c, label=lab, ms=5)
    Jd = np.gradient(phi)
    for xi, pi, ji in zip(x, phi, Jd):
        if ji <= 0:
            a1.annotate(r'$\partial\phi/\partial x = %.2f$' % ji, (xi, pi), textcoords='offset points', xytext=(-12, -20), ha='right',
                        fontsize=9.5, color=c)
a1.axvline(N - 1, color=MUTED, ls=':', lw=1)
a1.text(N - 1 - 0.1, a1.get_ylim()[0] + 0.3, '影像邊界', ha='right', fontsize=9, color=MUTED)
a1.set_xlabel('x（voxel）'); a1.set_ylabel(r'$\phi(x) = x + U(x)$')
a1.set_title(r'(a) 一維示意：$U(x) = u(x) + U_{\uparrow}(x + u(x))$，$U_{\uparrow} = %.1f$、$u = %.1f$' % (Uv, uv),
             fontsize=10.5, loc='left')
a1.legend(fontsize=8.5, loc='upper left')
clean(a1, 'both')
NM = {'mix_pyramid': 'Coarse-to-fine', 'mix_pyramid_500': 'Coarse-to-fine\n（500 epochs）', 'mix_cascade_pyramid': 'Cascade +\ncoarse-to-fine'}
xs = np.arange(3)
z = [bg[e]['bg_zeros'] for e in NM]
b = [bg[e]['bg_border'] for e in NM]
a2.bar(xs - 0.2, z, 0.38, color=RED, label="'zeros'")
a2.bar(xs + 0.2, b, 0.38, color=BLUE, label="'border'（僅改推論）")
for i, (zz, bb) in enumerate(zip(z, b)):
    a2.text(i - 0.2, zz, '{:,.0f}'.format(zz), ha='center', va='bottom', fontsize=9.5)
    a2.text(i + 0.2, bb, '{:,.0f}'.format(bb), ha='center', va='bottom', fontsize=9.5)
a2.set_xticks(xs); a2.set_xticklabels(list(NM.values()), fontsize=9.5)
a2.set_ylabel('腦外 folding voxels（每位平均）')
a2.set_title('(b) 推論時改變 padding 之結果（%d 位；Dice 差異 ≤ %.5f）'
             % (bg['mix_cascade_pyramid']['n'], max(bg[e]['dice_maxdiff'] for e in NM)), fontsize=10.5, loc='left')
a2.legend(fontsize=8.5)
clean(a2, 'y')
save(fig, '1028_bgfold.png')
