# -*- coding: utf-8 -*-
"""殘留旁邊的結構 Dice，人數加大版：test 50 + val 50 + MRS 70（2026-09-30）。

使用者覺得只有 test 50 位、每組 10 位太少，借 val 和第五包 MRS 來算（三批都沒進過訓練）。
做法跟 skullstrip_label_dice.py 一樣（只平均殘留旁邊的結構），差別：
  - 三批分開算、也合起來算。三批來自不同研究，起點和殘留多寡本來就可能不一樣，
    合起來時殘留多寡用「在自己那批裡的百分位」、模型貢獻減掉自己那批的平均，避免把批次差異當成殘留的影響
  - 分組改成「每一批各取殘留最多／最少的 1/4」
  - 旁邊的結構：三批各自殘留最多的 10 位一起統計
排除 A0131（起點離群，同 §21）、FSS_A001（跟 val 的 A001 是同一顆腦，只算一次）。
模型都是 mix_exp3 第 240 輪。

輸出（models/skullstrip_check/）：label_dice_pooled_summary.csv、label_dice_pooled.png

用法：python ASD\\skullstrip_label_dice_pooled.py
"""
import os
import sys
import csv
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from check_skullstrip import measure_top, measure_base, measure_back
from skullstrip_label_dice import NAME, EVAL, MIN_SHARE

J = lambda *p: os.path.join(ROOT, *p)
SK = J('models', 'skullstrip_check')
COHORTS = [  # (名字, 殘留 CSV, 用 split 篩, 配準前, 配準後, 影像資料夾)
    ('test', J('models', 'skullstrip_check', 'skullstrip_all520.csv'), 'test',
     J('models', 'mix_exp2', 'dice_baseline.csv'), J('models', 'mix_exp3', 'dice_0240.csv'),
     J('data', 'mixed_preprocessed_v2', 'test')),
    ('val', J('models', 'skullstrip_check', 'skullstrip_all520.csv'), 'val',
     J('models', 'mix_exp2', 'dice_baseline_val.csv'), os.path.join(SK, 'cohorts', 'dice_mix_exp3_0240_val.csv'),
     J('data', 'mixed_preprocessed_v2', 'val')),
    ('MRS', os.path.join(SK, 'skullstrip_MRS.csv'), None,
     os.path.join(SK, 'cohorts', 'dice_baseline_MRS.csv'), os.path.join(SK, 'cohorts', 'dice_mix_exp3_0240_MRS.csv'),
     J('data', 'MRS_preprocessed_v1', 'train')),
]
EXCLUDE = {'A0131', 'FSS_A001'}
N_NEAR = 10            # 每批取殘留最多幾位來找「旁邊的結構」
REGIONS = [('top', '頭頂', 'top_vertex_mm', lambda v, s: measure_top(v, s)[1]),
           ('base', '顱底', 'base_blob10', lambda v, s: measure_base(v, s)[1]),
           ('back', '後腦杓', 'back_occ_mm', lambda v, s: measure_back(v, s)[1])]


def read(p, key='file'):
    with open(p, encoding='utf-8') as f:
        return {r[key].replace('.npz', ''): r for r in csv.DictReader(f)}


def load():
    rows = []
    for name, res_csv, split, before, after, npz in COHORTS:
        R = read(res_csv, 'subject')
        B, A = read(before), read(after)
        for s, r in R.items():
            if (split and r['split'] != split) or s in EXCLUDE or s not in A or s not in B:
                continue
            rows.append({'cohort': name, 'subject': s, 'npz': os.path.join(npz, s + '.npz'),
                         'res': r, 'before': B[s], 'after': A[s]})
    return rows


def near_labels(sel, fn):
    tot = {}
    for r in sel:
        d = np.load(r['npz'])
        vol, seg = d['vol'], d['seg'].astype(int)
        m = fn(vol, seg)
        if not m.any():
            continue
        _, ind = ndi.distance_transform_edt(seg == 0, return_indices=True)
        lab, cnt = np.unique(seg[tuple(ind[:, m])], return_counts=True)
        for l, c in zip(lab, cnt / cnt.sum()):
            tot[int(l)] = tot.get(int(l), 0.0) + c / len(sel)
    return dict(sorted(tot.items(), key=lambda x: -x[1]))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    RUST, TEAL, PAPER, MUTED = '#A34F1B', '#0E7C7B', '#FAFAF8', '#5F6A6B'
    CCOL = {'test': '#0E7C7B', 'val': '#3A9E9C', 'MRS': '#A34F1B'}

    rows = load()
    names = [c[0] for c in COHORTS]
    print('人數：', {c: sum(r['cohort'] == c for r in rows) for c in names})
    out, avg = [], lambda row, labs: float(np.nanmean([float(row['label_%d' % l]) for l in labs]))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor=PAPER)
    for c_, (k, name, metric, fn) in enumerate(REGIONS):
        # 1. 旁邊的結構：每批殘留最多的 N_NEAR 位一起統計
        sel = []
        for cn in names:
            rr = sorted([r for r in rows if r['cohort'] == cn], key=lambda r: -float(r['res'][metric]))
            sel += rr[:N_NEAR]
        share = near_labels(sel, fn)
        labs = [l for l, v in share.items() if v >= MIN_SHARE and l in EVAL]
        # 2. 每人：只平均這些結構
        for r in rows:
            r['b'], r['a'] = avg(r['before'], labs), avg(r['after'], labs)
            r['g'] = r['a'] - r['b']
            r['x'] = float(r['res'][metric])
        # 3. 批內百分位、批內置中
        for cn in names:
            rr = [r for r in rows if r['cohort'] == cn]
            xs = np.array([r['x'] for r in rr])
            gm = np.mean([r['g'] for r in rr])
            for r in rr:
                r['pct'] = (np.sum(xs < r['x']) + 0.5 * (np.sum(xs == r['x']) - 1)) / (len(xs) - 1)
                r['gc'] = r['g'] - gm
        res = {'region': name, 'metric': metric, 'labels': '、'.join(NAME.get(l, str(l)) for l in labs),
               'near_shares': '、'.join('%s %.1f%%' % (NAME.get(l, str(l)), 100 * v) for l, v in share.items() if v >= 0.005)}
        for cn in names:
            rr = [r for r in rows if r['cohort'] == cn]
            rho, p = spearmanr([r['x'] for r in rr], [r['g'] for r in rr])
            res.update({cn + '_n': len(rr), cn + '_r': rho, cn + '_p': p})
        rho, p = spearmanr([r['pct'] for r in rows], [r['gc'] for r in rows])
        dirty = [r for r in rows if r['pct'] >= 0.75]
        clean = [r for r in rows if r['pct'] <= 0.25]
        res.update({'pooled_n': len(rows), 'pooled_r': rho, 'pooled_p': p,
                    'dirty_n': len(dirty), 'clean_n': len(clean),
                    'dirty_before': np.mean([r['b'] for r in dirty]), 'dirty_after': np.mean([r['a'] for r in dirty]),
                    'clean_before': np.mean([r['b'] for r in clean]), 'clean_after': np.mean([r['a'] for r in clean]),
                    'dirty_gain_c': np.mean([r['gc'] for r in dirty]), 'clean_gain_c': np.mean([r['gc'] for r in clean]),
                    'mw_p': mannwhitneyu([r['gc'] for r in dirty], [r['gc'] for r in clean]).pvalue})
        res['dirty_gain'] = res['dirty_after'] - res['dirty_before']
        res['clean_gain'] = res['clean_after'] - res['clean_before']
        out.append(res)
        print('%-4s 旁邊的結構：%s' % (name, res['near_shares']))
        print('     只平均 %s' % res['labels'])
        print('     各批 r：' + '  '.join('%s %+.2f（p=%.3f，n=%d）' % (cn, res[cn + '_r'], res[cn + '_p'], res[cn + '_n']) for cn in names))
        print('     合起來 r=%+.2f p=%.4f（n=%d）｜殘留最多 1/4（%d 人）貢獻 %+.3f vs 最少 1/4（%d 人）%+.3f，組間 p=%.4f'
              % (rho, p, len(rows), len(dirty), res['dirty_gain'], len(clean), res['clean_gain'], res['mw_p']))

        # 上排：殘留最多 1/4 vs 最少 1/4
        ax = axes[0][c_]
        xs = np.arange(3)
        vd = [res['dirty_before'], res['dirty_after'], res['dirty_gain']]
        vc = [res['clean_before'], res['clean_after'], res['clean_gain']]
        ax.bar(xs - 0.2, vd, width=0.38, color=RUST, label='殘留最多 1/4（%d 人）' % len(dirty))
        ax.bar(xs + 0.2, vc, width=0.38, color=TEAL, label='殘留最少 1/4（%d 人）' % len(clean))
        for xx, v in zip(xs - 0.2, vd):
            ax.text(xx, v + 0.006, '%.3f' % v, ha='center', fontsize=11, fontweight='bold', color=RUST)
        for xx, v in zip(xs + 0.2, vc):
            ax.text(xx, v + 0.006, '%.3f' % v, ha='center', fontsize=11, fontweight='bold', color=TEAL)
        ax.set_xticks(xs)
        ax.set_xticklabels(['起點（只做線性對位）', '配準後', '模型貢獻'], fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_title('%s殘留旁邊的結構' % name, fontsize=13.5, fontweight='bold')
        ax.text(0.5, 0.97, '只平均：' + res['labels'], transform=ax.transAxes, ha='center', va='top',
                fontsize=10.5, color=MUTED)
        ax.grid(axis='y', alpha=.3)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if c_ == 0:
            ax.legend(fontsize=10.5, frameon=False, loc='center right')
            ax.set_ylabel('Dice（mix_exp3）', fontsize=11)

        # 下排：每個人一個點
        ax = axes[1][c_]
        # 圖上不分來源（使用者 09-30：不用標是 test / val / MRS），所有人同一種顏色
        ax.scatter([r['pct'] for r in rows], [r['gc'] for r in rows], s=22, alpha=0.75, color='#3A9E9C')
        xx = np.array([r['pct'] for r in rows])
        yy = np.array([r['gc'] for r in rows])
        b1, b0 = np.polyfit(xx, yy, 1)
        ax.plot([0, 1], [b0, b0 + b1], color='#141A1D', lw=1.8)
        ax.axhline(0, color=MUTED, lw=0.8, ls=':')
        ptxt = 'p < 0.001' if p < 0.001 else 'p = %.3f' % p
        ax.set_title('合起來：r = %+.2f，%s（%d 人）' % (rho, ptxt, len(rows)), fontsize=12.5, fontweight='bold')
        ax.set_xlabel('殘留多寡（百分位，右邊 = 殘留多）', fontsize=10.5)
        if c_ == 0:
            ax.set_ylabel('模型貢獻（跟平均比）', fontsize=10.5)
        ax.grid(alpha=.3)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle('只用殘留旁邊的結構算 Dice（%d 人）' % len(rows), fontsize=14.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(SK, 'label_dice_pooled.png'), dpi=110, facecolor=PAPER)
    plt.close(fig)
    p = os.path.join(SK, 'label_dice_pooled_summary.csv')
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print('->', p)
    print('->', os.path.join(SK, 'label_dice_pooled.png'))


if __name__ == '__main__':
    main()
