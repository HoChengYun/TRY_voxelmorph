# -*- coding: utf-8 -*-
"""只用「殘留旁邊的結構」算 Dice。老師 2026-09-30 紅字：「Dice 只算沒切乾淨附近的區域就好」。

老師的意思（使用者 09-30 轉述）：FreeSurfer 分了很多種結構，以前 Dice 是 30 種全部平均；
看殘留的時候不要全部平均，只挑殘留旁邊的那幾種結構，平均它們的 Dice。
（09-30 第一版 skullstrip_local_dice.py 是「切一塊空間、只算框內的體素」，那是另一件事，留著當補充。）

步驟
  1. 找殘留旁邊的結構：殘留每一點，找離它最近的已標記體素是哪個結構（distance transform 的最近點），
     用「該指標殘留最多的 10 位」統計；佔殘留點 >= 5%、而且在 30 個評估結構（labels.npz）裡的才算。
  2. 每位受試者：只平均這幾個結構的 Dice（每個結構照原本的方式算整個結構，數字直接讀 dice_*.csv）。
  3. 比「殘留最多 10 位」vs「最乾淨 10 位」的起點／配準後／模型貢獻，再看 50 位的相關。

輸出（models/skullstrip_check/）：label_dice_summary.csv、label_dice.png

用法：
    python ASD\\skullstrip_label_dice.py
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

CSV = os.path.join(ROOT, 'models', 'skullstrip_check', 'skullstrip_all520.csv')
AFTER = os.path.join(ROOT, 'models', 'mix_exp3', 'dice_0240.csv')
BEFORE = os.path.join(ROOT, 'models', 'mix_exp2', 'dice_baseline.csv')
TEST = os.path.join(ROOT, 'data', 'mixed_preprocessed_v2', 'test')
OUT = os.path.join(ROOT, 'models', 'skullstrip_check')
N, MIN_SHARE, EXCLUDE = 10, 0.05, {'A0131'}
EVAL = set(np.load(os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))['labels'].astype(int).tolist())
NAME = {2: '左大腦白質', 3: '左大腦皮質', 4: '左側腦室', 7: '左小腦白質', 8: '左小腦皮質', 10: '左視丘',
        11: '左尾狀核', 12: '左殼核', 13: '左蒼白球', 14: '第三腦室', 15: '第四腦室', 16: '腦幹',
        17: '左海馬迴', 18: '左杏仁核', 24: '腦脊髓液', 28: '左腹側間腦', 31: '左脈絡叢', 41: '右大腦白質',
        42: '右大腦皮質', 43: '右側腦室', 46: '右小腦白質', 47: '右小腦皮質', 49: '右視丘', 50: '右尾狀核',
        51: '右殼核', 52: '右蒼白球', 53: '右海馬迴', 54: '右杏仁核', 60: '右腹側間腦', 63: '右脈絡叢', 85: '視交叉'}
REGIONS = [('top', '頭頂', 'top_vertex_mm', lambda v, s: measure_top(v, s)[1]),
           ('base', '顱底', 'base_blob10', lambda v, s: measure_base(v, s)[1]),
           ('back', '後腦杓', 'back_occ_mm', lambda v, s: measure_back(v, s)[1])]


def read(p):
    with open(p, encoding='utf-8') as f:
        return {r['file'][:-4]: r for r in csv.DictReader(f)}


def near_labels(subjects, fn):
    """殘留每一點最近的結構，在這群人裡各佔多少（每人先算比例再平均）。"""
    tot = {}
    for s in subjects:
        d = np.load(os.path.join(TEST, s + '.npz'))
        vol, seg = d['vol'], d['seg'].astype(int)
        res = fn(vol, seg)
        if not res.any():
            continue
        _, ind = ndi.distance_transform_edt(seg == 0, return_indices=True)
        lab, cnt = np.unique(seg[tuple(ind[:, res])], return_counts=True)
        for l, c in zip(lab, cnt / cnt.sum()):
            tot[int(l)] = tot.get(int(l), 0.0) + c / len(subjects)
    return dict(sorted(tot.items(), key=lambda x: -x[1]))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    RUST, TEAL, PAPER = '#A34F1B', '#0E7C7B', '#FAFAF8'

    with open(CSV, encoding='utf-8') as f:
        meta = {r['subject']: r for r in csv.DictReader(f) if r['split'] == 'test' and r['subject'] not in EXCLUDE}
    A, B = read(AFTER), read(BEFORE)
    keep = sorted(s for s in meta if s in A and s in B)
    avg = lambda row, labs: float(np.nanmean([float(row['label_%d' % l]) for l in labs]))

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), facecolor=PAPER)
    for ax, (k, name, metric, fn) in zip(axes, REGIONS):
        order = sorted(keep, key=lambda s: -float(meta[s][metric]))
        dirty, clean = order[:N], order[-N:]
        share = near_labels(dirty, fn)
        labs = [l for l, v in share.items() if v >= MIN_SHARE and l in EVAL]
        x = np.array([float(meta[s][metric]) for s in keep])
        bef = np.array([avg(B[s], labs) for s in keep])
        aft = np.array([avg(A[s], labs) for s in keep])
        r_g, p_g = spearmanr(x, aft - bef)
        r_a, p_a = spearmanr(x, aft)
        idx = {s: i for i, s in enumerate(keep)}
        g = lambda ss, arr: float(np.mean([arr[idx[s]] for s in ss]))
        res = {'region': name, 'metric': metric, 'labels': ' '.join(str(l) for l in labs),
               'label_names': '、'.join(NAME.get(l, str(l)) for l in labs),
               'near_shares': '、'.join('%s %.1f%%' % (NAME.get(l, str(l)), 100 * v) for l, v in share.items() if v >= 0.005),
               'dirty_before': g(dirty, bef), 'dirty_after': g(dirty, aft), 'clean_before': g(clean, bef),
               'clean_after': g(clean, aft), 'spearman_gain': r_g, 'p_gain': p_g, 'spearman_after': r_a, 'p_after': p_a,
               'mw_p_gain': float(mannwhitneyu([(aft - bef)[idx[s]] for s in dirty],
                                               [(aft - bef)[idx[s]] for s in clean]).pvalue)}
        res['dirty_gain'] = res['dirty_after'] - res['dirty_before']
        res['clean_gain'] = res['clean_after'] - res['clean_before']
        rows.append(res)
        print('%-4s 旁邊的結構：%s' % (name, res['near_shares']))
        print('     只平均 %s：髒 %.3f→%.3f（%+.3f）  乾淨 %.3f→%.3f（%+.3f）  貢獻 r=%+.2f p=%.3f  配準後 r=%+.2f p=%.3f  組間 p=%.3f'
              % (res['label_names'], res['dirty_before'], res['dirty_after'], res['dirty_gain'], res['clean_before'],
                 res['clean_after'], res['clean_gain'], r_g, p_g, r_a, p_a, res['mw_p_gain']))

        xs = np.arange(3)
        vd = [res['dirty_before'], res['dirty_after'], res['dirty_gain']]
        vc = [res['clean_before'], res['clean_after'], res['clean_gain']]
        ax.bar(xs - 0.2, vd, width=0.38, color=RUST, label='殘留最多 %d 位' % N)
        ax.bar(xs + 0.2, vc, width=0.38, color=TEAL, label='最乾淨 %d 位' % N)
        for xx, v in zip(xs - 0.2, vd):
            ax.text(xx, v + 0.006, '%.3f' % v, ha='center', fontsize=11, fontweight='bold', color=RUST)
        for xx, v in zip(xs + 0.2, vc):
            ax.text(xx, v + 0.006, '%.3f' % v, ha='center', fontsize=11, fontweight='bold', color=TEAL)
        ax.set_xticks(xs)
        ax.set_xticklabels(['起點（只做線性對位）', '配準後', '模型貢獻'], fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_title('%s殘留旁邊的結構' % name, fontsize=13.5, fontweight='bold')
        ax.text(0.5, 0.97, '只平均：' + res['label_names'], transform=ax.transAxes, ha='center', va='top',
                fontsize=10.5, color='#5F6A6B', wrap=True)
        ptxt = 'p < 0.001' if p_g < 0.001 else 'p = %.3f' % p_g
        ax.text(0.5, -0.16, '殘留 vs 模型貢獻（50 位）：r = %+.2f，%s' % (r_g, ptxt),
                transform=ax.transAxes, ha='center', fontsize=11, color='#141A1D')
        ax.grid(axis='y', alpha=.3)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=11, frameon=False, loc='center right')
    axes[0].set_ylabel('Dice（mix_exp3，只平均殘留旁邊的結構）', fontsize=11)
    fig.suptitle('只用殘留旁邊的結構算 Dice：殘留最多 vs 最乾淨（test，排除 A0131）', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(OUT, 'label_dice.png'), dpi=115, facecolor=PAPER)
    plt.close(fig)
    p = os.path.join(OUT, 'label_dice_summary.csv')
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('->', p)
    print('->', os.path.join(OUT, 'label_dice.png'))


if __name__ == '__main__':
    main()
