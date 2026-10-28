# -*- coding: utf-8 -*-
"""去顱骨殘留改用「體積」量（老師 2026-10-13 紅字，10/14 簡報第 12、18 頁）。

老師：「下次可以換成算 freesurfer seg 以外的都當成是顱骨」「去看 freesurfer seg 外面剩多少殘留體積」
      「可以用體積來估計看看」「為何只有顱底是體積另外兩個是厚度」。
使用者 10-13：把 FreeSurfer 的 seg 想成一顆球，沿著球的表面往外找，球外面看到的殘留都算沒去乾淨的體積。

做法（每位受試者；affine 後的 atlas 空間，1 voxel = 1 mm³）
  1. 球 = seg > 0（FreeSurfer aseg 有標的地方）做半徑 CLOSE_R mm 的 closing 再補洞：把腦溝、左右半腦中間的縫補起來。
     10-13 實測：不補的話，腦溝裡、正中縫裡的灰色組織（皮質的部分容積、大腦鐮）都會被算成殘留，一位就有 248 cm³
  2. 殘留 = 球以外、影像不是 0 的 voxel（老師 10-13：「球表面外面就當殘留體積，不用門檻」）。
     FreeSurfer 去顱骨後腦外本來是 0，留下來的（腦周圍的腦脊髓液、腦膜、骨髓…）全部算；每位約 30～50 萬 mm³。
     另記一欄 res_bright＝球外亮度 >= 皮質（3、42）中位數一半的部分（以前厚度法的門檻，腦脊髓液比它暗），只拿來對照。
     亮度 > 0 與 > 0.001 只差約 0.6%（affine 內插在邊緣產生的極小值），不影響
  3. 殘留每一點找最近的球表面點（distance transform 的最近點），依那個表面點在腦的哪一側分區：
     （表面點 − 全腦中心）÷ 腦在該軸的半長，絕對值最大的那一軸決定區域：上＝顱頂、下＝顱底、後＝枕部、前＝額部、左右＝兩側
  4. 主要結果是整顆腦的總殘留體積（使用者 10-13：「老師想要一起看，先不要分區」）；分區的欄位留著備用。
     每區的殘留體積（mm³）；另記離球表面的距離分層，以及球裡面、seg 外面的亮組織（腦溝、正中縫，不算殘留，只記下來）

陣列方向（RAS）：第 0 軸越大越右、第 1 軸越大越前、第 2 軸越大越上（10-13 用小腦、左右皮質的位置確認）。

輸出（models/skullstrip_check/）
  residue_volume_r<半徑>.csv          每人一列：各區體積、距離分層、球內 seg 外的亮組織、門檻
  residue_volume_r<半徑>_near.json    每人每區：殘留最近的結構各幾個 voxel（找「殘留旁邊的結構」用）
  residue_volume_r<半徑>_summary.csv  --analyze 的結果
  residue_volume_<受試者>_r<半徑>.png  --show 的示意圖
  主要結果用半徑 5 mm；3、8 mm 是敏感度檢查（10-13：sub-0043 總量 135／121／106 cm³，8 mm 開始把大腦、小腦之間的縫也包進球）

用法
  python ASD\\skullstrip_volume.py --scan --analyze --r 5      # 170 位約 10 分鐘
  python ASD\\skullstrip_volume.py --show sub-0043

tigerbx 版（2026-10-22，老師 10/14 紅字 p12「換乾淨的 tbet（ex tigerbx）看 Dice 差多少」；使用者：「看 tigerbx 那一套跑出來的效果」）：
  --data tigerbx：影像、標籤換成 data\\tigerbx_preprocessed_v2（tigerbx 的 tbet＋aseg43），球用 tigerbx 的標籤補；
  Dice 用 tigerbx 標籤（atlas 用 atlas_mni152_09c_v3_seg_tigerbx.npz），affine 起點與配準後都在 skullstrip_check\\cohorts\\ 用現在的資料重算
  （dice_tiger_baseline_<test|val>.csv、dice_tiger_exp3_0210_<test|val>.csv）。只有 test、val（MRS 沒有 tigerbx），同樣排除 A0131、FSS_A001。
  輸出檔名多 _tigerbx：residue_volume_tigerbx_r5.csv、_near.json、_summary.csv。
  python ASD\\skullstrip_volume.py --data tigerbx --scan --analyze --r 5
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from skullstrip_label_dice import NAME, EVAL, MIN_SHARE
from skullstrip_label_dice_pooled import COHORTS, EXCLUDE, N_NEAR, read

SK = os.path.join(ROOT, 'models', 'skullstrip_check')
TAG = ''                                                   # '' = FreeSurfer（原本的）；'_tigerbx' = tigerbx 版（use('tigerbx')）
OUT_CSV = lambda r: os.path.join(SK, 'residue_volume%s_r%d.csv' % (TAG, r))
OUT_NEAR = lambda r: os.path.join(SK, 'residue_volume%s_r%d_near.json' % (TAG, r))
SUMMARY = lambda r: os.path.join(SK, 'residue_volume%s_r%d_summary.csv' % (TAG, r))
_J = lambda *p: os.path.join(ROOT, *p)
TIGER_COHORTS = [  # 同 COHORTS 的格式：(名字, 殘留 CSV（取 split）, split, 配準前, 配準後, 影像資料夾)
    ('test', os.path.join(SK, 'skullstrip_all520.csv'), 'test', os.path.join(SK, 'cohorts', 'dice_tiger_baseline_test.csv'),
     os.path.join(SK, 'cohorts', 'dice_tiger_exp3_0210_test.csv'), _J('data', 'tigerbx_preprocessed_v2', 'test')),
    ('val', os.path.join(SK, 'skullstrip_all520.csv'), 'val', os.path.join(SK, 'cohorts', 'dice_tiger_baseline_val.csv'),
     os.path.join(SK, 'cohorts', 'dice_tiger_exp3_0210_val.csv'), _J('data', 'tigerbx_preprocessed_v2', 'val')),
]


def use(data):
    """切換資料：'freesurfer'（預設）或 'tigerbx'。"""
    global COHORTS, TAG
    if data == 'tigerbx':
        COHORTS, TAG = TIGER_COHORTS, '_tigerbx'
REGIONS = [('top', '顱頂'), ('base', '顱底'), ('back', '枕部'), ('front', '額部'), ('side', '兩側')]
BANDS = [(0, 2), (2, 5), (5, 10), (10, 999)]          # 離腦表面的距離（mm）：(lo, hi]
CLOSE_R = 5                                            # 「球」的 closing 半徑（mm）


def make_ball(brain, r=CLOSE_R):
    """半徑 r 的 closing（用兩次 distance transform 做，等於圓球形的 structuring element）＋補洞。"""
    dil = ndi.distance_transform_edt(~brain) <= r
    return ndi.binary_fill_holes((ndi.distance_transform_edt(dil) > r) | brain)


def measure_volume(vol, seg, r=CLOSE_R):
    """回傳 (指標, 每區最近結構的 voxel 數, 殘留遮罩, 每個殘留點的區域編號, 球)。"""
    brain = seg > 0
    ball = make_ball(brain, r)
    thr = 0.5 * float(np.median(vol[np.isin(seg, [3, 42])]))
    res = (~ball) & (vol > 0)
    dist, ind = ndi.distance_transform_edt(~ball, return_indices=True)
    _, ind_b = ndi.distance_transform_edt(~brain, return_indices=True)
    pts = np.argwhere(brain)
    c, h = pts.mean(0), (pts.max(0) - pts.min(0)) / 2
    near = ind[:, res].T                                  # 每個殘留點最近的球表面 voxel
    u = (near - c) / h
    ax = np.abs(u).argmax(1)
    sg = np.sign(u[np.arange(len(u)), ax])
    reg = np.select([(ax == 2) & (sg > 0), (ax == 2) & (sg < 0), (ax == 1) & (sg < 0), (ax == 1) & (sg > 0)],
                    [0, 1, 2, 3], default=4)
    d = dist[res]
    lab = seg[tuple(ind_b[:, res])]                       # 每個殘留點最近的有標結構
    m = {'thr': round(thr, 4), 'res_total': int(res.sum()), 'res_bright': int(((~ball) & (vol >= thr)).sum())}
    nearcnt = {}
    for k, (key, _) in enumerate(REGIONS):
        sel = reg == k
        m['res_' + key] = int(sel.sum())
        ll, cc = np.unique(lab[sel], return_counts=True)
        nearcnt[key] = {int(a): int(b) for a, b in zip(ll, cc)}
    for lo, hi in BANDS:
        m['res_d%d_%s' % (lo, hi if hi < 999 else 'up')] = int(((d > lo) & (d <= hi)).sum())
    m['inside_ball'] = int((ball & ~brain & (vol >= thr)).sum())
    return m, nearcnt, res, reg, ball


def subjects():
    """test、val、MRS 三批（同 skullstrip_label_dice_pooled.py，排除 A0131、FSS_A001）；tigerbx 版只有 test、val。"""
    out = []
    for name, res_csv, split, before, after, npz in COHORTS:
        R, B, A = read(res_csv, 'subject'), read(before), read(after)
        for s, r in R.items():
            if (split and r['split'] != split) or s in EXCLUDE or s not in A or s not in B:
                continue
            out.append(dict(cohort=name, subject=s, npz=os.path.join(npz, s + '.npz'), before=B[s], after=A[s]))
    return out


def scan(rad):
    rows, near = [], {}
    S = subjects()
    for k, r in enumerate(S):
        d = np.load(r['npz'])
        m, nc, _, _, _ = measure_volume(d['vol'], d['seg'].astype(int), rad)
        rows.append(dict(cohort=r['cohort'], subject=r['subject'], **m))
        near[r['cohort'] + '/' + r['subject']] = nc
        if (k + 1) % 20 == 0:
            print('%d/%d' % (k + 1, len(S)), flush=True)
    with open(OUT_CSV(rad), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_NEAR(rad), 'w', encoding='utf-8') as f:
        json.dump(near, f)
    print('->', OUT_CSV(rad))
    print('->', OUT_NEAR(rad))


def analyze(rad):
    S = {(r['cohort'], r['subject']): r for r in subjects()}
    V = list(csv.DictReader(open(OUT_CSV(rad), encoding='utf-8')))
    near = json.load(open(OUT_NEAR(rad), encoding='utf-8'))
    names = [c[0] for c in COHORTS]
    avg = lambda row, labs: float(np.nanmean([float(row['label_%d' % l]) for l in labs]))
    out = []
    for key, zh in [('total', '全部')] + REGIONS:
        metric = 'res_' + key
        keys = [k for k, _ in REGIONS] if key == 'total' else [key]
        # 1. 旁邊的結構：每批殘留最多的 N_NEAR 位，每人先算比例再平均
        sel = []
        for cn in names:
            sel += sorted([v for v in V if v['cohort'] == cn], key=lambda v: -int(v[metric]))[:N_NEAR]
        tot = {}
        for v in sel:
            cnt = {}
            for kk in keys:
                for l, c in near[v['cohort'] + '/' + v['subject']][kk].items():
                    cnt[int(l)] = cnt.get(int(l), 0) + c
            s = sum(cnt.values())
            for l, c in cnt.items():
                tot[l] = tot.get(l, 0.0) + (c / s if s else 0) / len(sel)
        share = dict(sorted(tot.items(), key=lambda x: -x[1]))
        labs = [l for l, x in share.items() if x >= MIN_SHARE and l in EVAL]
        # 2. 每人：只平均這些結構的 Dice
        rows = []
        for v in V:
            r = S[(v['cohort'], v['subject'])]
            b, a = avg(r['before'], labs), avg(r['after'], labs)
            rows.append(dict(cohort=v['cohort'], x=float(v[metric]), b=b, a=a, g=a - b))
        # 3. 批內百分位、批內置中（同 skullstrip_label_dice_pooled.py）
        for cn in names:
            rr = [r for r in rows if r['cohort'] == cn]
            xs = np.array([r['x'] for r in rr])
            gm = np.mean([r['g'] for r in rr])
            for r in rr:
                r['pct'] = (np.sum(xs < r['x']) + 0.5 * (np.sum(xs == r['x']) - 1)) / (len(xs) - 1)
                r['gc'] = r['g'] - gm
        res = {'region': zh, 'metric': metric, 'labels': '、'.join(NAME.get(l, str(l)) for l in labs),
               'near_shares': '、'.join('%s %.1f%%' % (NAME.get(l, str(l)), 100 * x) for l, x in share.items() if x >= 0.02),
               'median_mm3': float(np.median([r['x'] for r in rows]))}
        for cn in names:
            rr = [r for r in rows if r['cohort'] == cn]
            rho, p = spearmanr([r['x'] for r in rr], [r['g'] for r in rr])
            res.update({cn + '_n': len(rr), cn + '_median': float(np.median([r['x'] for r in rr])), cn + '_r': rho, cn + '_p': p})
        rho, p = spearmanr([r['pct'] for r in rows], [r['gc'] for r in rows])
        dirty = [r for r in rows if r['pct'] >= 0.75]
        clean = [r for r in rows if r['pct'] <= 0.25]
        res.update({'pooled_n': len(rows), 'pooled_r': rho, 'pooled_p': p, 'dirty_n': len(dirty), 'clean_n': len(clean),
                    'dirty_before': np.mean([r['b'] for r in dirty]), 'dirty_after': np.mean([r['a'] for r in dirty]),
                    'clean_before': np.mean([r['b'] for r in clean]), 'clean_after': np.mean([r['a'] for r in clean]),
                    'mw_p': mannwhitneyu([r['gc'] for r in dirty], [r['gc'] for r in clean]).pvalue})
        res['dirty_gain'] = res['dirty_after'] - res['dirty_before']
        res['clean_gain'] = res['clean_after'] - res['clean_before']
        out.append(res)
        print('%s（%s）中位數 %.0f mm³；旁邊的結構：%s' % (zh, metric, res['median_mm3'], res['near_shares']))
        print('    只平均 %s' % res['labels'])
        print('    各批 r：' + '  '.join('%s %+.2f（p=%.3f，n=%d）' % (cn, res[cn + '_r'], res[cn + '_p'], res[cn + '_n']) for cn in names))
        print('    合起來 r=%+.2f p=%.2g（n=%d）｜殘留最多 1/4：%.3f → %.3f（%+.3f）；最少 1/4：%.3f → %.3f（%+.3f）；組間 p=%.2g'
              % (rho, p, len(rows), res['dirty_before'], res['dirty_after'], res['dirty_gain'],
                 res['clean_before'], res['clean_after'], res['clean_gain'], res['mw_p']))
    p = SUMMARY(rad)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print('->', p)


def show(subject, r=CLOSE_R):
    """一位受試者：三個方向各一刀，殘留（球外、影像非零）一個顏色，不分區（使用者 10-13：「老師想要一起看」）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    rec = [x for x in subjects() if x['subject'] == subject][0]
    d = np.load(rec['npz'])
    vol, seg = d['vol'], d['seg'].astype(int)
    m, _, res, _, ball = measure_volume(vol, seg, r)
    brain = seg > 0
    c = np.argwhere(brain).mean(0).astype(int)
    views = [('冠狀面', lambda a: a[:, c[1], :].T), ('矢狀面（正中）', lambda a: a[c[0], :, :].T),
             ('橫切面', lambda a: a[:, :, c[2] + 25].T)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor='#FAFAF8', gridspec_kw={'width_ratios': [1, 1.15, 1]})
    for ax, (t, f) in zip(axes, views):
        ax.imshow(f(vol), cmap='gray', origin='lower', vmin=0, vmax=1)
        R = f(res).astype(float)
        ax.imshow(np.ma.masked_where(R == 0, R), cmap=ListedColormap(['#E74C3C']), origin='lower', alpha=0.6,
                  interpolation='nearest')
        ax.contour(f(brain.astype(float)), levels=[0.5], colors='#F1C40F', linewidths=0.6)
        ax.contour(f(ball.astype(float)), levels=[0.5], colors='#5DADE2', linewidths=1.3)
        ax.set_title(t, fontsize=12)
        ax.axis('off')
    fig.suptitle('%s：殘留體積 %.1f cm³（球外、影像非零；closing 半徑 %d mm）\n'
                 '黃線＝FreeSurfer seg；藍線＝球（seg 補起腦溝、正中縫）；紅色＝殘留' % (subject, m['res_total'] / 1000, r),
                 fontsize=12.5)
    fig.tight_layout()
    p = os.path.join(SK, 'residue_volume%s_%s_r%d.png' % (TAG, subject, r))
    fig.savefig(p, dpi=120, facecolor='#FAFAF8')
    print('->', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--show')
    ap.add_argument('--r', type=int, default=CLOSE_R, help='「球」的 closing 半徑（mm）')
    ap.add_argument('--data', default='freesurfer', choices=['freesurfer', 'tigerbx'])
    a = ap.parse_args()
    use(a.data)
    if a.scan:
        scan(a.r)
    if a.analyze:
        analyze(a.r)
    if a.show:
        show(a.show, a.r)
    if not (a.scan or a.analyze or a.show):
        ap.error('要給 --scan、--analyze 或 --show')
