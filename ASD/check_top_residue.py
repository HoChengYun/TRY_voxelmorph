# -*- coding: utf-8 -*-
"""頭頂的「殘留」到底是什麼：貼在皮質外面的東西，還是 FreeSurfer 漏標的皮質？（2026-10-01）

§24.2 發現頭頂殘留越多、皮質的模型貢獻越小（170 人 r = -0.40）。但 measure_top 的「殘留」是
「FreeSurfer 標到的腦的最上緣，再往上還亮著的東西」，兩種情況都會被算進去：
  A 真的殘留：硬腦膜／骨髓沒去乾淨，跟皮質中間隔著一條暗的腦脊髓液
  B 漏標的皮質：那層其實是皮質，只是 FreeSurfer 沒標，直接黏著標到的腦

拆法（每條垂直線、只看顱頂區，門檻跟 measure_top 一樣＝灰質中位數的一半）：
  黏著：從標到的腦最上緣往上，一路都亮、中間沒暗過的那一段
  隔開：中間先暗過（< 門檻，就是腦脊髓液那種暗），再亮起來的
  兩段加起來剛好等於 top_vertex_mm。⚠️ 黏著那段每個人都有約 1 格：腦的邊緣那一格一半是灰質、一半是腦脊髓液

再加兩個判斷（B 的話，FreeSurfer 標到的皮質應該缺一塊、變薄）：
  頂端標籤：顱頂區每條垂直線，標到的腦最上面那一格是皮質還是白質
  皮質厚度：同一條線上，白質最上緣到標到的腦最上緣有幾格

輸出（models/skullstrip_check/）：
  top_residue_split.csv     每人：黏著／隔開各幾 mm、亮度（灰質 = 1）、皮質厚度、模型貢獻
  top_residue_zoom.png      每批殘留最多 5 位＋最乾淨 3 位，頭頂放大（冠狀切面，綠 = FreeSurfer 標成皮質）
  top_residue_profile.png   從腦裡面往外走的亮度：殘留最多 1/4 vs 最少 1/4
  top_residue_example.png   解釋用：殘留多的一位 vs 乾淨的一位，原圖 vs 加標記（要加 --example）

用法：python ASD\\check_top_residue.py
      python ASD\\check_top_residue.py --example MRS0381-2 T054
"""
import os
import sys
import csv
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_skullstrip import VERTEX_BAND
from skullstrip_label_dice_pooled import load, COHORTS, SK

CTX, WM = [3, 42], [2, 41]   # 頭頂殘留旁邊的結構就是左右大腦皮質（§24.2）
N_SHOW, N_CLEAN = 5, 3
D = np.arange(-10, 15)       # 亮度剖面：離「標到的腦最上緣」幾格（負 = 腦裡面）
GREEN, RED = '#2FD07A', '#FF3B30'


def split_top(vol, seg):
    gm = float(np.median(vol[np.isin(seg, CTX)]))
    thr = 0.5 * gm
    brain = seg > 0
    W = vol.shape[2]
    z = np.arange(W)[None, None, :]
    has = brain.any(axis=2)
    ztop = np.where(has, (brain * z).max(axis=2), -1)
    above = (z > ztop[:, :, None]) & has[:, :, None]
    bright = above & (vol >= thr)                                   # = measure_top 的殘留
    zdark = np.where(above & (vol < thr), z, W).min(axis=2)         # 往上第一個暗的位置
    cont = bright & (z < zdark[:, :, None])
    sep = bright & (z > zdark[:, :, None])
    vertex = has & (ztop >= ztop[has].max() - VERTEX_BAND)
    vx = vertex[:, :, None]
    zsep = np.where(sep, z, W).min(axis=2)
    gap = (zsep - zdark)[vertex & sep.any(axis=2)]
    # B 的話標到的皮質會缺一塊：看最上面那格標什麼、白質最上緣到腦最上緣有幾格
    xs, ys = np.nonzero(vertex)
    wm = np.isin(seg, WM)
    zwm = np.where(wm.any(axis=2), (wm * z).max(axis=2), -1)[xs, ys]
    zt = ztop[xs, ys]
    zz = np.clip(zt[:, None] + D[None, :], 0, W - 1)
    r = dict(total_mm=float(bright.sum(axis=2)[vertex].mean()),
             cont_mm=float(cont.sum(axis=2)[vertex].mean()),
             sep_mm=float(sep.sum(axis=2)[vertex].mean()),
             cont_int=float(vol[cont & vx].mean() / gm) if (cont & vx).any() else np.nan,
             sep_int=float(vol[sep & vx].mean() / gm) if (sep & vx).any() else np.nan,
             gap_mm=float(np.median(gap)) if gap.size else np.nan,
             top_ctx=float(np.isin(seg[xs, ys, zt], CTX).mean()),
             ctx_thick=float(np.median((zt - zwm)[zwm >= 0])),
             profile=vol[xs[:, None], ys[:, None], zz].mean(axis=0) / gm)
    return r, dict(bright=bright & vx, ctx=np.isin(seg, CTX), ztop=ztop, has=has, vertex=vertex)


def crop(a, x0, x1, z0, z1):
    out = np.zeros((x1 - x0, z1 - z0), a.dtype)
    sx0, sz0 = max(x0, 0), max(z0, 0)
    sx1, sz1 = min(x1, a.shape[0]), min(z1, a.shape[1])
    out[sx0 - x0:sx1 - x0, sz0 - z0:sz1 - z0] = a[sx0:sx1, sz0:sz1]
    return out


def panel(ax, vol, seg, m, title, size=(110, 50), overlay=True):
    """冠狀切面：挑顱頂區殘留最多的那一刀，放大頭頂（預設 110 x 50 mm）。"""
    y = int(m['bright'].sum(axis=(0, 2)).argmax())
    xs = np.nonzero(m['vertex'][:, y])[0]
    xc = int(round(xs.mean())) if xs.size else vol.shape[0] // 2
    zt = int(m['ztop'][:, y][m['has'][:, y]].max())       # 這一刀的腦頂（不是整顆腦的最高點）
    w, h = size
    box = (xc - w // 2, xc + w // 2, zt - int(h * 0.6), zt + int(h * 0.4))
    img = crop(vol[:, y, :], *box)
    vmax = float(np.percentile(vol[seg > 0], 99.5))
    ax.imshow(img.T, origin='lower', cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
    if overlay:
        ctx = crop(m['ctx'][:, y, :], *box).T
        rgba = np.zeros(ctx.shape + (4,))
        rgba[ctx] = to_rgba(GREEN, 0.38)
        ax.imshow(rgba, origin='lower', interpolation='nearest')
        rb = crop(m['bright'][:, y, :], *box)
        if rb.any():
            ax.contour(rb.T.astype(float), [0.5], colors=RED, linewidths=1.4)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def example(heavy_id, clean_id):
    """解釋用的對照圖：殘留多的一位 vs 乾淨的一位，左邊原圖、右邊加標記。
    用法：python ASD\\check_top_residue.py --example MRS0381-2 T054"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    PAPER, INK = '#FAFAF8', '#141A1D'
    rows = {r['subject']: r for r in load()}
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.4), facecolor=PAPER)
    for i, (sid, who, note) in enumerate([
            (heavy_id, '殘留多的人',
             '綠色（皮質）一樣是完整的一條；紅線圈起來的，是綠色外面多出來的一層灰灰的東西，比皮質暗'),
            (clean_id, '乾淨的人', '綠色外面幾乎沒有東西，紅色只有邊緣零星幾點')]):
        d = np.load(rows[sid]['npz'])
        vol, seg = d['vol'], d['seg'].astype(int)
        r, m = split_top(vol, seg)
        panel(axes[i][0], vol, seg, m, '', size=(80, 36), overlay=False)
        panel(axes[i][1], vol, seg, m, '', size=(80, 36), overlay=True)
        axes[i][0].set_ylabel('%s\n（頭頂殘留 %.1f mm）' % (who, r['total_mm']), fontsize=12.5, color=INK)
        axes[i][1].text(0.5, -0.08, note, transform=axes[i][1].transAxes, ha='center', va='top', fontsize=11.5,
                        color=INK)
    axes[0][0].set_title('原圖（頭頂放大，冠狀切面）', fontsize=13, fontweight='bold')
    axes[0][1].set_title('綠 = FreeSurfer 標成皮質　紅線 = 程式算成「殘留」', fontsize=13, fontweight='bold')
    fig.tight_layout(h_pad=2.5)
    p = os.path.join(SK, 'top_residue_example.png')
    fig.savefig(p, dpi=120, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    PAPER, INK, MUTED, RUST, TEAL = '#FAFAF8', '#141A1D', '#5F6A6B', '#A34F1B', '#0E7C7B'

    rows = load()
    names = [c[0] for c in COHORTS]
    print('人數：', {c: sum(r['cohort'] == c for r in rows) for c in names})
    avg = lambda row: float(np.mean([float(row['label_%d' % l]) for l in CTX]))
    for i, r in enumerate(rows):
        d = np.load(r['npz'])
        r.update(split_top(d['vol'], d['seg'].astype(int))[0])
        r['gain'] = avg(r['after']) - avg(r['before'])
        if abs(r['total_mm'] - float(r['res']['top_vertex_mm'])) > 0.01:
            print('[!] %s 跟殘留 CSV 對不上：%.3f vs %s' % (r['subject'], r['total_mm'], r['res']['top_vertex_mm']))
        if (i + 1) % 50 == 0:
            print('%d/%d' % (i + 1, len(rows)), flush=True)

    # 批內百分位、批內置中（同 skullstrip_label_dice_pooled.py）
    for cn in names:
        rr = [r for r in rows if r['cohort'] == cn]
        gm = np.mean([r['gain'] for r in rr])
        for k in ('total_mm', 'cont_mm', 'sep_mm'):
            xs = np.array([r[k] for r in rr])
            for r in rr:
                r[k + '_pct'] = (np.sum(xs < r[k]) + 0.5 * (np.sum(xs == r[k]) - 1)) / (len(xs) - 1)
        for r in rr:
            r['gain_c'] = r['gain'] - gm

    stats = {}
    for k in ('total_mm', 'cont_mm', 'sep_mm'):
        rho, p = spearmanr([r[k + '_pct'] for r in rows], [r['gain_c'] for r in rows])
        per = {cn: spearmanr([r[k] for r in rows if r['cohort'] == cn],
                             [r['gain'] for r in rows if r['cohort'] == cn]) for cn in names}
        stats[k] = (rho, p, per)
        print('%-8s 合起來 r=%+.2f p=%.2g｜' % (k, rho, p)
              + '  '.join('%s %+.2f (p=%.3f)' % (cn, v[0], v[1]) for cn, v in per.items()))

    # 每批殘留最多 N_SHOW 位、全部最乾淨 N_CLEAN 位
    heavy, clean = [], []
    for cn in names:
        rr = sorted([r for r in rows if r['cohort'] == cn], key=lambda r: -r['total_mm'])
        heavy += rr[:N_SHOW]
    clean = sorted(rows, key=lambda r: r['total_mm_pct'] + 1e-3 * r['total_mm'])[:N_CLEAN]
    heavy.sort(key=lambda r: -r['total_mm'])
    for tag, grp in (('殘留最多', heavy), ('最乾淨', clean)):
        print('\n%s：' % tag)
        for r in grp:
            print('  %-12s 殘留 %.2f mm = 黏著 %.2f + 隔開 %.2f｜亮度（灰質=1）黏著 %.2f 隔開 %.2f｜中間暗 %s 格｜貢獻 %+.3f'
                  % (r['subject'], r['total_mm'], r['cont_mm'], r['sep_mm'], r['cont_int'], r['sep_int'],
                     '%.0f' % r['gap_mm'] if r['gap_mm'] == r['gap_mm'] else '—', r['gain']))
    for tag, grp in (('殘留最多 15 位', heavy), ('其他', [r for r in rows if r not in heavy])):
        print('%s：黏著 %.2f mm、隔開 %.2f mm（平均）' % (tag, np.mean([r['cont_mm'] for r in grp]),
                                                     np.mean([r['sep_mm'] for r in grp])))
    hv = [r for r in rows if r['total_mm_pct'] >= 0.75]
    cl = [r for r in rows if r['total_mm_pct'] <= 0.25]
    cmp_ = {}
    for k, name in (('top_ctx', '最上面那格標成皮質的比例'), ('ctx_thick', '白質最上緣到腦最上緣（格）'),
                    ('cont_int', '黏著那段的亮度（皮質 = 1）')):
        a = [r[k] for r in hv if r[k] == r[k]]
        b = [r[k] for r in cl if r[k] == r[k]]
        cmp_[k] = (np.mean(a), np.mean(b), mannwhitneyu(a, b).pvalue if len(set(a + b)) > 1 else 1.0)
        print('%s：殘留最多 1/4（%d 人）%.2f vs 最少 1/4（%d 人）%.2f（p=%.3f）'
              % (name, len(a), cmp_[k][0], len(b), cmp_[k][1], cmp_[k][2]))

    keys = ['cohort', 'subject', 'total_mm', 'cont_mm', 'sep_mm', 'cont_int', 'sep_int', 'gap_mm',
            'top_ctx', 'ctx_thick', 'gain', 'total_mm_pct', 'cont_mm_pct', 'sep_mm_pct', 'gain_c']
    p = os.path.join(SK, 'top_residue_split.csv')
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in sorted(rows, key=lambda r: -r['total_mm']):
            w.writerow([r[k] if isinstance(r[k], str) else '%.4f' % r[k] for k in keys])
    print('->', p)

    # 圖一：頭頂放大
    show = heavy + clean
    nc = 3
    nr = int(np.ceil(len(show) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(16.5, 2.75 * nr + 0.9), facecolor=PAPER)
    for ax in axes.ravel()[len(show):]:
        ax.axis('off')
    for ax, r in zip(axes.ravel(), show):
        d = np.load(r['npz'])
        vol, seg = d['vol'], d['seg'].astype(int)
        _, m = split_top(vol, seg)
        tag = '（對照：最乾淨）' if r in clean else ''
        panel(ax, vol, seg, m, '%s　殘留 %.1f mm%s' % (r['subject'], r['total_mm'], tag))
    fig.legend(handles=[Patch(facecolor=to_rgba(GREEN, 0.6), label='FreeSurfer 標成皮質的'),
                        Line2D([], [], color=RED, lw=2, label='程式算成「殘留」的（邊界）')],
               loc='upper right', ncol=2, frameon=False, fontsize=11.5)
    fig.suptitle('頭頂放大（冠狀切面，每人挑殘留最多的那一刀，110 × 50 mm）', x=0.01, ha='left',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(SK, 'top_residue_zoom.png')
    fig.savefig(p, dpi=120, facecolor=PAPER)
    plt.close(fig)
    print('->', p)

    # 圖二：從腦裡面往頭頂外面走的亮度（顱頂區每條線平均），殘留最多 1/4 vs 最少 1/4
    fig, ax = plt.subplots(figsize=(11, 5.4), facecolor=PAPER)
    ax.axvspan(D[0] - 0.5, 0.5, color=GREEN, alpha=0.10, lw=0)
    for grp, name, col in ((hv, '殘留最多 1/4（%d 人）' % len(hv), RUST), (cl, '殘留最少 1/4（%d 人）' % len(cl), TEAL)):
        P = np.array([r['profile'] for r in grp])
        ax.fill_between(D, np.percentile(P, 25, axis=0), np.percentile(P, 75, axis=0), color=col, alpha=0.15, lw=0)
        ax.plot(D, P.mean(axis=0), color=col, lw=2.6, marker='o', ms=4, label=name)
    ax.axvline(0.5, color=MUTED, lw=1, ls='--')
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=':')
    ax.text(D[-1], 0.53, '超過這條就算成殘留（皮質亮度的一半）', ha='right', va='bottom', fontsize=10.5, color=MUTED)
    ax.text(0.2, 1.02, '← FreeSurfer 標到的腦（最上面是皮質）', transform=ax.get_xaxis_transform(), ha='right',
            fontsize=11.5, color='#1E7A4C', fontweight='bold')
    ax.text(0.8, 1.02, '腦外面 →', transform=ax.get_xaxis_transform(), ha='left', fontsize=11.5, color=INK,
            fontweight='bold')
    ax.text(-9.6, 0.06, '最上面那格標成皮質：%.0f%% vs %.0f%%\n白質最上緣到腦最上緣：%.1f vs %.1f 格（p = %.2f）'
            % (100 * cmp_['top_ctx'][0], 100 * cmp_['top_ctx'][1], cmp_['ctx_thick'][0], cmp_['ctx_thick'][1],
               cmp_['ctx_thick'][2]), fontsize=10.5, color=INK, va='bottom')
    ax.set_xlim(D[0] - 0.5, D[-1] + 0.5)
    ax.set_ylim(0, None)
    ax.set_xlabel('離 FreeSurfer 標到的腦最上緣幾 mm（往上為正）', fontsize=11)
    ax.set_ylabel('亮度（皮質中位數 = 1）', fontsize=11)
    ax.legend(fontsize=11, frameon=False, loc='upper right')
    ax.grid(alpha=.3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.suptitle('從腦裡面往頭頂外面走，亮度怎麼變（顱頂區每條垂直線平均，%d 人）' % len(rows),
                 fontsize=13.5, fontweight='bold', y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(SK, 'top_residue_profile.png')
    fig.savefig(p, dpi=115, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


if __name__ == '__main__':
    if '--example' in sys.argv:
        i = sys.argv.index('--example')
        example(sys.argv[i + 1], sys.argv[i + 2])
    else:
        main()
