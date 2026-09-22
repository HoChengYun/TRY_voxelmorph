# -*- coding: utf-8 -*-
"""去顱骨有沒有去乾淨，會不會影響非線性對位。

老師的問題：「去頭殼好壞會怎麼影響非線性對位的結果」，他點名兩個位置：
    上緣  —— 皮質上方還黏著一層硬腦膜／骨髓
    顱底  —— 小腦下方留了一整塊

所以量兩個指標（都只看影像本身，不跟別的工具比）：

  top_vertex_mm  上緣：每條垂直線找最高的腦組織，往上數還有幾 mm 是「組織」。
                 組織的門檻＝該受試者灰質中位數的一半 —— 用 0.02 會連腦脊髓液
                 都算進去，整片都被標紅。不要求跟腦相連，硬腦膜跟皮質中間本來
                 就隔著一層腦脊髓液。顱頂區＝最高點落在全腦最高點 25mm 以內的線。

  base_blob10    顱底：離已標記腦組織 10mm 以外還亮著的東西，最大一坨多大。
                 10mm 以外不可能是腦或緊貼腦的硬腦膜，一定是留下來的。
                 （>4mm 抓到的是每個人都有的硬腦膜那圈，分不出好壞，所以不用）

⚠️ 用 seg 當「腦組織」的定義，只能用在有 seg 的 npz（preprocess_fs.py 產生的）。
⚠️ 距離直接以體素數當 mm（前處理後是精確 1mm）。
⚠️ Dice 只有測試集有，所以出圖預設只挑測試集的人。

用法
----
    python ASD\\check_skullstrip.py --scan data\\mixed_preprocessed_v2 ^
        --out models\\skullstrip_check\\skullstrip_all520.csv

    python ASD\\check_skullstrip.py --show top --csv models\\skullstrip_check\\skullstrip_all520.csv ^
        --dice models\\mix_exp3\\dice_0240.csv --baseline models\\mix_exp2\\dice_baseline.csv ^
        --out models\\skullstrip_check\\top_compare.png
"""
import os
import csv
import glob
import argparse
import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERY_FAR = 10          # 顱底：離腦幾 mm 以外算殘留
VERTEX_BAND = 25       # 上緣：顱頂區的範圍


def measure_top(vol, seg):
    """上緣：皮質上方還留著多少組織。回傳 (指標, 標記, 厚度地圖, 有腦的線)。"""
    gm = np.median(vol[np.isin(seg, [3, 42])])
    thr = 0.5 * gm
    brain = seg > 0
    W = vol.shape[2]
    z = np.arange(W)[None, None, :]
    has = brain.any(axis=2)
    ztop = np.where(has, (brain * z).max(axis=2), -1)
    tissue = (z > ztop[:, :, None]) & has[:, :, None] & (vol >= thr)
    thick = tissue.sum(axis=2).astype(float)
    vertex = has & (ztop >= ztop[has].max() - VERTEX_BAND)
    m = dict(top_vertex_mm=round(float(thick[vertex].mean()), 3),
             top_all_mm=round(float(thick[has].mean()), 3),
             top_p95_mm=round(float(np.percentile(thick[vertex], 95)), 3))
    return m, tissue, thick, has


def measure_base(vol, seg):
    """顱底：離腦 10mm 以外還亮著的東西。回傳 (指標, 標記)。"""
    mask = vol > 0.02
    dist = ndi.distance_transform_edt(seg == 0)
    far, very = mask & (dist > 4), mask & (dist > VERY_FAR)
    lab, n = ndi.label(very)
    blob = int(np.bincount(lab.ravel())[1:].max()) if n else 0
    m = dict(base_blob10=blob,
             base_far10_pct=round(100 * very.sum() / mask.sum(), 3),
             base_far4_pct=round(100 * far.sum() / mask.sum(), 3))
    return m, very


def find(root, name):
    for sp in ('train', 'val', 'test', ''):
        p = os.path.join(root, sp, name + '.npz')
        if os.path.exists(p):
            return p, (sp or '-')
    raise SystemExit('[X] 找不到 %s（在 %s 底下）' % (name, root))


def scan(root, out):
    files = []
    for sp in ('train', 'val', 'test'):
        files += [(sp, p) for p in sorted(glob.glob(os.path.join(root, sp, '*.npz')))]
    if not files:
        files = [('-', p) for p in sorted(glob.glob(os.path.join(root, '*.npz')))]
    rows = []
    for k, (sp, p) in enumerate(files):
        d = np.load(p)
        vol, seg = d['vol'], d['seg']
        t, _, _, _ = measure_top(vol, seg)
        b, _ = measure_base(vol, seg)
        rows.append(dict(subject=os.path.basename(p)[:-4], split=sp, **t, **b))
        if (k + 1) % 50 == 0:
            print('%d/%d' % (k + 1, len(files)), flush=True)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('->', out)


def read_dice(p):
    if not p:
        return {}
    with open(p, encoding='utf-8') as f:
        return {r['file'].replace('.npz', ''): float(r['dice_mean']) for r in csv.DictReader(f)}


def pick(csv_path, metric, n, split, exclude):
    with open(csv_path, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f)
                if (split in ('all', r['split'])) and r['subject'] not in exclude]
    key = 'top_vertex_mm' if metric == 'top' else 'base_blob10'
    rows.sort(key=lambda r: -float(r[key]))
    return rows[:n], rows[-n:]


def show(root, metric, worst, clean, out, dice, base, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    def take(a, ax, i):
        return [a[i], a[:, i], a[:, :, i]][ax].T

    rows = [(r, '沒去乾淨') for r in worst] + [(r, '去得乾淨') for r in clean]
    n = len(rows)
    # 第一欄是文字（Dice 寫大一點），後面五欄才是影像
    fig, allax = plt.subplots(n, 6, figsize=(19.5, 2.9 * n), squeeze=False,
                              gridspec_kw={'width_ratios': [0.72, 1, 1, 1, 1, 1]})
    ax = allax[:, 1:]
    for r, (meta, grp) in enumerate(rows):
        s = meta['subject']
        p, sp = find(root, s)
        d = np.load(p)
        vol, seg = d['vol'], d['seg']
        if metric == 'top':
            _, mark, thick, has = measure_top(vol, seg)
            note = '顱頂殘留 %.2f mm' % float(meta['top_vertex_mm'])
        else:
            _, mark = measure_base(vol, seg)
            thick = has = None
            note = '最大一坨 %s 顆' % meta['base_blob10']
        D, H, W = vol.shape
        ztop = np.argwhere(seg > 0)[:, 2].max()
        crop = int(W * 0.56) if metric == 'top' else 0
        panels = [('矢狀（偏 30mm）', 0, D // 2 - 30, True),
                  ('冠狀（中）', 1, H // 2, True),
                  ('冠狀・沒塗色', 1, H // 2, False)]
        for c, (t, a_, i, paint) in enumerate(panels):
            rgb = np.dstack([take(vol, a_, i)] * 3)
            if paint:
                rgb[take(mark, a_, i)] = [1.0, 0.15, 0.1]
            ax[r][c].imshow(rgb[crop:, :], origin='lower')
            ax[r][c].axis('off')
            if r == 0:
                ax[r][c].set_title(t, fontsize=13)
        if metric == 'top':
            ax[r][3].imshow(take(vol, 2, ztop - 12), cmap='gray', origin='lower', vmin=0, vmax=1)
            t3 = '軸狀・頭頂下 12mm'
            im = ax[r][4].imshow(np.where(has, thick, np.nan).T, origin='lower',
                                 cmap='inferno', vmin=0, vmax=8)
            plt.colorbar(im, ax=ax[r][4], fraction=0.046, label='mm')
            t4 = '從頭頂往下看'
        else:
            ax[r][3].imshow(take(vol, 2, W // 2), cmap='gray', origin='lower', vmin=0, vmax=1)
            t3 = '軸狀（中）'
            ax[r][4].imshow(mark.max(axis=0).T, cmap='hot', origin='lower')
            t4 = '殘留壓成一張（側視）'
        ax[r][3].axis('off')
        ax[r][4].axis('off')
        if r == 0:
            ax[r][3].set_title(t3, fontsize=13)
            ax[r][4].set_title(t4, fontsize=13)
        col = '#A34F1B' if grp == '沒去乾淨' else '#0E7C7B'
        tx = allax[r][0]
        tx.axis('off')
        tx.text(.5, .93, grp, ha='center', va='top', fontsize=16, fontweight='bold', color=col)
        tx.text(.5, .76, '%s（%s）' % (s, sp), ha='center', va='top', fontsize=13, color='#141A1D')
        tx.text(.5, .64, note, ha='center', va='top', fontsize=12, color='#5F6A6B')
        if s in dice and s in base:
            tx.text(.5, .45, 'Dice %.3f' % dice[s], ha='center', va='center',
                    fontsize=26, fontweight='bold', color=col)
            tx.text(.5, .28, '起點 %.3f（%+.3f）' % (base[s], dice[s] - base[s]),
                    ha='center', va='center', fontsize=12.5, color='#5F6A6B')
    fig.suptitle(title, fontsize=15.5, fontweight='bold')
    plt.tight_layout(rect=[0.01, 0, 1, 0.985])
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plt.savefig(out, dpi=105, bbox_inches='tight')
    plt.close()
    print('->', out)


def explain(root, top_subject, base_subject, out):
    """畫一張說明圖：這兩個指標到底是怎麼算出來的。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    BLUE, RED = np.array([0.15, 0.45, 0.95]), np.array([1.0, 0.15, 0.1])

    def tint(gray, m, c, a=0.6):
        rgb = np.dstack([gray] * 3)
        rgb[m] = (1 - a) * rgb[m] + a * c
        return rgb

    fig, ax = plt.subplots(2, 4, figsize=(18, 8.6))

    # ── 上排：上緣 ─────────────────────────────────────────────
    p, _ = find(root, top_subject)
    d = np.load(p)
    vol, seg = d['vol'], d['seg']
    _, tissue, _, has = measure_top(vol, seg)
    gm = np.median(vol[np.isin(seg, [3, 42])])
    D, H, W = vol.shape
    i = H // 2
    crop = int(W * 0.58)
    g = vol[:, i].T[crop:]
    b = (seg[:, i] > 0).T[crop:]
    t = tissue[:, i].T[crop:]
    z = np.arange(W)[None, :]
    ztop = np.where((seg > 0).any(axis=2), ((seg > 0) * z[None]).max(axis=2), -1)

    ax[0][0].imshow(g, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax[0][0].set_title('① 原始影像', fontsize=14, fontweight='bold')
    ax[0][1].imshow(tint(g, b, BLUE), origin='lower')
    ax[0][1].set_title('② 藍＝FreeSurfer 標到的腦', fontsize=14, fontweight='bold')
    ax[0][2].imshow(tint(g, b, BLUE, .35), origin='lower')
    ax[0][2].plot(np.arange(D), ztop[:, i] - crop, color='#FFD400', lw=2.2)
    ax[0][2].set_xlim(0, D - 1)
    ax[0][2].set_ylim(0, W - crop - 1)
    ax[0][2].set_title('③ 黃線＝每一欄最高的腦組織', fontsize=14, fontweight='bold')
    ax[0][3].imshow(tint(g, t, RED, .75), origin='lower')
    ax[0][3].set_title('④ 紅＝黃線以上、亮度達到灰質一半', fontsize=14, fontweight='bold')
    ax[0][0].text(-0.06, 0.5, '上緣\n%s' % top_subject, transform=ax[0][0].transAxes,
                  rotation=90, va='center', ha='center', fontsize=15, fontweight='bold', color='#A34F1B')
    ax[0][3].text(0.5, -0.06, '數紅色有幾格 = 這一欄留了幾 mm。顱頂那一圈取平均 → %.2f mm'
                  % measure_top(vol, seg)[0]['top_vertex_mm'],
                  transform=ax[0][3].transAxes, ha='center', va='top', fontsize=12.5, color='#5F6A6B')
    ax[0][1].text(0.5, -0.06, '亮度門檻用「這個人自己的灰質中位數 × 0.5」＝ %.3f\n'
                              '用 0.02 會連腦脊髓液都算進去' % (0.5 * gm),
                  transform=ax[0][1].transAxes, ha='center', va='top', fontsize=12.5, color='#5F6A6B')

    # ── 下排：顱底 ─────────────────────────────────────────────
    p, _ = find(root, base_subject)
    d = np.load(p)
    vol, seg = d['vol'], d['seg']
    dist = ndi.distance_transform_edt(seg == 0)
    very = (vol > 0.02) & (dist > VERY_FAR)
    lab, n = ndi.label(very)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    big = lab == sizes.argmax()
    D, H, W = vol.shape
    i = D // 2
    g = vol[i].T
    b = (seg[i] > 0).T
    dm = dist[i].T
    v = very[i].T
    bg = big[i].T

    ax[1][0].imshow(g, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax[1][0].set_title('① 原始影像', fontsize=14, fontweight='bold')
    ax[1][1].imshow(tint(g, b, BLUE), origin='lower')
    ax[1][1].set_title('② 藍＝FreeSurfer 標到的腦', fontsize=14, fontweight='bold')
    im = ax[1][2].imshow(np.clip(dm, 0, 20), cmap='viridis', origin='lower')
    ax[1][2].set_title('③ 每個點離腦多遠（mm）', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax[1][2], fraction=0.046)
    r = tint(g, v, RED, .75)
    r[bg] = [1.0, 0.85, 0.0]
    ax[1][3].imshow(r, origin='lower')
    ax[1][3].set_title('④ 紅＝亮且離腦 >10mm，黃＝最大一坨', fontsize=14, fontweight='bold')
    ax[1][0].text(-0.06, 0.5, '顱底\n%s' % base_subject, transform=ax[1][0].transAxes,
                  rotation=90, va='center', ha='center', fontsize=15, fontweight='bold', color='#A34F1B')
    ax[1][3].text(0.5, -0.05, '數黃色有幾顆 = %d 顆。只取最大一坨，才不會被零星雜點灌水'
                  % int(big.sum()), transform=ax[1][3].transAxes, ha='center', va='top',
                  fontsize=12.5, color='#5F6A6B')
    ax[1][2].text(0.0, -0.05, '10mm 以外不可能是腦，也不可能是貼著腦的硬腦膜',
                  transform=ax[1][2].transAxes, ha='left', va='top', fontsize=12.5, color='#5F6A6B')

    for a_ in ax.ravel():
        a_.set_xticks([])
        a_.set_yticks([])
        for sp_ in a_.spines.values():
            sp_.set_visible(False)
    fig.suptitle('殘留物是怎麼算出來的', fontsize=17, fontweight='bold')
    plt.tight_layout(rect=[0.015, 0, 1, 0.965])
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close()
    print('->', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', metavar='DIR')
    ap.add_argument('--show', choices=['top', 'base', 'how'])
    ap.add_argument('--csv', help='--show 時讀這份掃描結果來挑人')
    ap.add_argument('--n', type=int, default=3, help='最嚴重／最乾淨各挑幾位')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--exclude', nargs='*', default=['A0131'],
                    help='不放進來的受試者（A0131 起點 0.563，離其他人一大截）')
    ap.add_argument('--dice', help='配準後的 dice CSV')
    ap.add_argument('--baseline', help='線性對位的 dice CSV')
    ap.add_argument('--data-root', default=os.path.join(ROOT, 'data', 'mixed_preprocessed_v2'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    if a.scan:
        scan(a.scan, a.out)
    elif a.show == 'how':
        explain(a.data_root, 'sub-0043', 'sub-0038', a.out)
    elif a.show:
        if not a.csv:
            ap.error('--show 要配 --csv')
        w, c = pick(a.csv, a.show, a.n, a.split, set(a.exclude))
        head = {'top': '上緣：皮質上方還留著的組織（紅色）',
                'base': '顱底：離腦 10mm 以外還留著的東西（紅色）'}[a.show]
        show(a.data_root, a.show, w, c, a.out, read_dice(a.dice), read_dice(a.baseline),
             head + '　—　測試集，上 %d 位沒去乾淨／下 %d 位去得乾淨' % (len(w), len(c)))
    else:
        ap.error('要給 --scan 或 --show')
