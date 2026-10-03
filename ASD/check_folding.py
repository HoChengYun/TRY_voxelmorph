# -*- coding: utf-8 -*-
"""擠爆（|J| <= 0）的點落在哪裡。老師 2026-09-30 紅字：「確認那些位置是被擠爆的（壞掉的點）」。

做法：模型對每位 test 受試者算形變場 → Jacobian 行列式 → <= 0 的體素就是擠爆的點
（算法跟 test_dice.py 的 jacobian_negative_ratio 一樣，擠爆比例會跟 dice_<epoch>.csv 的 jneg_pct 對得上）。
形變場定義在 atlas 的格子上（moved(p) = 受試者(p + u(p))），所以擠爆的點可以直接查 atlas 的 FreeSurfer 標籤。

輸出（models/folding_check/）：
    folding_by_subject.csv   每顆模型 × 每位受試者：擠爆比例、團塊數、最大團塊、在腦內的比例
    folding_by_label.csv     每顆模型 × 每個標籤：擠爆體素數、占全部擠爆的比例、每單位體積多容易擠爆
    heat_<實驗>.npz          每個體素有幾位受試者在這裡擠爆（atlas 空間）
    depth_<實驗>.npz         擠爆的點離腦表面的距離（直方圖）
    folding_where.png        熱圖疊在 atlas 上
    folding_regions.png      按區域分

用法：
    python ASD\check_folding.py --models mix_exp4:0230 mix_exp3:0240 mix_wide:0225 --gpu 0
    python ASD\check_folding.py --plot-only          # 只用存好的結果重畫圖
"""
import os
import sys
import csv
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', default=['mix_exp4:0230', 'mix_exp3:0240', 'mix_wide:0225'],
                help='實驗:epoch，例如 mix_exp3:0240')
ap.add_argument('--test-dir', default=os.path.join(ROOT, 'data', 'mixed_preprocessed_v2', 'test'))
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'folding_check'))
ap.add_argument('--gpu', default='0')
ap.add_argument('--plot-only', action='store_true')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# FreeSurfer aseg 標籤 → 中文名稱
NAME = {0: '未標記', 2: '左大腦白質', 3: '左大腦皮質', 4: '左側腦室', 5: '左側腦室下角', 7: '左小腦白質',
        8: '左小腦皮質', 10: '左視丘', 11: '左尾狀核', 12: '左殼核', 13: '左蒼白球', 14: '第三腦室',
        15: '第四腦室', 16: '腦幹', 17: '左海馬迴', 18: '左杏仁核', 24: '腦脊髓液', 26: '左伏隔核',
        28: '左腹側間腦', 30: '左血管', 31: '左脈絡叢', 41: '右大腦白質', 42: '右大腦皮質', 43: '右側腦室',
        44: '右側腦室下角', 46: '右小腦白質', 47: '右小腦皮質', 49: '右視丘', 50: '右尾狀核', 51: '右殼核',
        52: '右蒼白球', 53: '右海馬迴', 54: '右杏仁核', 58: '右伏隔核', 60: '右腹側間腦', 62: '右血管',
        63: '右脈絡叢', 72: '第五腦室', 77: '白質低信號', 80: '非白質低信號', 85: '視交叉',
        251: '胼胝體', 252: '胼胝體', 253: '胼胝體', 254: '胼胝體', 255: '胼胝體'}
# 區域分組（畫圖用）
GROUPS = [('大腦皮質', {3, 42}),
          ('大腦白質', {2, 41, 77, 251, 252, 253, 254, 255}),
          ('腦室・腦脊髓液・脈絡叢', {4, 5, 14, 15, 24, 43, 44, 72, 31, 63}),
          ('深部灰質・海馬・杏仁核', {10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60}),
          ('小腦・腦幹', {7, 8, 16, 46, 47}),
          ('其他標籤', {30, 62, 80, 85})]
REGION_NAMES = [g for g, _ in GROUPS] + ['腦內、沒有標籤', '腦外（背景）']
DEPTH_BINS = np.arange(-15, 41, 1.0)          # 到腦表面的距離（mm），正 = 腦內


def jac_det(f):
    """Jacobian 行列式。f: torch (3, D, H, W)，差分跟 np.gradient 一樣（test_dice.py 用的算法）。"""
    import torch
    g = [torch.gradient(f[c], dim=(0, 1, 2)) for c in range(3)]
    j11, j12, j13 = 1 + g[0][0], g[0][1], g[0][2]
    j21, j22, j23 = g[1][0], 1 + g[1][1], g[1][2]
    j31, j32, j33 = g[2][0], g[2][1], 1 + g[2][2]
    return (j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31)
            + j13 * (j21 * j32 - j22 * j31))


def atlas_maps():
    """atlas 上每個體素屬於哪個區域、離腦表面多遠。"""
    from scipy import ndimage
    vol = np.load(args.atlas)['vol']
    seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
    brain = (vol > 0.01) | (seg > 0)
    brain = ndimage.binary_fill_holes(brain)
    region = np.full(seg.shape, len(REGION_NAMES) - 1, np.int8)      # 預設腦外
    region[brain & (seg == 0)] = len(REGION_NAMES) - 2               # 腦內沒有標籤
    for i, (_, labs) in enumerate(GROUPS):
        region[np.isin(seg, list(labs))] = i
    depth = ndimage.distance_transform_edt(brain) - ndimage.distance_transform_edt(~brain)
    return vol, seg, brain, region, depth.astype(np.float32)


def run():
    os.environ['NEURITE_BACKEND'] = 'pytorch'
    os.environ['VXM_BACKEND'] = 'pytorch'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    import voxelmorph as vxm
    from scipy import ndimage

    device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
    vol_a, seg, brain, region, depth = atlas_maps()
    a_t = torch.from_numpy(vol_a.astype(np.float32))[None, None].to(device)
    files = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))
    n_vox = seg.size
    lab_ids = np.unique(seg)
    lab_vol = {int(l): int((seg == l).sum()) for l in lab_ids}
    s26 = np.ones((3, 3, 3), bool)

    sub_rows, lab_rows = [], []
    for spec in args.models:
        exp, ep = spec.split(':')
        with open(os.path.join(ROOT, 'models', exp, 'dice_%s.csv' % ep), encoding='utf-8') as fh:
            ref = {r['file'][:-4]: float(r['jneg_pct']) for r in csv.DictReader(fh)}
        model = vxm.networks.VxmDense.load(os.path.join(ROOT, 'models', exp, ep + '.pt'), device)
        model.to(device)
        model.eval()
        heat = np.zeros(seg.shape, np.uint8)
        lab_cnt = np.zeros(int(seg.max()) + 1, np.int64)
        reg_cnt = np.zeros(len(REGION_NAMES), np.int64)
        dhist = np.zeros(len(DEPTH_BINS) - 1, np.int64)
        for f in files:
            sid = os.path.basename(f)[:-4]
            v = torch.from_numpy(np.load(f)['vol'].astype(np.float32))[None, None].to(device)
            with torch.no_grad():
                _, flow = model(v, a_t, registration=True)
                fold = (jac_det(flow[0]) <= 0).cpu().numpy()
            n = int(fold.sum())
            heat += fold.astype(np.uint8)
            lab_cnt += np.bincount(seg[fold], minlength=lab_cnt.size)
            reg_cnt += np.bincount(region[fold], minlength=len(REGION_NAMES))
            dhist += np.histogram(depth[fold], bins=DEPTH_BINS)[0]
            if n:
                cc, ncc = ndimage.label(fold, structure=s26)
                sizes = np.bincount(cc.ravel())[1:]
                big = int(sizes.max())
                in_brain = float(brain[fold].mean())
                med_depth = float(np.median(depth[fold]))
            else:
                ncc, big, in_brain, med_depth = 0, 0, float('nan'), float('nan')
            sub_rows.append({'exp': exp, 'subject': sid, 'jneg_pct': 100.0 * n / n_vox,
                             'jneg_pct_csv': ref.get(sid, float('nan')), 'n_folded': n,
                             'n_clusters': int(ncc), 'largest_cluster': big,
                             'frac_in_brain': in_brain, 'median_depth_mm': med_depth})
            print('  %-9s %-10s 擠爆 %.4f%%（CSV %.4f%%）  %5d 點  %4d 團  最大 %3d 點'
                  % (exp, sid, 100.0 * n / n_vox, ref.get(sid, float('nan')), n, ncc, big))
        tot = max(int(lab_cnt.sum()), 1)
        for l in lab_ids:
            l = int(l)
            lab_rows.append({'exp': exp, 'label': l, 'name': NAME.get(l, str(l)), 'voxels': lab_vol[l],
                             'folded_total': int(lab_cnt[l]), 'share_pct': 100.0 * lab_cnt[l] / tot,
                             'rate_per_10k': 1e4 * lab_cnt[l] / (lab_vol[l] * len(files))})
        np.savez_compressed(os.path.join(args.out, 'heat_%s.npz' % exp), heat=heat, n_subjects=len(files))
        np.savez_compressed(os.path.join(args.out, 'depth_%s.npz' % exp), hist=dhist, bins=DEPTH_BINS,
                            region_counts=reg_cnt, region_names=np.array(REGION_NAMES),
                            region_voxels=np.bincount(region.ravel(), minlength=len(REGION_NAMES)))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    for name, rows in (('folding_by_subject.csv', sub_rows), ('folding_by_label.csv', lab_rows)):
        with open(os.path.join(args.out, name), 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('->', os.path.join(args.out, name))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from orient import canonical_axes, to_ras
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
    COL = {'mix_exp4': '#D9895A', 'mix_exp3': '#A34F1B', 'mix_wide': '#6B3FA0'}
    LAB = {'mix_exp4': '位移場・權重 2', 'mix_exp3': '位移場・權重 1', 'mix_wide': '位移場・權重 1・加寬'}
    exps = [s.split(':')[0] for s in args.models]

    vol, seg, brain, region, depth = atlas_maps()
    perm, flip = canonical_axes(seg)
    vol_r, brain_r = to_ras(vol, perm, flip), to_ras(brain.astype(np.uint8), perm, flip)

    # ── 熱圖：每一列一顆模型，直接切片（不投影），只標「至少 MIN_N 位在同一點擠爆」的地方 ──
    # 投影會把整條視線上任何一位的擠爆都疊上來，整顆腦都亮，看不出位置（2026-09-30 第一版就是這樣）
    MIN_N = 3
    heats = {e: to_ras(np.load(os.path.join(args.out, 'heat_%s.npz' % e))['heat'], perm, flip) for e in exps}
    vmax = min(15, max(int(h.max()) for h in heats.values()))
    idx = np.argwhere(brain_r > 0)
    z0, z1 = idx[:, 2].min(), idx[:, 2].max()
    views = [('軸狀・側腦室那層', 2, int(z0 + 0.50 * (z1 - z0))),
             ('軸狀・再往上', 2, int(z0 + 0.68 * (z1 - z0))),
             ('軸狀・接近頭頂', 2, int(z0 + 0.85 * (z1 - z0))),
             ('冠狀・中間', 1, int(np.median(idx[:, 1])))]
    fig, axes = plt.subplots(len(exps), len(views), figsize=(4.0 * len(views), 4.4 * len(exps)), facecolor=PAPER)
    axes = np.atleast_2d(axes)
    take = lambda a, ax_id, i: [a[i], a[:, i], a[:, :, i]][ax_id]
    for r, e in enumerate(exps):
        for c, (title, ax_id, i) in enumerate(views):
            ax = axes[r, c]
            hm = take(heats[e], ax_id, i).astype(float)
            ax.imshow(take(vol_r, ax_id, i).T, cmap='gray', origin='lower', vmin=0, vmax=1)
            ax.imshow(np.ma.masked_less(hm, MIN_N).T, cmap='autumn_r', origin='lower', vmin=MIN_N, vmax=vmax,
                      interpolation='nearest')
            ax.axis('off')
            if r == 0:
                ax.set_title(title, fontsize=13, fontweight='bold')
            if c == 0:
                ax.text(-0.03, 0.5, LAB.get(e, e), transform=ax.transAxes, rotation=90, ha='right',
                        va='center', fontsize=13, fontweight='bold', color=COL.get(e, INK))
    sm = plt.cm.ScalarMappable(cmap='autumn_r', norm=plt.Normalize(MIN_N, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label('同一個點上，51 位裡有幾位擠爆（%d 位以上才標）' % MIN_N, fontsize=11)
    fig.suptitle('擠爆的點在哪裡：51 位 test 疊在 atlas 上，只標 %d 位以上在同一點擠爆的地方' % MIN_N,
                 fontsize=15, fontweight='bold')
    fig.savefig(os.path.join(args.out, 'folding_where.png'), dpi=110, facecolor=PAPER, bbox_inches='tight')
    plt.close(fig)
    print('->', os.path.join(args.out, 'folding_where.png'))

    # ── 按區域：擠爆的點有幾成落在這一區／這一區每單位體積多容易擠爆 ──
    # 「其他標籤」（血管、低信號、視交叉）加起來不到 0.1%，體積又小，比例會失真，不畫
    show_r = [i for i, n in enumerate(REGION_NAMES) if n != '其他標籤']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=PAPER)
    wbar = 0.8 / len(exps)
    for j, e in enumerate(exps):
        z = np.load(os.path.join(args.out, 'depth_%s.npz' % e))
        cnt, vox = z['region_counts'].astype(float), z['region_voxels'].astype(float)
        share = 100 * cnt / max(cnt.sum(), 1)
        n_sub = int(np.load(os.path.join(args.out, 'heat_%s.npz' % e))['n_subjects'])
        rate = 1e4 * cnt / np.maximum(vox, 1) / n_sub
        x = np.arange(len(show_r)) + (j - (len(exps) - 1) / 2) * wbar
        axes[0].barh(x, share[show_r], height=wbar * .9, color=COL.get(e), label=LAB.get(e, e))
        axes[1].barh(x, rate[show_r], height=wbar * .9, color=COL.get(e), label=LAB.get(e, e))
    for ax, t in ((axes[0], '擠爆的點有幾 % 落在這一區'), (axes[1], '這一區每 1 萬個體素有幾個擠爆')):
        ax.set_yticks(range(len(show_r)))
        ax.set_yticklabels([REGION_NAMES[i] for i in show_r], fontsize=11)
        ax.invert_yaxis()
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=.3)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=11, frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'folding_regions.png'), dpi=120, facecolor=PAPER)
    plt.close(fig)
    print('->', os.path.join(args.out, 'folding_regions.png'))

    # ── 離腦表面多遠 ──
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=PAPER)
    for e in exps:
        z = np.load(os.path.join(args.out, 'depth_%s.npz' % e))
        h, b = z['hist'].astype(float), z['bins']
        ax.plot((b[:-1] + b[1:]) / 2, 100 * h / max(h.sum(), 1), color=COL.get(e), lw=2.2, label=LAB.get(e, e))
    ax.axvline(0, color=MUTED, ls='--', lw=1)
    ax.text(0.3, ax.get_ylim()[1] * 0.92, '← 腦外　　腦表面　　腦內 →', fontsize=11, color=MUTED)
    ax.set_xlabel('離腦表面的距離（mm，正 = 腦內）', fontsize=11)
    ax.set_ylabel('占全部擠爆的 %', fontsize=11)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(alpha=.3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'folding_depth.png'), dpi=120, facecolor=PAPER)
    plt.close(fig)
    print('->', os.path.join(args.out, 'folding_depth.png'))


def zoom(subject='T054', spec='mix_exp3:0240', half=16, step=2):
    """拿一位受試者，放大看他最大的一團擠爆點：格線在那裡交叉、翻過去。

    背景是受試者影像；黃線＝atlas 上每隔 step 格的正方格子，照形變場搬到受試者身上的樣子（同 visualize_reg_ixi.py）；
    紅點＝擠爆的點被搬到的位置。擠爆＝格子翻過去，所以紅點附近的黃線會交叉或擠成一團。
    3D 的擠爆不一定每個切面都看得到交叉，所以三個方向各切一刀。
    """
    os.environ['NEURITE_BACKEND'] = 'pytorch'
    os.environ['VXM_BACKEND'] = 'pytorch'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    import voxelmorph as vxm
    from scipy import ndimage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from orient import canonical_axes, to_ras, flow_to_ras
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    exp, ep = spec.split(':')
    device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
    atlas = np.load(args.atlas)['vol'].astype(np.float32)
    seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
    vol = np.load(os.path.join(args.test_dir, subject + '.npz'))['vol'].astype(np.float32)
    model = vxm.networks.VxmDense.load(os.path.join(ROOT, 'models', exp, ep + '.pt'), device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        _, flow = model(torch.from_numpy(vol)[None, None].to(device),
                        torch.from_numpy(atlas)[None, None].to(device), registration=True)
        det = jac_det(flow[0]).cpu().numpy()
    perm, flip = canonical_axes(seg)
    src = to_ras(vol, perm, flip)
    det = to_ras(det, perm, flip)
    u = flow_to_ras(flow[0].cpu().numpy(), perm, flip)
    seg_r = to_ras(seg, perm, flip)
    fold = det <= 0
    cc, n = ndimage.label(fold, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    big = int(sizes.argmax())
    center = np.round(ndimage.center_of_mass(cc == big)).astype(int)
    lab = NAME.get(int(seg_r[tuple(center)]), str(int(seg_r[tuple(center)])))

    take = lambda a, ax_id, i: [a[i], a[:, i], a[:, :, i]][ax_id]
    planes = [('軸狀', 2, (0, 1)), ('冠狀', 1, (0, 2)), ('矢狀', 0, (1, 2))]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), facecolor='#FAFAF8')
    for ax, (name, ax_id, (p, q)) in zip(axes, planes):
        i = center[ax_id]
        img, fm = take(src, ax_id, i), take(fold, ax_id, i)
        up, uq = take(u[p], ax_id, i), take(u[q], ax_id, i)
        a0, a1 = center[p] - half, center[p] + half
        b0, b1 = center[q] - half, center[q] + half
        ax.imshow(img.T, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='nearest')
        for j in range(b0, b1 + 1, step):                       # 橫線
            xs = np.arange(a0, a1 + 1)
            ax.plot(xs + up[a0:a1 + 1, j], j + uq[a0:a1 + 1, j], color='#FFD23F', lw=1.2)
        for k in range(a0, a1 + 1, step):                       # 直線
            ys = np.arange(b0, b1 + 1)
            ax.plot(k + up[k, b0:b1 + 1], ys + uq[k, b0:b1 + 1], color='#FFD23F', lw=1.2)
        pp, qq = np.nonzero(fm[a0:a1 + 1, b0:b1 + 1])
        pp, qq = pp + a0, qq + b0
        ax.scatter(pp + up[pp, qq], qq + uq[pp, qq], s=26, color='#E0262D', zorder=5, edgecolors='none')
        ax.set_xlim(a0 - 3, a1 + 3)
        ax.set_ylim(b0 - 3, b1 + 3)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('%s（這個切面有 %d 個擠爆點）' % (name, int(fm[a0:a1 + 1, b0:b1 + 1].sum())),
                     fontsize=13, fontweight='bold')
    fig.suptitle('%s 最大的一團擠爆點（%d 個點，在%s）放大來看：黃線＝格子，紅點＝擠爆的點'
                 % (subject, int(sizes[big]), lab), fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, '%s（%s）。每格 = %d mm。擠爆＝格子翻過去，紅點附近的黃線會交叉或擠在一起。'
             % (exp, ep, step), ha='center', fontsize=11.5, color='#5F6A6B')
    out = os.path.join(args.out, 'folding_zoom_%s.png' % subject)
    fig.savefig(out, dpi=115, facecolor='#FAFAF8', bbox_inches='tight')
    plt.close(fig)
    print('->', out, '｜最大一團 %d 點，中心 %s，%s' % (int(sizes[big]), center.tolist(), lab))


if __name__ == '__main__':
    if not args.plot_only:
        run()
    plot()
    zoom()
