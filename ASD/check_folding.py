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
    folding_views.png        一顆（預設 mix_exp3），軸狀／冠狀／矢狀各 4 刀（--views）
    folding_params.png       不同設定（速度場權重 1、0.5，位移場權重 2、1、加寬）× 三個方向（--views）

用法：
    python ASD\check_folding.py --models mix_exp4:0230 mix_exp3:0240 mix_wide:0225 --gpu 0
    python ASD\check_folding.py --models mix_exp6:0190 mix_exp7:0250 --gpu 0   # 補算別顆，CSV 裡其他模型的列會留著
    python ASD\check_folding.py --plot-only          # 只用存好的結果重畫圖
    python ASD\check_folding.py --plot-only --views  # 畫不同方向、不同設定那兩張
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
ap.add_argument('--plot-models', nargs='+', default=['mix_exp4', 'mix_exp3', 'mix_wide'],
                help='folding_where / regions / depth 這三張圖畫哪幾顆（要先算過）')
ap.add_argument('--views', action='store_true',
                help='改畫 folding_views.png（一顆、三個方向各 4 刀）和 folding_params.png（不同設定 × 三個方向）')
ap.add_argument('--view-one', default='mix_exp3', help='folding_views.png 畫哪一顆')
ap.add_argument('--zoom-pair', action='store_true',
                help='只畫 folding_zoom_pair_T054.png：加寬位移場 vs 加寬速度場，放大在同一個位置（要 GPU）')
ap.add_argument('--view-models', nargs='+',
                default=['mix_exp6', 'mix_exp7', 'mix_wide_vel', 'mix_exp4', 'mix_exp3', 'mix_wide'],
                help='folding_params.png 畫哪幾顆（由左到右；左邊三顆速度場、右邊三顆位移場）')
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
    from arch import load_model           # 新架構（串接等）也讀得到；舊模型照舊是 VxmDense.load
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
        model = load_model(os.path.join(ROOT, 'models', exp, ep + '.pt'), device)
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
        p = os.path.join(args.out, name)
        # 只換掉這次有算的模型，其他模型的列留著（2026-10-04 補算速度場那幾顆時，位移場三顆的結果不能被洗掉）
        redo = {r['exp'] for r in rows}
        old = []
        if os.path.exists(p):
            with open(p, encoding='utf-8') as fh:
                old = [r for r in csv.DictReader(fh) if r['exp'] not in redo]
        with open(p, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(old + rows)
        print('->', p, '（保留其他模型 %d 列）' % len(old))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from orient import canonical_axes, to_ras
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
    plt.rcParams['axes.unicode_minus'] = False
    INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
    COL = {'mix_exp4': '#D9895A', 'mix_exp3': '#A34F1B', 'mix_wide': '#6B3FA0'}
    LAB = {'mix_exp4': '位移場・權重 2', 'mix_exp3': '位移場・權重 1', 'mix_wide': '位移場・權重 1・加寬'}
    exps = list(args.plot_models)       # 跟 --models（要重算哪幾顆）分開：補算速度場時，這三張圖照樣畫位移場三顆

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
    from arch import load_model           # 新架構（串接等）也讀得到；舊模型照舊是 VxmDense.load
    from scipy import ndimage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from orient import canonical_axes, to_ras, flow_to_ras
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
    plt.rcParams['axes.unicode_minus'] = False

    exp, ep = spec.split(':')
    device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
    atlas = np.load(args.atlas)['vol'].astype(np.float32)
    seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
    vol = np.load(os.path.join(args.test_dir, subject + '.npz'))['vol'].astype(np.float32)
    model = load_model(os.path.join(ROOT, 'models', exp, ep + '.pt'), device)
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
    planes = [('軸狀面', 2, (0, 1)), ('冠狀面', 1, (0, 2)), ('矢狀面', 0, (1, 2))]
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
        ax.set_title('%s（本切面 %d 個 folding voxels）' % (name, int(fm[a0:a1 + 1, b0:b1 + 1].sum())),
                     fontsize=13, fontweight='bold')
    # 2026-10-07 起圖上的字用正式用語（簡報要給老師看）
    fig.suptitle('%s 最大之 folding 團簇（%d voxels，%s）局部放大：黃線＝形變網格，紅點＝folding voxel'
                 % (subject, int(sizes[big]), lab), fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, '%s（epoch %s）。網格間距 %d mm。Folding：網格局部翻轉，紅點附近之網格線交叉或重疊。'
             % (exp, ep, step), ha='center', fontsize=11.5, color='#5F6A6B')
    out = os.path.join(args.out, 'folding_zoom_%s.png' % subject)
    fig.savefig(out, dpi=115, facecolor='#FAFAF8', bbox_inches='tight')
    plt.close(fig)
    print('->', out, '｜最大一團 %d 點，中心 %s，%s' % (int(sizes[big]), center.tolist(), lab))


VIEW_LAB = {'mix_exp4': 'Displacement・λ = 2', 'mix_exp3': 'Displacement・λ = 1', 'mix_wide': 'Displacement・λ = 1・2× width',
            'mix_exp5': 'SVF・λ = 2', 'mix_exp6': 'SVF・λ = 1', 'mix_exp7': 'SVF・λ = 0.5',
            'mix_wide_vel': 'SVF・λ = 1・2× width'}


def views(min_n=3):
    """2026-10-04 使用者要的：換不同方向切、比不同設定。
    folding_views.png：一顆（--view-one），軸狀／冠狀／矢狀各 4 刀
    folding_params.png：不同設定（--view-models）× 三個方向各 1 刀
    同 folding_where.png：只標「至少 min_n 位在同一點擠爆」的地方。每個方向用整顆腦的範圍裁切，同一列比例一樣。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from orient import canonical_axes, to_ras
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
    plt.rcParams['axes.unicode_minus'] = False
    INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'

    vol, seg, brain, region, depth = atlas_maps()
    perm, flip = canonical_axes(seg)
    vol_r, brain_r = to_ras(vol, perm, flip), to_ras(brain.astype(np.uint8), perm, flip)
    idx = np.argwhere(brain_r > 0)
    lo, hi = idx.min(axis=0), idx.max(axis=0)
    at = lambda ax_id, f: int(round(lo[ax_id] + f * (hi[ax_id] - lo[ax_id])))
    take = lambda a, ax_id, i: [a[i], a[:, i], a[:, :, i]][ax_id]
    plane = {0: (1, 2), 1: (0, 2), 2: (0, 1)}           # 切第 ax_id 軸時，畫面橫軸、縱軸是哪兩軸
    heat = lambda e: to_ras(np.load(os.path.join(args.out, 'heat_%s.npz' % e))['heat'], perm, flip)
    with open(os.path.join(args.out, 'folding_by_subject.csv'), encoding='utf-8') as fh:
        sub = list(csv.DictReader(fh))

    def show(ax, h, ax_id, i, vmax):
        ax.imshow(take(vol_r, ax_id, i).T, cmap='gray', origin='lower', vmin=0, vmax=1)
        hm = take(h, ax_id, i).astype(float).T
        ax.imshow(np.ma.masked_less(hm, min_n), cmap='autumn_r', origin='lower', vmin=min_n, vmax=vmax,
                  interpolation='nearest')
        p, q = plane[ax_id]
        ax.set_xlim(lo[p] - 4, hi[p] + 4)
        ax.set_ylim(lo[q] - 4, hi[q] + 4)
        ax.axis('off')

    def cbar(fig, axes, vmax):
        sm = plt.cm.ScalarMappable(cmap='autumn_r', norm=plt.Normalize(min_n, vmax))
        cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
        cb.set_label('出現 folding 之受試者數（≥ %d 位才顯示）' % min_n, fontsize=17)
        cb.ax.tick_params(labelsize=15)

    # ── 一顆，三個方向各 4 刀 ──
    h1 = heat(args.view_one)
    vmax = min(15, max(int(h1.max()), min_n + 1))
    VIEWS = [('軸狀面', 2, [(0.30, '下部'), (0.50, '側腦室層'), (0.68, '上部'), (0.85, '顱頂')]),
             ('冠狀面', 1, [(0.25, '枕部'), (0.45, '後部'), (0.62, '前部'), (0.80, '額部')]),
             ('矢狀面', 0, [(0.18, '左外側'), (0.38, '左內側'), (0.62, '右內側'), (0.82, '右外側')])]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.5), facecolor=PAPER)
    for r, (dname, ax_id, cuts) in enumerate(VIEWS):
        for c, (f, nm) in enumerate(cuts):
            show(axes[r, c], h1, ax_id, at(ax_id, f), vmax)
            axes[r, c].set_title('%s・%s' % (dname, nm), fontsize=21, fontweight='bold')
    cbar(fig, axes, vmax)
    p = os.path.join(args.out, 'folding_views.png')
    fig.savefig(p, dpi=110, facecolor=PAPER, bbox_inches='tight')
    plt.close(fig)
    print('->', p, '（%s）' % args.view_one)

    # ── 不同設定 × 三個方向 ──
    models = list(args.view_models)
    H = {e: heat(e) for e in models}
    vmax = min(15, max(max(int(h.max()) for h in H.values()), min_n + 1))
    CUTS = [('軸狀面・上部', 2, 0.68), ('冠狀面・後部', 1, 0.45), ('矢狀面・左外側', 0, 0.18)]
    fig, axes = plt.subplots(len(CUTS), len(models), figsize=(3.4 * len(models) + 1.2, 10.8), facecolor=PAPER)
    for c, e in enumerate(models):
        rr = [r for r in sub if r['exp'] == e]
        j = float(np.mean([float(r['jneg_pct']) for r in rr]))
        ns = [int(r['n_folded']) for r in rr]
        jt = '0%' if j == 0 else ('< 0.001%' if j < 0.001 else '%.3f%%' % j)
        if np.mean(ns) >= 1:
            nt = '%s voxels／位' % format(int(round(np.mean(ns))), ',')
        else:                                    # 速度場權重 1：51 位裡只有 3 位有，「每人約 0 點」會被看成完全沒有
            nt = '%d 位；最多 %d voxels' % (sum(x > 0 for x in ns), max(ns))
        # 模型名稱拆兩行（「Displacement・λ = 1・2× width」一行會壓到隔壁欄）
        axes[0, c].set_title('%s\nfolding %s\n%s' % (VIEW_LAB.get(e, e).replace('・', '\n', 1), jt, nt),
                             fontsize=19, fontweight='bold', color=INK)
        for r, (nm, ax_id, f) in enumerate(CUTS):
            show(axes[r, c], H[e], ax_id, at(ax_id, f), vmax)
            if c == 0:
                axes[r, 0].text(-0.05, 0.5, nm, transform=axes[r, 0].transAxes, rotation=90, ha='right', va='center',
                                fontsize=19, fontweight='bold', color=MUTED)
        print('   %-9s 同一點 %d 位以上的體素：%d' % (e, min_n, int((H[e][brain_r > 0] >= min_n).sum())))
    cbar(fig, axes, vmax)
    p = os.path.join(args.out, 'folding_params.png')
    fig.savefig(p, dpi=110, facecolor=PAPER, bbox_inches='tight')
    plt.close(fig)
    print('->', p)


def zoom_pair(subject='T054', specs=('mix_wide:0225', 'mix_wide_vel:0240'), half=16, step=2):
    """兩顆模型放大在「同一個位置」比：位置取第一顆最大的一團擠爆點（2026-10-06 使用者要 ⑤ 的視覺化比較）。
    一顆一列、三個方向各一格；黃線＝格子、紅點＝擠爆的點（畫法同 zoom()）。-> folding_zoom_pair_<subject>.png"""
    os.environ['NEURITE_BACKEND'] = 'pytorch'
    os.environ['VXM_BACKEND'] = 'pytorch'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    import voxelmorph as vxm
    from arch import load_model           # 新架構（串接等）也讀得到；舊模型照舊是 VxmDense.load
    from scipy import ndimage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from orient import canonical_axes, to_ras, flow_to_ras
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
    plt.rcParams['axes.unicode_minus'] = False

    device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
    atlas = np.load(args.atlas)['vol'].astype(np.float32)
    seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
    vol = np.load(os.path.join(args.test_dir, subject + '.npz'))['vol'].astype(np.float32)
    perm, flip = canonical_axes(seg)
    src = to_ras(vol, perm, flip)
    res = []
    for spec in specs:
        exp, ep = spec.split(':')
        model = load_model(os.path.join(ROOT, 'models', exp, ep + '.pt'), device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            _, flow = model(torch.from_numpy(vol)[None, None].to(device),
                            torch.from_numpy(atlas)[None, None].to(device), registration=True)
            det = jac_det(flow[0]).cpu().numpy()
        res.append((exp, to_ras(det, perm, flip) <= 0, flow_to_ras(flow[0].cpu().numpy(), perm, flip)))
        del model, flow
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    cc, n = ndimage.label(res[0][1], structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    big = int(sizes.argmax())
    center = np.round(ndimage.center_of_mass(cc == big)).astype(int)
    seg_r = to_ras(seg, perm, flip)
    lab = NAME.get(int(seg_r[tuple(center)]), str(int(seg_r[tuple(center)])))

    take = lambda a, ax_id, i: [a[i], a[:, i], a[:, :, i]][ax_id]
    planes = [('軸狀面', 2, (0, 1)), ('冠狀面', 1, (0, 2)), ('矢狀面', 0, (1, 2))]
    fig, axes = plt.subplots(len(res), 3, figsize=(11, 3.9 * len(res)), facecolor='#FAFAF8')   # 簡報上約 8 吋寬，字要夠大
    for r, (exp, fold, u) in enumerate(res):
        for c, (name, ax_id, (p, q)) in enumerate(planes):
            ax = axes[r, c]
            i = center[ax_id]
            img, fm = take(src, ax_id, i), take(fold, ax_id, i)
            up, uq = take(u[p], ax_id, i), take(u[q], ax_id, i)
            a0, a1 = center[p] - half, center[p] + half
            b0, b1 = center[q] - half, center[q] + half
            ax.imshow(img.T, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='nearest')
            for j in range(b0, b1 + 1, step):
                xs = np.arange(a0, a1 + 1)
                ax.plot(xs + up[a0:a1 + 1, j], j + uq[a0:a1 + 1, j], color='#FFD23F', lw=1.2)
            for k in range(a0, a1 + 1, step):
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
            ax.set_title('%s：%d 個 folding voxels' % (name, int(fm[a0:a1 + 1, b0:b1 + 1].sum())),
                         fontsize=15, fontweight='bold')
        axes[r, 0].text(-0.06, 0.5, VIEW_LAB.get(exp, exp), transform=axes[r, 0].transAxes, rotation=90,
                        ha='right', va='center', fontsize=17, fontweight='bold', color='#141A1D')
        print('   %-13s 整顆腦擠爆 %d 點；放大這塊（%d³）裡 %d 點'
              % (exp, int(fold.sum()), 2 * half + 1,
                 int(fold[tuple(slice(cc_ - half, cc_ + half + 1) for cc_ in center)].sum())))
    fig.tight_layout(rect=[0.02, 0, 1, 1])
    out = os.path.join(args.out, 'folding_zoom_pair_%s.png' % subject)
    fig.savefig(out, dpi=115, facecolor='#FAFAF8', bbox_inches='tight')
    plt.close(fig)
    print('->', out, '｜位置：%s 最大一團 %d 點，中心 %s，%s' % (res[0][0], int(sizes[big]), center.tolist(), lab))

    # ── 大圖：同樣三個切面，但整片腦都畫出來，藍框＝上面放大的那一塊（2026-10-06 使用者：「可以來大圖的嗎」）──
    from matplotlib.patches import Rectangle
    gstep = 4                                                    # 整片腦格子畫疏一點（每格 4 mm），不然糊成一片
    fig, axes = plt.subplots(len(res), 3, figsize=(13, 4.6 * len(res)), facecolor='#FAFAF8')
    for r, (exp, fold, u) in enumerate(res):
        for c, (name, ax_id, (p, q)) in enumerate(planes):
            ax = axes[r, c]
            i = center[ax_id]
            img, fm = take(src, ax_id, i), take(fold, ax_id, i)
            up, uq = take(u[p], ax_id, i), take(u[q], ax_id, i)
            inb = img > 0.02                                     # 只框腦的範圍，四周黑底裁掉
            pr, qr = np.nonzero(inb.any(axis=1))[0], np.nonzero(inb.any(axis=0))[0]
            p0, p1, q0, q1 = pr.min(), pr.max(), qr.min(), qr.max()
            ax.imshow(img.T, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='nearest')
            for j in range(q0, q1 + 1, gstep):
                xs = np.arange(p0, p1 + 1)
                ax.plot(xs + up[p0:p1 + 1, j], j + uq[p0:p1 + 1, j], color='#FFD23F', lw=0.7)
            for k in range(p0, p1 + 1, gstep):
                ys = np.arange(q0, q1 + 1)
                ax.plot(k + up[k, q0:q1 + 1], ys + uq[k, q0:q1 + 1], color='#FFD23F', lw=0.7)
            pp, qq = np.nonzero(fm)
            ax.scatter(pp + up[pp, qq], qq + uq[pp, qq], s=7, color='#E0262D', zorder=5, edgecolors='none')
            a0, a1 = center[p] - half - 3, center[p] + half + 3      # 跟放大圖的範圍一樣
            b0, b1 = center[q] - half - 3, center[q] + half + 3
            ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False, ec='#2B9BE0', lw=2.4, zorder=6))
            ax.set_xlim(p0 - 3, p1 + 3)
            ax.set_ylim(q0 - 3, q1 + 3)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('%s：%s 個 folding voxels' % (name, format(int(fm.sum()), ',')), fontsize=15, fontweight='bold')
        axes[r, 0].text(-0.06, 0.5, VIEW_LAB.get(exp, exp), transform=axes[r, 0].transAxes, rotation=90,
                        ha='right', va='center', fontsize=17, fontweight='bold', color='#141A1D')
    fig.tight_layout(rect=[0.02, 0, 1, 1])
    out = os.path.join(args.out, 'folding_full_pair_%s.png' % subject)
    fig.savefig(out, dpi=130, facecolor='#FAFAF8', bbox_inches='tight')
    plt.close(fig)
    print('->', out, '（大圖：整片腦，藍框＝放大的那一塊；格子每 %d mm）' % gstep)


if __name__ == '__main__':
    if args.zoom_pair:
        zoom_pair()
        sys.exit()
    if not args.plot_only:
        run()
    if args.views:
        views()
    else:
        plot()
        zoom()
