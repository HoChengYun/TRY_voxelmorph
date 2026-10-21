# -*- coding: utf-8 -*-
"""腦外背景的 folding 是從哪來的（2026-10-18；手冊 §28.1 寫「原因未查」的那件事）。

現象（test 51 位，每位平均）：mix_pyramid 腦外背景約 10,300 個 |J| ≤ 0、mix_pyramid_500 約 12,700、
mix_cascade_pyramid 約 49,800；mix_exp6、mix_cascade 幾乎是 0。腦內（atlas 非背景）都只有個位數到幾十個。

猜測：接位移時取到影像外面。SpatialTransformer 用 grid_sample，padding_mode 是預設的 'zeros'：
    由粗到細每一層  U_k(x) = u_k(x) + U↑(x + u_k(x))     （arch.py VxmPyramid.fields）
    串兩顆          U(x)   = u(x)   + U_1(x + u(x))       （arch.py VxmCascade.forward）
    速度場積分每一步 v(x)  = v(x)   + v(x + v(x))         （VecInt，scaling and squaring）
x + u(x) 落在影像外面時，後面那一項取到 0（或在邊界上跟 0 內插），總位移在邊界附近突然變小 → 擠爆。
VxmDense 也有後兩條，但它在影像邊界的位移很小；由粗到細最粗那層（1/16，每格 16 mm）出的是大範圍、平滑的位移，
一路延伸到影像邊界，放大後邊界上還有好幾 mm 的位移，才會撞到這件事。

這支查三件事（--subjects 指定的幾位）：
  1. 位置：背景 folding 離影像邊界幾格、離腦幾 mm、x + U(x) 是不是在影像外面、影像邊界上的位移有多大
  2. 出在哪一步：由粗到細每一層的 u_k（這一層的小修正，積分完）、U↑（上一層放大）、U_k（接起來）各有多少 folding
  3. 驗證：推論時把 grid_sample 改成 padding_mode='border'（取最靠近的邊界值，不補 0），不重新訓練，
     看背景 folding 會不會消失、Dice 變不變（--border-all：test 51 位都算）

輸出 models\\folding_check\\bg\\：bg_stats.csv、bg_levels.csv、bg_border_test.csv、bg_cache_<受試者>.npz（畫圖用）；
--plot 畫 bg_where_<受試者>.png（folding 投影）、bg_levels.png（每一層的 folding 數）、bg_border_<受試者>.png（zeros vs border）。

用法
  python ASD\\check_bg_folding.py --gpu 0                       # 5 位：位置、每一層、border 驗證
  python ASD\\check_bg_folding.py --gpu 0 --border-all          # 再加 test 51 位的 border 驗證
  python ASD\\check_bg_folding.py --plot
"""
import os
import sys
import csv
import argparse
import contextlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--models', nargs='+', default=['mix_exp6:0190', 'mix_cascade:0170', 'mix_pyramid:0245',
                                                'mix_pyramid_500:0335', 'mix_cascade_pyramid:0140'])
ap.add_argument('--subjects', nargs='+', default=['T054', 'D031', 'T079', 'VNT045', 'sub-0043'])
ap.add_argument('--test-dir', default=os.path.join(ROOT, 'data', 'mixed_preprocessed_v2', 'test'))
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'folding_check', 'bg'))
ap.add_argument('--gpu', default='0')
ap.add_argument('--border-all', action='store_true', help='border 驗證改成 test 全部的人（只算 Dice 和 folding 數）')
ap.add_argument('--plot', action='store_true')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
LABELS = np.load(args.labels)['labels'].astype(int).tolist()
FG = atlas_vol > 0
SHAPE = atlas_vol.shape


def jac_det(f):
    """det(I + ∇u)，中央差分（同 test_dice.py）。f: (3, X, Y, Z)。"""
    d = [[np.gradient(f[c], axis=a) for a in range(3)] for c in range(3)]
    return ((1 + d[0][0]) * ((1 + d[1][1]) * (1 + d[2][2]) - d[1][2] * d[2][1])
            - d[0][1] * (d[1][0] * (1 + d[2][2]) - d[1][2] * d[2][0])
            + d[0][2] * (d[1][0] * d[2][1] - (1 + d[1][1]) * d[2][0]))


def border_dist(shape):
    """每個 voxel 離影像邊界幾格（最外面一圈 = 0）。"""
    idx = np.ogrid[tuple(slice(0, s) for s in shape)]
    d = np.minimum(idx[0], shape[0] - 1 - idx[0])
    for i, s in zip(idx[1:], shape[1:]):
        d = np.minimum(d, np.minimum(i, s - 1 - i))
    return d


def outside(f):
    """x + u(x) 落在影像外面（任一軸 < 0 或 > N-1）。"""
    shape = f.shape[1:]
    out = np.zeros(shape, bool)
    for c in range(3):
        p = np.arange(shape[c]).reshape([-1 if a == c else 1 for a in range(3)]) + f[c]
        out |= (p < 0) | (p > shape[c] - 1)
    return out


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']    # ≤、≥、− JhengHei 沒有，缺的字用 DejaVu Sans 補
    plt.rcParams['axes.unicode_minus'] = False
    brain = FG

    # atlas 是 RAS：軸 0 左→右、軸 1 後→前、軸 2 下→上
    FACES = [('左', 0, 0), ('右', 0, 1), ('後', 1, 0), ('前', 1, 1), ('下', 2, 0), ('上', 2, 1)]

    def by_face(fold):
        """每個 folding 點歸到最近的那一面（影像六個面），回傳各面的個數。"""
        pts = np.argwhere(fold)
        if len(pts) == 0:
            return np.zeros(6, int)
        d = np.stack([pts[:, ax] if hi == 0 else SHAPE[ax] - 1 - pts[:, ax] for _, ax, hi in FACES], 1)
        return np.bincount(d.argmin(1), minlength=6)

    def proj(a, fold, ax_, title=None, slab=8):
        """只投影正中間 2×slab+1 片（整顆投影會把垂直於視線的那一面整片疊進來，看起來像在腦裡）。
        folding 點放大 2 格才看得到（都貼在影像邊框上）。外框 = 影像的邊界。"""
        mid = SHAPE[ax_] // 2
        sl = [slice(None)] * 3
        sl[ax_] = slice(mid - slab, mid + slab + 1)
        bg = brain[tuple(sl)].sum(axis=ax_).astype(float)
        f2 = fold[tuple(sl)].sum(axis=ax_) > 0
        cnt = ndimage.maximum_filter(f2, size=3) if f2.any() else f2
        a.imshow(np.rot90(bg), cmap='gray_r', alpha=0.35, interpolation='nearest')
        a.imshow(np.rot90(np.ma.masked_where(~cnt, cnt.astype(float))),
                 cmap=matplotlib.colors.ListedColormap(['#e31a1c']), vmin=0, vmax=1, interpolation='nearest')
        a.contour(np.rot90(bg > 0), levels=[0.5], colors='k', linewidths=0.6)
        for sp in a.spines.values():
            sp.set_linewidth(1.5)
        a.set_xticks([]); a.set_yticks([])
        if title:
            a.set_title(title, fontsize=9.5)

    # 1. folding 在哪：每個模型一列；矢狀面、冠狀面投影（外框 = 影像邊界）＋ 各面的個數
    from scipy import ndimage
    for s in args.subjects:
        p = os.path.join(args.out, 'bg_cache_%s.npz' % s)
        if not os.path.exists(p):
            continue
        c = np.load(p)
        names = [str(n) for n in c['names'] if str(n) != 'mix_exp6']
        fig, axs = plt.subplots(len(names), 3, figsize=(11, 2.9 * len(names)),
                                gridspec_kw={'width_ratios': [1.15, 1, 1.1]})
        for i, n in enumerate(names):
            fold = c['fold_' + n]
            proj(axs[i, 0], fold, 0, '矢狀面：正中間 17 片（左邊 = 後、上 = 頭頂）' if i == 0 else None)
            proj(axs[i, 1], fold, 1, '冠狀面：正中間 17 片（上 = 頭頂）' if i == 0 else None)
            nf, nb = int(c['nfg_' + n]), int(c['nbg_' + n])
            axs[i, 0].set_ylabel('%s\n腦內 %d、背景 %s' % (n, nf, format(nb, ',')), fontsize=9)
            a = axs[i, 2]
            cnt = by_face(fold & ~brain)
            a.barh(np.arange(6), cnt, color='#de2d26')
            a.set_yticks(np.arange(6)); a.set_yticklabels(['%s面' % f for f, _, _ in FACES], fontsize=8.5)
            a.invert_yaxis()
            for k, v in enumerate(cnt):
                a.text(v, k, ' ' + format(int(v), ','), va='center', fontsize=8)
            a.set_xlim(0, max(1, cnt.max()) * 1.3)
            a.tick_params(axis='x', labelsize=7.5)
            if i == 0:
                a.set_title('背景 folding 貼在影像的哪一面', fontsize=9.5)
        fig.suptitle('%s：|J| ≤ 0 的位置（外框 = 影像邊界 192×224×192；黑線 = 腦的輪廓；紅 = folding，放大 2 格顯示）' % s,
                     fontsize=10.5)
        fig.tight_layout()
        out = os.path.join(args.out, 'bg_where_%s.png' % s)
        fig.savefig(out, dpi=130); plt.close(fig)
        print('  ->', out)

    # 1b. 原理（一維）：最後一格往外取，取到影像外面 → 補 0 → 總位移掉下來 → φ(x) = x + U(x) 往回走 = 擠爆
    N, Uval, uval = 192, 2.1, 0.6                      # 影像邊界上 U↑ 約 2.1 mm（實測全尺寸那一層 2.13）、這一層往外修 0.6
    x = np.arange(N - 8, N)
    def sample(U, p, mode):                            # grid_sample（align_corners=True）的一維版本
        q = np.clip(p, 0, N - 1) if mode == 'border' else p
        i0 = np.floor(q).astype(int)
        w = q - i0
        get = lambda i: np.where((i >= 0) & (i <= N - 1), U[np.clip(i, 0, N - 1)], 0.0)
        return (1 - w) * get(i0) + w * get(i0 + 1)
    Uup = np.full(N, Uval)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for mode, col, lab in [('zeros', '#de2d26', "補 0（padding_mode='zeros'，現在的設定）"),
                           ('border', '#3182bd', "取邊界值（padding_mode='border'）")]:
        U = uval + sample(Uup, x + uval, mode)
        phi = x + U
        J = np.gradient(phi)
        axs[0].plot(x, U, 'o-', color=col, label=lab)
        axs[1].plot(x, phi, 'o-', color=col, label=lab)
        for xi, pi, ji in zip(x, phi, J):
            if ji <= 0:
                axs[1].annotate('J = %.2f' % ji, (xi, pi), textcoords='offset points', xytext=(-48, -16), fontsize=8,
                                color=col)
    axs[0].axvline(N - 1, color='k', ls=':', lw=1); axs[1].axvline(N - 1, color='k', ls=':', lw=1)
    axs[0].set_title('總位移 U(x) = u(x) + U↑(x + u(x))', fontsize=10)
    axs[0].set_xlabel('x（最右邊 = 影像最後一格 %d）' % (N - 1)); axs[0].set_ylabel('位移（mm）')
    axs[1].set_title('搬到哪裡 φ(x) = x + U(x)：往回走就是擠爆（J ≤ 0）', fontsize=10)
    axs[1].set_xlabel('x'); axs[1].set_ylabel('φ(x)')
    axs[0].legend(fontsize=8)
    fig.suptitle('一維示意：U↑ = %.1f mm（邊界上實測約 2.1）、這一層再往外修 u = %.1f mm；最後一格要去 %.1f 取 U↑，已經在影像外面'
                 % (Uval, uval, N - 1 + uval), fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(args.out, 'bg_mechanism.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print('  ->', out)

    # 2. 每一層的 folding 數（第一位受試者的平均改成 5 位平均）
    p = os.path.join(args.out, 'bg_levels.csv')
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p, encoding='utf-8')))
        models = []
        for r in rows:
            if r['model'] not in models:
                models.append(r['model'])
        fig, axs = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), squeeze=False)
        for a, m in zip(axs[0], models):
            rs = [r for r in rows if r['model'] == m]
            steps = []
            for r in rs:
                if r['step'] not in steps:
                    steps.append(r['step'])
            def mean(col):
                return [np.mean([float(r[col]) for r in rs if r['step'] == st]) for st in steps]
            x = np.arange(len(steps))
            a.bar(x - 0.27, mean('fold_u'), 0.27, label='u_k（這一層的修正）', color='#9ecae1')
            a.bar(x, mean('fold_Uup'), 0.27, label='U↑（上一層放大）', color='#bdbdbd')
            a.bar(x + 0.27, mean('fold_U'), 0.27, label='U_k（接起來）', color='#de2d26')
            a.set_xticks(x); a.set_xticklabels(steps, rotation=30, fontsize=8)
            a.set_yscale('symlog', linthresh=10)
            a.set_title(m, fontsize=10); a.set_ylabel('|J| ≤ 0 的 voxel 數（該層格子上；5 位平均）', fontsize=8)
            a.legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(args.out, 'bg_levels.png')
        fig.savefig(out, dpi=130); plt.close(fig)
        print('  ->', out)

    # 3. zeros vs border：同一位、cascade_pyramid
    for s in args.subjects[:1]:
        p = os.path.join(args.out, 'bg_cache_%s.npz' % s)
        if not os.path.exists(p):
            continue
        c = np.load(p)
        names = [str(n) for n in c['names'] if 'border_' + str(n) in c.files]
        fig, axs = plt.subplots(len(names), 2, figsize=(7.5, 3.1 * len(names)), squeeze=False)
        for i, n in enumerate(names):
            for j, (key, lab) in enumerate([('fold_' + n, "padding_mode='zeros'（訓練與評估的設定）"),
                                            ('border_' + n, "padding_mode='border'（只改推論）")]):
                a = axs[i, j]
                fold = c[key]
                proj(a, fold, 0)
                nb = int((fold & ~brain).sum()); nf = int((fold & brain).sum())
                a.set_title('%s\n腦內 %d、背景 %s' % (lab, nf, format(nb, ',')), fontsize=8.5)
                if j == 0:
                    a.set_ylabel(n, fontsize=9)
        fig.suptitle('%s 矢狀面正中間 17 片（外框 = 影像邊界）：推論時改成取邊界值，背景 folding 會不會消失' % s, fontsize=10.5)
        fig.tight_layout()
        out = os.path.join(args.out, 'bg_border_%s.png' % s)
        fig.savefig(out, dpi=130); plt.close(fig)
        print('  ->', out)


if args.plot:
    plot()
    sys.exit()

# ── 推論 ────────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '-1' else ''
import torch                                                                  # noqa: E402
import torch.nn.functional as nnf                                             # noqa: E402
from arch import load_model, VxmPyramid, VxmCascade                           # noqa: E402
import voxelmorph as vxm                                                      # noqa: E402

device = 'cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu'
_grid_sample = nnf.grid_sample


@contextlib.contextmanager
def padding(mode):
    """暫時把所有 grid_sample 的 padding_mode 換掉（layers.py 用 nnf.grid_sample，呼叫時才查屬性，所以換得到）。"""
    def gs(*a, **k):
        k.setdefault('padding_mode', mode)
        return _grid_sample(*a, **k)
    torch.nn.functional.grid_sample = gs
    try:
        yield
    finally:
        torch.nn.functional.grid_sample = _grid_sample


atlas_t = torch.from_numpy(atlas_vol)[None, None].to(device)
warp_nn = vxm.torch.layers.SpatialTransformer(SHAPE, mode='nearest').to(device)
BD = border_dist(SHAPE)
from scipy import ndimage                                                     # noqa: E402
BRAIN_DIST = ndimage.distance_transform_edt(~FG)                              # 離腦（atlas 非背景）幾 mm


def pyramid_levels(m, source, target, rec, tag):
    """照 VxmPyramid.fields 一模一樣算一次，順便記下每一層的 u_k、U↑、U_k（numpy，在該層的格子上）。"""
    fm, ff = m.encode(source), m.encode(target)
    U, h = None, None
    for k in range(m.n, -1, -1):
        Uup = None
        if U is not None:
            U, h = m.up_flow(U), m.up_feat(h)
            Uup = U
        m_k = fm[k] if U is None else m.warp[k](fm[k], U)
        x = torch.cat([m_k, ff[k]] + ([h] if h is not None else []), dim=1)
        for blk in m.blocks[k]:
            x = blk(x)
        h = x
        v = m.flows[k](h)
        u = m.integrate[k](v) if m.integrate is not None else v
        U = u if U is None else u + m.warp[k](U, u)
        rec.append(dict(step='%s1/%d' % (tag, 2 ** k) if k else '%s全尺寸' % tag,
                        u=u[0].cpu().numpy(), Uup=None if Uup is None else Uup[0].cpu().numpy(), U=U[0].cpu().numpy()))
    return U


def run_levels(model, src):
    """回傳（全尺寸總位移, 每一步的紀錄）。由粗到細記每一層；串接另外記「兩顆接起來」那一步。"""
    rec = []
    if isinstance(model, VxmPyramid):
        U = pyramid_levels(model, src, atlas_t, rec, '')
    elif isinstance(model, VxmCascade):
        U = None
        for i, m in enumerate(model.stages):
            s = src if U is None else model.transformer(src, U)
            tag = '第%d顆 ' % (i + 1)
            if model.stage == 'pyramid':
                u = pyramid_levels(m, s, atlas_t, rec, tag)
            else:
                _, u = m(s, atlas_t, registration=True)
                rec.append(dict(step=tag.strip(), u=u[0].cpu().numpy(), Uup=None, U=u[0].cpu().numpy()))
            Uprev = U
            U = u if U is None else u + model.transformer(U, u)
            if Uprev is not None:
                rec.append(dict(step='兩顆接起來', u=u[0].cpu().numpy(), Uup=Uprev[0].cpu().numpy(), U=U[0].cpu().numpy()))
    else:
        _, U = model(src, atlas_t, registration=True)
    return U, rec


def dice_mean(seg_w):
    out = []
    for lab in LABELS:
        x, y = seg_w == lab, atlas_seg == lab
        s = x.sum() + y.sum()
        out.append(np.nan if s == 0 else 2.0 * (x & y).sum() / s)
    return float(np.nanmean(out))


def evaluate(model, vol, seg):
    v = torch.from_numpy(vol)[None, None].to(device)
    _, U = model(v, atlas_t, registration=True)
    sw = warp_nn(torch.from_numpy(seg.astype(np.float32))[None, None].to(device), U)
    f = U[0].cpu().numpy()
    det = jac_det(f)
    return f, det, dice_mean(np.round(sw[0, 0].cpu().numpy()).astype(np.int32))


models = {}
for spec in args.models:
    exp, ep = spec.split(':')
    models[exp] = load_model(os.path.join(ROOT, 'models', exp, '%s.pt' % ep), device).to(device).eval()

stats, levels, border = [], [], []
with torch.no_grad():
    for s in args.subjects:
        d = np.load(os.path.join(args.test_dir, s + '.npz'))
        vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
        cache = {'names': np.array(list(models))}
        for name, model in models.items():
            f, det, dz = evaluate(model, vol, seg)
            fold = det <= 0
            bgf = fold & ~FG
            oob = outside(f)
            r = {'subject': s, 'model': name, 'dice': dz, 'fold_fg': int((fold & FG).sum()), 'fold_bg': int(bgf.sum())}
            for lo, hi in [(0, 0), (1, 1), (2, 4), (5, 9), (10, 999)]:
                r['bg_border_%d_%d' % (lo, hi)] = int((bgf & (BD >= lo) & (BD <= hi)).sum())
            r['bg_oob_pct'] = 100.0 * float((bgf & oob).sum()) / max(1, int(bgf.sum()))
            r['bg_brain_mm_median'] = float(np.median(BRAIN_DIST[bgf])) if bgf.any() else np.nan
            mag = np.sqrt((f ** 2).sum(0))
            r['disp_border_mean'] = float(mag[BD <= 2].mean())          # 影像最外面 3 圈的平均位移（voxel = mm）
            r['disp_brain_mean'] = float(mag[FG].mean())
            r['oob_border_pct'] = 100.0 * float(oob[BD <= 2].mean())    # 最外面 3 圈有多少比例取到影像外面
            stats.append(r)
            cache['fold_' + name] = fold
            cache['nfg_' + name] = r['fold_fg']
            cache['nbg_' + name] = r['fold_bg']
            print('  %-9s %-20s Dice %.4f  腦內 %4d  背景 %6d（離邊界 0 格 %d、1 格 %d、2～4 格 %d、≥5 格 %d）'
                  '  取到影像外面 %.0f%%  邊界位移 %.2f mm' % (
                      s, name, dz, r['fold_fg'], r['fold_bg'], r['bg_border_0_0'], r['bg_border_1_1'],
                      r['bg_border_2_4'], r['bg_border_5_9'] + r['bg_border_10_999'], r['bg_oob_pct'],
                      r['disp_border_mean']), flush=True)

            # 每一層
            if isinstance(model, (VxmPyramid, VxmCascade)):
                v = torch.from_numpy(vol)[None, None].to(device)
                U, rec = run_levels(model, v)
                assert np.allclose(U[0].cpu().numpy(), f, atol=1e-4), '逐層重算跟 model() 對不上'
                for x in rec:
                    bd = border_dist(x['U'].shape[1:])
                    fu = jac_det(x['U']) <= 0
                    levels.append({'subject': s, 'model': name, 'step': x['step'],
                                   'fold_u': int((jac_det(x['u']) <= 0).sum()),
                                   'fold_Uup': int((jac_det(x['Uup']) <= 0).sum()) if x['Uup'] is not None else 0,
                                   'fold_U': int(fu.sum()), 'fold_U_border01': int((fu & (bd <= 1)).sum()),
                                   'oob_at_fold_pct': 100.0 * float((fu & outside(x['u'])).sum()) / max(1, int(fu.sum())),
                                   'Uup_border_mag': float(np.sqrt((x['Uup'] ** 2).sum(0))[bd <= 1].mean())
                                   if x['Uup'] is not None else 0.0})

            # 驗證：只改推論的 padding
            if name != 'mix_exp6':
                with padding('border'):
                    fb, detb, db = evaluate(model, vol, seg)
                foldb = detb <= 0
                border.append({'subject': s, 'model': name, 'dice_zeros': dz, 'dice_border': db,
                               'fg_zeros': r['fold_fg'], 'fg_border': int((foldb & FG).sum()),
                               'bg_zeros': r['fold_bg'], 'bg_border': int((foldb & ~FG).sum()),
                               'max_disp_diff_fg': float(np.abs(fb - f)[:, FG].max())})
                cache['border_' + name] = foldb
                print('            border：Dice %.4f（%+.5f）  腦內 %d  背景 %d  腦內位移最大差 %.3f mm' % (
                    db, db - dz, border[-1]['fg_border'], border[-1]['bg_border'], border[-1]['max_disp_diff_fg']),
                    flush=True)
        np.savez_compressed(os.path.join(args.out, 'bg_cache_%s.npz' % s), **cache)

    if args.border_all:
        import glob
        files = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))
        for name, model in models.items():
            if name == 'mix_exp6':
                continue
            for p in files:
                s = os.path.basename(p)[:-4]
                if s in args.subjects:
                    continue
                d = np.load(p)
                vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
                f, det, dz = evaluate(model, vol, seg)
                with padding('border'):
                    fb, detb, db = evaluate(model, vol, seg)
                fold, foldb = det <= 0, detb <= 0
                border.append({'subject': s, 'model': name, 'dice_zeros': dz, 'dice_border': db,
                               'fg_zeros': int((fold & FG).sum()), 'fg_border': int((foldb & FG).sum()),
                               'bg_zeros': int((fold & ~FG).sum()), 'bg_border': int((foldb & ~FG).sum()),
                               'max_disp_diff_fg': float(np.abs(fb - f)[:, FG].max())})
            print('  border（%s，%d 位）完成' % (name, len([b for b in border if b['model'] == name])), flush=True)


def save(rows, fn):
    with open(os.path.join(args.out, fn), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ('%.6f' % v if isinstance(v, float) else v) for k, v in r.items()})
    print('  CSV ->', os.path.join(args.out, fn))


save(stats, 'bg_stats.csv')
if levels:
    save(levels, 'bg_levels.csv')
if border:
    save(border, 'bg_border_test.csv')
    print()
    for name in models:
        b = [x for x in border if x['model'] == name]
        if not b:
            continue
        dd = np.array([x['dice_border'] - x['dice_zeros'] for x in b])
        print('  %-20s %d 位：Dice %+.5f（最大 |差| %.5f）  背景 folding %.0f → %.0f  腦內 %.1f → %.1f' % (
            name, len(b), dd.mean(), np.abs(dd).max(), np.mean([x['bg_zeros'] for x in b]),
            np.mean([x['bg_border'] for x in b]), np.mean([x['fg_zeros'] for x in b]), np.mean([x['fg_border'] for x in b])))
