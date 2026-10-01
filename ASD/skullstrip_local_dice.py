# -*- coding: utf-8 -*-
"""只在「沒切乾淨附近的區域」算 Dice。老師 2026-09-30 紅字：「Dice 只算沒切乾淨附近的區域就好」。

§21 比的是全腦 Dice（30 個結構平均），殘留只黏在腦的邊邊，影響可能被整顆腦稀釋掉。
這裡改成只看殘留附近：在 atlas 上框出三塊區域，每一塊只在框內算 Dice。

    頭頂  腦的最上面 25mm（對應 top_vertex_mm，皮質上方黏著的硬腦膜／骨髓）
    顱底  腦的最下面 25mm（對應 base_blob10，小腦下方留的一整塊）
    後腦杓 腦的最後面 25mm（對應 back_occ_mm，老師說後腦杓也有）

每一塊區域：同一個框套在每位受試者身上（公平），比「該指標殘留最多的 10 位」vs「最乾淨的 10 位」，
看起點（只做線性對位）、配準後、模型貢獻。

框內 Dice＝2|A∩B∩框| / (|A∩框| + |B∩框|)，A＝搬過去的受試者標籤、B＝atlas 標籤；
只算在 atlas 框內至少有 100 個體素的結構，再平均。另外也算「整顆腦當一個結構」的框內 Dice。

⚠️ 用 atlas 空間的框：配準後受試者已經對到 atlas，框的位置就是殘留附近。
   起點（只做線性對位）也用同一個框，受試者只做了線性對位、也已經在 atlas 空間。

輸出（models/skullstrip_check/）：
    local_dice_<實驗>.csv     每位受試者 × 每塊區域的框內 Dice（起點 / 配準後）
    local_dice_summary.csv    每塊區域：髒 10 位 vs 乾淨 10 位
    local_dice.png

用法：
    python ASD\\skullstrip_local_dice.py --csv models\\skullstrip_check\\skullstrip_all520.csv --gpu 0
"""
import os
import sys
import csv
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default=os.path.join(ROOT, 'models', 'skullstrip_check', 'skullstrip_all520.csv'))
ap.add_argument('--models', nargs='+', default=['mix_exp3:0240'], help='實驗:epoch')
ap.add_argument('--test-dir', default=os.path.join(ROOT, 'data', 'mixed_preprocessed_v2', 'test'))
ap.add_argument('--atlas', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3.npz'))
ap.add_argument('--atlas-seg', default=os.path.join(ROOT, 'IXI', 'atlas_mni152_09c_v3_seg.npz'))
ap.add_argument('--labels', default=os.path.join(ROOT, 'voxelmorph-code', 'data', 'labels.npz'))
ap.add_argument('--band', type=int, default=25, help='每塊區域的厚度（mm）')
ap.add_argument('--n', type=int, default=10, help='髒／乾淨各取幾位')
ap.add_argument('--exclude', nargs='*', default=['A0131'], help='A0131 起點 0.563，離其他人一大截（同 §21）')
ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'skullstrip_check'))
ap.add_argument('--gpu', default='0')
ap.add_argument('--plot-only', action='store_true')
args = ap.parse_args()

REGIONS = [('top', '頭頂', 'top_vertex_mm'), ('base', '顱底', 'base_blob10'), ('back', '後腦杓', 'back_occ_mm')]


def boxes(seg):
    """atlas 上的三塊框。陣列是 RAS：第 1 軸越大越前面、第 2 軸越大越上面。"""
    brain = seg > 0
    idx = np.argwhere(brain)
    z = np.arange(seg.shape[2])[None, None, :]
    y = np.arange(seg.shape[1])[None, :, None]
    return {'top': brain & (z >= idx[:, 2].max() - args.band),
            'base': brain & (z <= idx[:, 2].min() + args.band),
            'back': brain & (y <= idx[:, 1].min() + args.band)}


def local(a, b, box, labels):
    """框內 Dice：30 個結構的平均（框內至少 100 體素的才算）＋ 整顆腦當一個結構。"""
    aa, bb = a[box], b[box]
    vals = []
    for l in labels:
        x, y_ = aa == l, bb == l
        if y_.sum() < 100:
            continue
        vals.append(2.0 * (x & y_).sum() / max(x.sum() + y_.sum(), 1))
    xb, yb = aa > 0, bb > 0
    whole = 2.0 * (xb & yb).sum() / max(xb.sum() + yb.sum(), 1)
    return float(np.mean(vals)), whole, len(vals)


def run():
    os.environ['NEURITE_BACKEND'] = 'pytorch'
    os.environ['VXM_BACKEND'] = 'pytorch'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    import voxelmorph as vxm

    device = torch.device('cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
    atlas = np.load(args.atlas)['vol'].astype(np.float32)
    aseg = np.load(args.atlas_seg)['seg'].astype(np.int32)
    labels = np.load(args.labels)['labels'].astype(int).tolist()
    B = boxes(aseg)
    for k, m in B.items():
        print('框 %-5s %7d 體素' % (k, int(m.sum())))
    a_t = torch.from_numpy(atlas)[None, None].to(device)
    warp_nn = vxm.torch.layers.SpatialTransformer(atlas.shape, mode='nearest').to(device)
    files = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))

    for spec in args.models:
        exp, ep = spec.split(':')
        model = vxm.networks.VxmDense.load(os.path.join(ROOT, 'models', exp, ep + '.pt'), device)
        model.to(device)
        model.eval()
        rows = []
        for f in files:
            sid = os.path.basename(f)[:-4]
            d = np.load(f)
            seg = d['seg'].astype(np.int32)
            with torch.no_grad():
                v = torch.from_numpy(d['vol'].astype(np.float32))[None, None].to(device)
                _, flow = model(v, a_t, registration=True)
                s = torch.from_numpy(seg.astype(np.float32))[None, None].to(device)
                segw = np.round(warp_nn(s, flow)[0, 0].cpu().numpy()).astype(np.int32)
            row = {'subject': sid}
            for k, _, _ in REGIONS:
                b0, w0, n = local(seg, aseg, B[k], labels)
                b1, w1, _ = local(segw, aseg, B[k], labels)
                row.update({k + '_before': b0, k + '_after': b1, k + '_brain_before': w0,
                            k + '_brain_after': w1, k + '_nlabels': n})
            rows.append(row)
            print('  %-9s %-10s 頭頂 %.3f→%.3f  顱底 %.3f→%.3f  後腦杓 %.3f→%.3f'
                  % (exp, sid, row['top_before'], row['top_after'], row['base_before'],
                     row['base_after'], row['back_before'], row['back_after']))
        p = os.path.join(args.out, 'local_dice_%s.csv' % exp)
        with open(p, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('->', p)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    RUST, TEAL, MUTED, PAPER = '#A34F1B', '#0E7C7B', '#5F6A6B', '#FAFAF8'

    with open(args.csv, encoding='utf-8') as fh:
        meta = {r['subject']: r for r in csv.DictReader(fh) if r['split'] == 'test'}
    out_rows = []
    exps = [s.split(':')[0] for s in args.models]
    fig, axes = plt.subplots(len(exps), 3, figsize=(15, 4.6 * len(exps)), facecolor=PAPER, squeeze=False)
    for r_, exp in enumerate(exps):
        with open(os.path.join(args.out, 'local_dice_%s.csv' % exp), encoding='utf-8') as fh:
            L = {r['subject']: r for r in csv.DictReader(fh)}
        keep = [s for s in L if s in meta and s not in set(args.exclude)]
        for c, (k, name, metric) in enumerate(REGIONS):
            order = sorted(keep, key=lambda s: -float(meta[s][metric]))
            groups = {'dirty': order[:args.n], 'clean': order[-args.n:]}
            x = np.array([float(meta[s][metric]) for s in keep])
            bef = np.array([float(L[s][k + '_before']) for s in keep])
            aft = np.array([float(L[s][k + '_after']) for s in keep])
            res = {'exp': exp, 'region': name, 'metric': metric, 'n_all': len(keep),
                   'r_metric_after': float(np.corrcoef(x, aft)[0, 1]),
                   'r_metric_gain': float(np.corrcoef(x, aft - bef)[0, 1])}
            for g, ss in groups.items():
                gb = np.mean([float(L[s][k + '_before']) for s in ss])
                ga = np.mean([float(L[s][k + '_after']) for s in ss])
                res.update({g + '_metric': float(np.mean([float(meta[s][metric]) for s in ss])),
                            g + '_before': gb, g + '_after': ga, g + '_gain': ga - gb})
            out_rows.append(res)

            ax = axes[r_][c]
            xs = np.arange(3)
            vals_d = [res['dirty_before'], res['dirty_after'], res['dirty_gain']]
            vals_c = [res['clean_before'], res['clean_after'], res['clean_gain']]
            ax.bar(xs - 0.2, vals_d, width=0.38, color=RUST, label='殘留最多 %d 位' % args.n)
            ax.bar(xs + 0.2, vals_c, width=0.38, color=TEAL, label='最乾淨 %d 位' % args.n)
            for xx, v in zip(xs - 0.2, vals_d):
                ax.text(xx, v + 0.005, '%.3f' % v, ha='center', fontsize=11, fontweight='bold', color=RUST)
            for xx, v in zip(xs + 0.2, vals_c):
                ax.text(xx, v + 0.005, '%.3f' % v, ha='center', fontsize=11, fontweight='bold', color=TEAL)
            ax.set_xticks(xs)
            ax.set_xticklabels(['起點（只做線性對位）', '配準後', '模型貢獻'], fontsize=11)
            ax.set_ylim(0, 1.0)
            ax.set_title('%s（只算腦的最%s 25mm）' % (name, {'top': '上面', 'base': '下面', 'back': '後面'}[k]),
                         fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=.3)
            ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if c == 0:
                ax.legend(fontsize=11, frameon=False, loc='upper left')
                ax.set_ylabel('框內 Dice（%s）' % exp, fontsize=11)
    fig.suptitle('只在殘留附近算 Dice：殘留最多 vs 最乾淨（test，排除 A0131）', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'local_dice.png'), dpi=115, facecolor=PAPER)
    plt.close(fig)
    p = os.path.join(args.out, 'local_dice_summary.csv')
    with open(p, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    for r in out_rows:
        print('%-9s %-4s 髒 %.3f→%.3f（%+.3f）  乾淨 %.3f→%.3f（%+.3f）  r(殘留,配準後)=%+.2f  r(殘留,貢獻)=%+.2f'
              % (r['exp'], r['region'], r['dirty_before'], r['dirty_after'], r['dirty_gain'],
                 r['clean_before'], r['clean_after'], r['clean_gain'], r['r_metric_after'], r['r_metric_gain']))
    print('->', p)
    print('->', os.path.join(args.out, 'local_dice.png'))


if __name__ == '__main__':
    if not args.plot_only:
        run()
    summarize()
