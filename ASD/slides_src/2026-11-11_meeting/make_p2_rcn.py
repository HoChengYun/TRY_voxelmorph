# -*- coding: utf-8 -*-
"""Part 2：RCN 小腦白質異常之分析（正式用語版；11-03 分析見手冊 §31.11）
→ models\\deck_charts\\1111_rcn_mech.png、1111_rcn_atlas.png、1111_rcn_slices.png

  mech  ：(a) 各級（affine stem、VTN 1～3）之小腦白質 Dice，兩種 atlas；(b) 受試者小腦白質相對亮度 vs 形變後體積
  atlas ：各結構之 Dice 與形變後體積，affine → RCN，兩種 atlas（MNI152、真人 sub-0036）
  slices：VNT045 之小腦（矢狀、冠狀），atlas｜Moving（affine）｜RCN 形變後，兩種 atlas
資料：models\\orig_rcn_vtn3\\vis\\（MNI152 那顆之每級、每位統計）、models\\orig_rcn_vtn3_subj\\vis\\（sub-0036 那顆；group_stats、intensity、per_structure_*）、
      models\\orig_dwp\\vis_0032\\per_structure_{affine,rcn}.csv（MNI152 之 affine 與 RCN 每結構）。切片第一次跑時從 11-02/03 之推論快取取出，
      存成 models\\deck_charts\\1111_rcn_slices_cache_VNT045.npz。
"""
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import J, OUT, INK, MUTED, GRAY, RULE, W, save, clean

C_MNI, C_SUB = '#2F7FD0', '#D08C2B'
M = J('models')
V0, V1 = J('models', 'orig_rcn_vtn3', 'vis'), J('models', 'orig_rcn_vtn3_subj', 'vis')
gs_ = pd.read_csv(os.path.join(V1, 'group_stats.csv'))
p0 = pd.read_csv(os.path.join(V0, 'per_stage_stats.csv'))


def cb_wm(path, stage=None):
    q = pd.read_csv(path)
    if stage is not None:
        q = q[q.stage == stage].rename(columns={'vol_ratio_or_sim_r': 'vol_ratio'})
    q = q[q.label.astype(str).isin(['7', '46'])].copy()
    q['dice'] = q.dice.astype(float)
    return q.groupby('subject').dice.mean()


aff0 = cb_wm(J('models', 'orig_dwp', 'vis_0032', 'per_structure_affine.csv')).mean()
aff1 = cb_wm(os.path.join(V1, 'per_structure_affine.csv')).mean()

# ════ mech ════
fig, ax = plt.subplots(1, 2, figsize=(W, 4.3), gridspec_kw=dict(width_ratios=[1, 1.1]))
fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.15, wspace=0.22)
a = ax[0]
x = np.arange(4)
st0 = [p0[p0.stage == k].dice_cb_wm.mean() for k in range(4)]
st1 = [gs_[(gs_.stage == 'stage%d' % k) & (gs_.group == 'cb_wm')].dice.mean() for k in range(4)]
for st, af, col, lab in ((st0, aff0, C_MNI, 'MNI152 atlas'), (st1, aff1, C_SUB, 'sub-0036 atlas')):
    a.plot(x, st, color=col, lw=2, marker='o', ms=7, mfc=col, mec='white', mew=1.2, zorder=3)
    a.axhline(af, color=col, lw=1, ls=(0, (4, 3)), zorder=2)
    for k in range(4):
        a.text(k, st[k] + 0.006, '%.3f' % st[k], ha='center', va='bottom', fontsize=8.5, color=INK)
    a.text(3.15, st[-1], lab, va='center', ha='left', fontsize=9, color=INK)
a.set_xticks(x, ['Affine\nstem', 'VTN 1', 'VTN 2', 'VTN 3'])
a.set_xlim(-0.3, 4.1)
a.set_ylim(0.62, 0.84)
a.set_ylabel('小腦白質 Dice（左右合併）', fontsize=10)
a.set_title('(a) 各級輸出之小腦白質 Dice（test 51 位平均；虛線 = affine）', fontsize=10.5, loc='left', color=INK)
clean(a)

a = ax[1]
it = pd.read_csv(os.path.join(V1, 'intensity.csv')).pivot(index='subject', columns='group', values='mean_intensity')
rel = it['cb_wm'] / it['wm']
subj = rel.drop(index=['sub-0036 (atlas)', 'MNI152 (atlas)'])
ps = pd.read_csv(os.path.join(V0, 'per_subject_stats.csv')).set_index('subject')
g3 = gs_[(gs_.stage == 'stage3') & (gs_.group == 'cb_wm')].set_index('subject')
for vol, col, lab, xa in ((ps.rcn_vol_cb_wm, C_MNI, 'MNI152', rel['MNI152 (atlas)']), (g3.vol_ratio, C_SUB, 'sub-0036', rel['sub-0036 (atlas)'])):
    y = vol.reindex(subj.index)
    rs = subj.corr(y, method='spearman')
    a.scatter(subj, y, s=22, color=col, edgecolor='white', linewidth=0.7, zorder=3,
              label='%s（中位數 %.2f；Spearman ρ = %s）' % (lab, y.median(), ('%.2f' % rs).replace('-', '−')))
    a.axvline(xa, color=col, lw=1.1, ls=(0, (4, 3)), zorder=2)
    a.text(xa + (0.004 if lab == 'sub-0036' else -0.004), 2.95, '%s atlas\n%.3f' % (lab, xa), ha='left' if lab == 'sub-0036' else 'right',
           va='top', fontsize=8.5, color=INK)
a.axhline(1, color=GRAY, lw=0.9)
a.set_xlim(0.69, 1.0)
a.set_ylim(0.8, 3.1)
a.set_xlabel('受試者之小腦白質相對亮度（小腦白質 ÷ 大腦白質）', fontsize=10)
a.set_ylabel('形變後小腦白質體積（atlas = 1）', fontsize=10)
a.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='none', fontsize=8.8, bbox_to_anchor=(0.94, 0.8))
a.set_title('(b) 相對亮度與形變後體積（每點一位；虛線 = atlas 之相對亮度）', fontsize=10.5, loc='left', color=INK)
clean(a, grid='both')
save(fig, '1111_rcn_mech.png')

# ════ atlas：各結構 ════
N = [('小腦白質', (7, 46)), ('小腦皮質', (8, 47)), ('大腦白質', (2, 41)), ('大腦皮質', (3, 42)), ('側腦室', (4, 43)), ('視丘', (10, 49)), ('尾狀核', (11, 50)),
     ('殼核', (12, 51)), ('蒼白球', (13, 52)), ('海馬迴', (17, 53)), ('杏仁核', (18, 54)), ('腹側間腦', (28, 60)), ('脈絡叢', (31, 63)), ('第三腦室', (14,)),
     ('第四腦室', (15,)), ('腦幹', (16,)), ('腦脊髓液', (24,))]


def long(p, stage=None):
    q = pd.read_csv(p)
    if stage is not None:
        q = q[q.stage == stage].rename(columns={'vol_ratio_or_sim_r': 'vol_ratio'})
    q = q[q.label.astype(str) != 'mean30'].copy()
    q['label'] = q.label.astype(int)
    q['dice'] = q.dice.astype(float)
    q['vol_ratio'] = q.vol_ratio.astype(float)
    return q


S = {'mni_aff': long(J('models', 'orig_dwp', 'vis_0032', 'per_structure_affine.csv')), 'mni_rcn': long(J('models', 'orig_dwp', 'vis_0032', 'per_structure_rcn.csv')),
     'sub_aff': long(os.path.join(V1, 'per_structure_affine.csv')), 'sub_rcn': long(os.path.join(V1, 'per_structure_stages.csv'), stage=3)}
rows = []
for n, ls in N:
    r = {'n': n}
    for k, df in S.items():
        g = df[df.label.isin(ls)].groupby('subject')
        r['D_' + k], r['V_' + k] = g.dice.mean().mean(), g.vol_ratio.mean().median()
    rows.append(r)
t = pd.DataFrame(rows).set_index('n')
t = t.loc[(t.D_mni_rcn - t.D_mni_aff).sort_values().index]
y = np.arange(len(t))
off = 0.17
fig, ax = plt.subplots(1, 2, figsize=(W, 5.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.09, wspace=0.05)
for a, key, xl, ttl in ((ax[0], 'D', (0.05, 0.97), 'Dice'), (ax[1], 'V', (0.55, 1.85), '形變後體積（atlas = 1，中位數）')):
    if key == 'V':
        a.axvline(1, color=MUTED, lw=1, zorder=1)
    for atl, col, dy in (('mni', C_MNI, -off), ('sub', C_SUB, off)):
        x0, x1 = t['%s_%s_aff' % (key, atl)].values, t['%s_%s_rcn' % (key, atl)].values
        for i in range(len(t)):
            a.annotate('', xy=(x1[i], y[i] + dy), xytext=(x0[i], y[i] + dy),
                       arrowprops=dict(arrowstyle='-|>', color=col, lw=1.2, alpha=0.55, shrinkA=4, shrinkB=5, mutation_scale=8), zorder=2)
        a.scatter(x0, y + dy, s=40, facecolor='white', edgecolor=col, linewidth=1.4, marker='s', zorder=3)
        a.scatter(x1, y + dy, s=48, color=col, edgecolor='white', linewidth=1, zorder=4)
    a.set_xlim(*xl)
    a.set_yticks(y)
    a.set_yticklabels(t.index if key == 'D' else [], fontsize=10, color=INK)
    a.set_ylim(len(t) - 0.4, -0.6)
    a.set_title(ttl, fontsize=10.5, loc='left', color=INK)
    clean(a, grid='x')
    a.spines['left'].set_visible(False)
    a.tick_params(axis='y', length=0)
i0 = list(t.index).index('小腦白質')
ax[1].text(t.V_mni_rcn['小腦白質'] + 0.025, i0 - off, '%.2f' % t.V_mni_rcn['小腦白質'], va='center', fontsize=9, color=INK)
ax[1].text(t.V_sub_rcn['小腦白質'] + 0.025, i0 + off, '%.2f' % t.V_sub_rcn['小腦白質'], va='center', fontsize=9, color=INK)
mk = lambda m, fc, ec, l: Line2D([], [], marker=m, color='none', mfc=fc, mec=ec, mew=1.4, ms=8, label=l)     # noqa: E731
fig.legend(handles=[mk('s', 'white', C_MNI, 'MNI152 atlas：affine'), mk('o', C_MNI, 'white', 'MNI152 atlas：RCN'),
                    mk('s', 'white', C_SUB, 'sub-0036 atlas：affine'), mk('o', C_SUB, 'white', 'sub-0036 atlas：RCN')],
           loc='upper center', ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.53, 1.0))
save(fig, '1111_rcn_atlas.png')

# ════ slices（VNT045）════
SUBJ = 'VNT045'
OLD = r'C:\Users\h4524\AppData\Local\Temp\claude\C--Users-h4524-claude-cheng\b94b7d4d-abf6-4d59-84e4-c9799aade1ce\scratchpad'
mni = np.load(J('IXI', 'atlas_mni152_09c_v3.npz'))['vol'].astype(np.float32)
mseg = np.load(J('IXI', 'atlas_mni152_09c_v3_seg.npz'))['seg'].astype(np.int32)
a36 = np.load(J('data', 'mixed_preprocessed_v2', 'val', 'sub-0036.npz'))
sub, sseg = a36['vol'].astype(np.float32), a36['seg'].astype(np.int32)
d = np.load(J('data', 'mixed_preprocessed_v2', 'test', SUBJ + '.npz'))
vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
take = lambda v, p, i: {'sag': v[i, :, :], 'cor': v[:, i, :]}[p]     # noqa: E731


def cuts(S_):
    wm = np.isin(S_, (7, 46))
    return int(np.argmax(np.isin(S_, (7,)).sum(axis=(1, 2)))), int(np.argmax(wm.sum(axis=(0, 2))))


cpath = os.path.join(OUT, '1111_rcn_slices_cache_%s.npz' % SUBJ)
if not os.path.exists(cpath):
    Cc = {}
    for tag, f, S_ in (('mni', 'cache_%s_rcn.npz', mseg), ('sub', 'cache_%s_rcnsub.npz', sseg)):
        q = np.load(os.path.join(OLD, f % SUBJ))
        sg, cr = cuts(S_)
        for p, i in (('sag', sg), ('cor', cr)):
            Cc['%s_%s_img' % (tag, p)] = take(q['moved'], p, i).astype(np.float32)
            Cc['%s_%s_seg' % (tag, p)] = take(q['seg'], p, i).astype(np.int16)
    np.savez_compressed(cpath, **Cc)
Cc = dict(np.load(cpath))
L_WM, L_CTX, L_BS = '#eda100', '#e87ba4', '#008300'


def overlay(a, sl):
    rgba = np.zeros(sl.shape + (4,))
    for labs, col in (((7, 46), L_WM), ((8, 47), L_CTX), ((16,), L_BS)):
        rgba[np.isin(sl, labs)] = list(matplotlib.colors.to_rgba(col, 0.62))
    a.imshow(np.transpose(rgba, (1, 0, 2)), origin='lower', interpolation='nearest')


fig = plt.figure(figsize=(W, 7.4))
G = GridSpec(4, 7, figure=fig, width_ratios=[1, 1, 1, 0.07, 1, 1, 1], left=0.045, right=0.997, top=0.88, bottom=0.06, wspace=0.03, hspace=0.05)
ci = [[0, 1, 2], [4, 5, 6]]
axs = {}
for g, (tag, name, A, S_) in enumerate((('mni', 'MNI152 atlas', mni, mseg), ('sub', 'sub-0036 atlas', sub, sseg))):
    sg, cr = cuts(S_)
    wm = np.isin(S_, (7, 46))
    for row, (p, i) in enumerate((('sag', sg), ('cor', cr))):
        m = take(np.isin(S_, (7, 8, 46, 47)), p, i)
        P, Q = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        lim = (P.min() - 8, P.max() + 8), (Q.min() - 8, Q.max() + 8)
        cols = [(take(A, p, i), take(S_, p, i), 'Fixed（%s）' % name), (take(vol, p, i), take(seg, p, i), 'Moving（affine）'),
                (Cc['%s_%s_img' % (tag, p)], Cc['%s_%s_seg' % (tag, p)], 'RCN 形變後')]
        for j, (im, sl, tt) in enumerate(cols):
            for k in range(2):
                a = fig.add_subplot(G[row * 2 + k, ci[g][j]])
                axs[row * 2 + k, g, j] = a
                a.imshow(im.T, origin='lower', cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                if k == 1:
                    overlay(a, sl)
                    if j > 0:
                        a.contour(take(wm, p, i).T, levels=[0.5], colors='white', linewidths=1.0)
                a.set_xlim(*lim[0]); a.set_ylim(*lim[1])
                a.axis('off')
                if row == 0 and k == 0:
                    a.set_title(tt, fontsize=9.5, color=INK, pad=3)
for g, name in enumerate(('MNI152 atlas', '真人 atlas（sub-0036）')):
    p0_, p2_ = axs[0, g, 0].get_position(), axs[0, g, 2].get_position()
    yy = p0_.y1 + 0.045
    fig.add_artist(Line2D([p0_.x0, p2_.x1], [yy - 0.006, yy - 0.006], color=RULE, lw=1.2, transform=fig.transFigure))
    fig.text((p0_.x0 + p2_.x1) / 2, yy, name, ha='center', va='bottom', fontsize=11, fontweight='bold', color=INK)
for row, pn in enumerate(('矢狀面', '冠狀面')):
    for k, s_ in ((0, '影像'), (1, '標籤')):
        pos = axs[row * 2 + k, 0, 0].get_position()
        fig.text(0.02, pos.y0 + pos.height / 2, pn + '\n' + s_, rotation=90, va='center', ha='center', fontsize=9, color=MUTED, linespacing=1.3)
fig.legend(handles=[Patch(color=L_WM, label='小腦白質'), Patch(color=L_CTX, label='小腦皮質'), Patch(color=L_BS, label='腦幹'),
                    Line2D([], [], color=MUTED, lw=1.5, label='Fixed 之小腦白質邊界（白線）')],
           loc='lower center', ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.52, 0.0))
save(fig, '1111_rcn_slices.png')
