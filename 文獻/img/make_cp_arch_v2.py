# -*- coding: utf-8 -*-
"""文獻/cascade與pyramid原理筆記.md 用的「新版」架構圖（2026-10-21，照原作程式補上細節之後；ASD/arch.py、手冊 §29）。

畫法照 ASD/slides_src/2026-10-14_redpen/make_arch.py（10/14 簡報的 1014_arch_cascade.png、1014_arch_pyramid_box.png），
結構不變，**紅字標出新加的部分**：
  cp_arch_cascade_v2.png      cascade（--rcn）：每顆輸出 ×1/n；串接那一層的 ST 與合成取邊界值（padding = border）
  cp_arch_pyramid_box_v2.png  coarse-to-fine 的 decoder 第 l 層（--dwp）：合成時 φ_l 當取樣位置的那條路 detach；
                              l = 2、3、4（1/4、1/8、1/16）的總形變 Φ_l 另外算子層 loss
10/14 簡報用的原圖（models\\deck_charts\\1014_arch_*.png）不動。

用法：python 文獻\\img\\make_cp_arch_v2.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
W, DPI = 12.13, 200
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, PURPLE = '#0E7C7B', '#7F77DD'
_ENC, _DEC, _RED = '#F1EFE8', '#E6F2F1', '#B03A2E'
NEW = '#C0392B'                                            # 新加的部分


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def canvas(h):
    fig = plt.figure(figsize=(W, h), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W * 10)
    ax.set_ylim(0, h * 10)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, text, fc='#FFFFFF', ec=INK, fs=15, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, zorder=3)


def arr(ax, p0, p1, text=None, dy=0.9, fs=14, ls='-', color=INK, ha='center'):
    ax.annotate('', xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle='-|>', lw=1.2, color=color, linestyle=ls, mutation_scale=12, shrinkA=0, shrinkB=0))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, text, ha=ha, va='bottom', fontsize=fs, color=INK)


def poly(ax, pts, color=INK, ls='-', lw=1.2, head=True):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=color, ls=ls, lw=lw, zorder=1)
    if head:
        arr(ax, pts[-2], pts[-1], color=color, ls=ls)


# ── cascade（--rcn）──────────────────────────────────────────────────
fig, ax = canvas(2.95)
Y = 16.5
ax.add_patch(FancyBboxPatch((9.5, 8.5), 31.5, 17.5, boxstyle='round,pad=0,rounding_size=1', fc=TEAL, alpha=0.07, ec='none'))
ax.add_patch(FancyBboxPatch((42.5, 8.5), 32.5, 17.5, boxstyle='round,pad=0,rounding_size=1', fc=PURPLE, alpha=0.08, ec='none'))
ax.text(10.5, 24.6, 'Stage 1', fontsize=13, fontweight='bold', color=TEAL, va='center')
ax.text(43.5, 24.6, 'Stage 2', fontsize=13, fontweight='bold', color=PURPLE, va='center')
box(ax, 1, 19.5, 6, 6, r'$m$', fc='#F1F0EA')
box(ax, 1, 7.5, 6, 6, r'$f$', fc='#F1F0EA')
box(ax, 12, Y - 3, 9, 6, r'$g_{\theta_1}$', fs=16)
arr(ax, (7, 22.5), (12, Y + 1.5))
arr(ax, (7, 10.5), (12, Y - 1.5))
arr(ax, (21, Y), (24.5, Y), r'$v_1$')
box(ax, 24.5, Y - 3, 6, 6, r'$\exp$')
arr(ax, (30.5, Y), (34, Y), r'$\phi_1$')
box(ax, 34, Y - 3, 5.5, 6, 'ST', fs=13)
arr(ax, (39.5, Y), (46, Y), r'$m\circ\phi_1$')
box(ax, 46, Y - 3, 9, 6, r'$g_{\theta_2}$', fs=16)
arr(ax, (55, Y), (58.5, Y), r'$v_2$')
box(ax, 58.5, Y - 3, 6, 6, r'$\exp$')
arr(ax, (64.5, Y), (68.5, Y), r'$\phi_2$')
ax.add_patch(Circle((71, Y), 2.4, fc='#FFFFFF', ec=INK, lw=1.2, zorder=2))
ax.text(71, Y, r'$\circ$', ha='center', va='center', fontsize=20, zorder=3)
arr(ax, (73.4, Y), (84, Y), r'$\Phi=\phi_1\circ\phi_2$')
box(ax, 84, Y - 3, 5.5, 6, 'ST', fs=13)
arr(ax, (89.5, Y), (95, Y), r'$m\circ\Phi$')
box(ax, 95, Y - 4.5, 25, 9, r'$-\mathrm{NCC}(f,\,m\circ\Phi)$' + '\n' + r'$+\,\lambda\sum_k\Vert\nabla v_k\Vert^2$', fs=14, fc='#FFF6EE')
poly(ax, [(4, 25.5), (4, 27.6), (36.75, 27.6), (36.75, Y + 3)])
poly(ax, [(36.75, 27.6), (86.75, 27.6), (86.75, Y + 3)])
poly(ax, [(4, 7.5), (4, 4.6), (50.5, 4.6), (50.5, Y - 3)])
poly(ax, [(50.5, 4.6), (107.5, 4.6), (107.5, Y - 4.5)])
poly(ax, [(32.25, Y), (32.25, 10.6), (71, 10.6), (71, Y - 2.4)], color=TEAL)
ax.text(33.2, 11.4, r'$\phi_1$', fontsize=13, color=TEAL, va='bottom')
# 新加的（紅字）
for gx in (16.5, 50.5):
    ax.text(gx, Y + 3.6, r'輸出 $\times\frac{1}{n}$', ha='center', va='bottom', fontsize=11, color=NEW, fontweight='bold')
for sx in (36.75, 86.75):
    ax.text(sx + 3.4, Y - 4.2, 'border', ha='left', va='center', fontsize=9.5, color=NEW, fontweight='bold')
ax.text(73.6, Y - 3.6, 'border', ha='left', va='center', fontsize=9.5, color=NEW, fontweight='bold')
ax.text(121, 0.6, 'ST：spatial transformer（trilinear）　exp：scaling and squaring　'
        '紅字：照 RCN 原作補上（--rcn）：每顆輸出 ×1/n（n = 2），平滑項算在乘過的 v_k 上；串接那一層的 ST 與合成取邊界值',
        ha='right', va='bottom', fontsize=9.5, color=MUTED)
save(fig, 'cp_arch_cascade_v2.png')

# ── coarse-to-fine：decoder 第 l 層（--dwp）──────────────────────────────
fig, ax = canvas(4.75)
LP, LM, LC, LF = 41.5, 34, 26.5, 19                  # 比原圖整體往上 4.5，下面留給新加的子層 loss
LV, LH = 19.5, 15
ax.add_patch(FancyBboxPatch((11.5, 12), 87.8, 34.8, boxstyle='round,pad=0,rounding_size=1', fc='#F0F7F6', ec=TEAL, lw=1.3,
                            zorder=0))
ax.text(12.8, 45.1, r'$D_l$' + '：decoder 第 ' + r'$l$' + ' 層', ha='left', va='center', fontsize=12.5, color=TEAL,
        fontweight='bold')
for y, t, fc in [(LP, r'$\Phi_{l+1}$', _DEC), (LM, r'$F_m^{\,l}$', _ENC), (LC, r'$h_{l+1}$', _DEC), (LF, r'$F_f^{\,l}$', _ENC)]:
    box(ax, 1, y - 2.5, 9, 5, t, fc=fc, fs=15)
for y, sub in [(LP, 'trilinear, ×2'), (LC, 'nearest')]:
    box(ax, 13.5, y - 2.5, 9, 5, '')
    ax.text(18, y + 0.9, r'$\mathcal{U}$', ha='center', va='center', fontsize=15, color=INK, zorder=3)
    ax.text(18, y - 1.3, sub, ha='center', va='center', fontsize=8.5, color=MUTED, zorder=3)
    arr(ax, (10, y), (13.5, y))
PX, CX = 26.5, 93
ax.plot([22.5, CX], [LP, LP], color=TEAL, lw=1.3, ls=':', zorder=1)
arr(ax, (PX, LP), (PX, LM + 1.8), color=TEAL, ls=':')
arr(ax, (CX, LP), (CX, LC + 2), color=TEAL, ls=':')
ax.text(60, LP + 0.5, r'$\tilde{\Phi}_{l+1}=\mathcal{U}(\Phi_{l+1})$', ha='center', va='bottom', fontsize=13, color=TEAL)
ax.add_patch(Circle((PX, LM), 1.8, fc='white', ec=INK, lw=1.2, zorder=4))
ax.text(PX, LM, 'W', ha='center', va='center', fontsize=11, zorder=5)
arr(ax, (10, LM), (PX - 1.8, LM))
arr(ax, (PX + 1.8, LM), (38, LM), r'$F_m^{\,l}\circ\tilde{\Phi}_{l+1}$', dy=0.5, fs=12)
arr(ax, (22.5, LC), (38, LC), r'$\mathcal{U}(h_{l+1})$', dy=0.5, fs=12)
arr(ax, (10, LF), (38, LF))
for y in (LM, LC, LF):
    ax.text(35.3, y - 0.6, '(32)', ha='center', va='top', fontsize=9.5, color=MUTED)
box(ax, 38, 16, 5, 20.5, r'$[\cdot]$', fs=15)
arr(ax, (43, LC), (47, LC))
ax.add_patch(FancyBboxPatch((45.5, LC - 5), 31, 11, boxstyle='round,pad=0,rounding_size=0.8', fc='none', ec=TEAL, lw=1.2,
                            ls='--', zorder=1))
ax.text(61, LC + 7.6, r'$g_l$', ha='center', va='center', fontsize=15, color=TEAL)
box(ax, 47, LC - 3, 11, 6, '3×3×3 conv\n+ LeakyReLU\n96 → 32 ch', fs=9.5)
arr(ax, (58, LC), (64, LC), r'$h_l$', dy=0.4, fs=13)
box(ax, 64, LC - 3, 11, 6, '3×3×3 conv\n32 → 3 ch', fs=9.5)
arr(ax, (75, LC), (81, LC), r'$v_l$', dy=0.4, fs=13)
box(ax, 81, LC - 3, 7, 6, r'$\exp$', fs=14)
arr(ax, (88, LC), (CX - 2, LC), r'$\phi_l$', dy=0.4, fs=13)
ax.add_patch(Circle((CX, LC), 2, fc='white', ec=INK, lw=1.2, zorder=4))
ax.text(CX, LC, r'$\circ$', ha='center', va='center', fontsize=17, zorder=5)
arr(ax, (CX + 2, LC), (101, LC), r'$\Phi_l$', dy=0.4, fs=13)
poly(ax, [(61, LC), (61, LH), (101, LH)])
poly(ax, [(78, LC), (78, LV), (101, LV)], color=_RED, ls='--')
ax.text(97, LH + 0.4, r'$h_l$', ha='center', va='bottom', fontsize=12)
ax.text(97, LV + 0.4, r'$v_l$', ha='center', va='bottom', fontsize=12, color=_RED)
ax.text(101.8, LC, r'$D_{l-1}$' + '（' + r'$l=0$' + '：輸出 ' + r'$\Phi_0$' + '）', ha='left', va='center', fontsize=11.5)
ax.text(101.8, LV, r'$\lambda\Vert\nabla v_l\Vert^2$', ha='left', va='center', fontsize=11.5, color=_RED)
ax.text(101.8, LH, r'$D_{l-1}$', ha='left', va='center', fontsize=11.5)
# 新加的（紅字）：① 合成時 φ_l 當取樣位置的那條路 detach；② Φ_l 另外算子層 loss
ax.text(63.5, LC + 6.8, r'$\Phi_l=\phi_l+\tilde{\Phi}_{l+1}(x+\mathrm{detach}(\phi_l))$', ha='left', va='bottom',
        fontsize=10, color=NEW)
poly(ax, [(100, LC), (100, 8), (98.6, 8)], color=NEW, ls='--')
ax.text(98, 8, r'$l=2,3,4$' + '：子層 loss ' + r'$2^{-l}\,[-\mathrm{NCC}(f_{\downarrow},\,m_{\downarrow}\circ\Phi_l)+\lambda\Vert\nabla\Phi_l\Vert^2]$',
        ha='right', va='center', fontsize=10.5, color=NEW)
ax.text(1, 3.4, '紅字：照 DWP 原作補上（--dwp）。detach 只切「取樣位置」那條路的梯度，往前算出來的 ' + r'$\Phi_l$'
        + ' 不變；子層 loss 用 average pooling 縮小的 m、f（' + r'$m_{\downarrow}$' + '、' + r'$f_{\downarrow}$'
        + '），權重 1/16、1/8、1/4', ha='left', va='center', fontsize=10, color=MUTED)
ax.text(1, 0.9, '灰底：encoder 特徵；青底：' + r'$D_{l+1}$' + ' 之輸出；括號為通道數（' + r'$l=2$' + '）；'
        + r'$l=4$' + '：' + r'$\tilde{\Phi}_5=\mathrm{Id}$' + '、無 ' + r'$h_5$' + '；' + r'$l=0$' + '：'
        + r'$F^{\,0}$' + ' 為影像本身', ha='left', va='center', fontsize=10, color=MUTED)
save(fig, 'cp_arch_pyramid_box_v2.png')
