# -*- coding: utf-8 -*-
"""③ 合成時 detach 之說明（11-11 使用者：「感覺可以放簡報上，不然我會被問爆」）→ models\\deck_charts\\1111_detach.png

上：示意圖——前向計算（兩者相同）與訓練時梯度之兩條路徑；detach 截斷「u 改變取樣位置」那一條。
下：式 (6) 合成、式 (7) 對 u 之偏微分（不 detach：I + ∇Ũ；detach：I）。
依 ASD/arch.py VxmPyramid.fields：U = u + warp(U↑, u.detach() if detach_compose else u)（DWP 原作 vxm_pwc.py 第 97 行）。
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

from style import INK, MUTED, PAPER, W, save

GRAD, CUT = '#D08C2B', '#B5487A'
H = 4.1
fig = plt.figure(figsize=(W, H), facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')


def box(x, y, w, h, t, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc='white', ec=INK, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.35)


def arr(p, q, col=MUTED, ls='-', lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=13, color=col, lw=lw, ls=ls,
                                 connectionstyle='arc3,rad=%g' % rad))


# 節點：上排 u、下排 Ũ、中間取樣、右邊相加與輸出
box(0.3, 2.75, 2.0, 0.6, r'$\varphi_l=\exp(v_l)$' + '\n本層之修正（位移場）', fs=9.5)
box(0.3, 0.95, 2.0, 0.6, r'$\tilde{\Phi}_{l+1}=\mathcal{U}(\Phi_{l+1})$' + '\n上一層之總形變，放大 2 倍', fs=9.5)
box(4.3, 0.95, 2.5, 0.6, '於位置 ' + r'$x+\varphi_l(x)$' + ' 取 ' + r'$\tilde{\Phi}_{l+1}$', fs=10.5)
ax.add_patch(Circle((8.5, 2.0), 0.2, fc='white', ec=INK, lw=1.5))
ax.text(8.5, 2.0, '+', ha='center', va='center', fontsize=14, color=INK)
box(9.5, 1.7, 2.3, 0.6, r'$\Phi_l$' + '　本層之總形變')


def line(xs, ys, col=MUTED, ls='-', lw=1.5):
    ax.plot(xs, ys, color=col, ls=ls, lw=lw, solid_capstyle='butt')


# 前向（實線，有無 detach 皆相同）
line([2.3, 8.5], [3.05, 3.05]); arr((8.5, 3.05), (8.5, 2.22))                 # ① u 直接相加
ax.text(5.4, 2.97, '① 直接相加', ha='center', va='top', fontsize=10, color=MUTED)
arr((2.3, 2.85), (4.75, 1.57))                                                 # ② u 決定取樣位置
ax.text(4.05, 2.3, '② 決定取樣位置', ha='left', va='center', fontsize=10, color=MUTED)
arr((2.3, 1.3), (4.28, 1.3))                                                   # Ũ 的值
line([6.8, 8.5], [1.3, 1.3]); arr((8.5, 1.3), (8.5, 1.78))
arr((8.7, 2.0), (9.48, 2.0))
# 梯度（虛線，往回）
line([8.75, 8.75, 2.32], [2.1, 3.28, 3.28], GRAD, ls=(0, (4, 2)), lw=1.6)
arr((2.5, 3.28), (2.32, 3.28), GRAD, lw=1.6)
ax.text(5.4, 3.36, '梯度 I：保留', ha='center', va='bottom', fontsize=10.5, color=GRAD, fontweight='bold')
arr((4.95, 1.72), (2.45, 3.02), GRAD, ls=(0, (4, 2)), lw=1.6)
ax.text(3.72, 2.38, '✕', ha='center', va='center', fontsize=22, color=CUT, fontweight='bold')
ax.text(0.35, 2.15, '梯度 ' + r'$\nabla\tilde{\Phi}_{l+1}$' + '：\ndetach 後截斷', ha='left', va='center', fontsize=10.5, color=CUT,
        fontweight='bold', linespacing=1.3)
arr((4.28, 1.08), (2.32, 1.08), GRAD, ls=(0, (4, 2)), lw=1.6)
ax.text(3.3, 0.9, r'$\tilde{\Phi}_{l+1}$' + ' 照常收到梯度', ha='center', va='top', fontsize=10, color=GRAD)
# 圖例
ax.plot([0.3, 0.8], [3.75, 3.75], color=MUTED, lw=1.5)
ax.text(0.9, 3.75, '前向計算（有無 detach 皆相同）', va='center', fontsize=10.5, color=INK)
ax.plot([4.2, 4.7], [3.75, 3.75], color=GRAD, lw=1.6, ls=(0, (4, 2)))
ax.text(4.8, 3.75, '訓練時回傳之梯度', va='center', fontsize=10.5, color=INK)
ax.text(7.05, 3.75, '✕', va='center', ha='center', fontsize=15, color=CUT, fontweight='bold')
ax.text(7.25, 3.75, 'detach 截斷之路徑', va='center', fontsize=10.5, color=INK)
# 公式
ax.plot([0.3, W - 0.3], [0.62, 0.62], color='#D9D9D2', lw=0.8)
ax.text(W / 2, 0.36, r'$\Phi_l(x)=\varphi_l(x)+\tilde{\Phi}_{l+1}\!\left(x+\varphi_l(x)\right),\qquad'
        r'\frac{\partial \Phi_l}{\partial \varphi_l}=I+\nabla\tilde{\Phi}_{l+1}\!\left(x+\varphi_l(x)\right)\ \quad\longrightarrow\quad\mathrm{(detach)}\ \frac{\partial \Phi_l}{\partial \varphi_l}=I$',
        ha='center', va='center', fontsize=17, color=INK)
ax.text(W - 0.25, 0.36, r'$(6)$', ha='right', va='center', fontsize=15, color=INK)
save(fig, '1111_detach.png')
