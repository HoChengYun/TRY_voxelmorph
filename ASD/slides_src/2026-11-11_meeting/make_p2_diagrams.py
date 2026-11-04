# -*- coding: utf-8 -*-
"""Part 2：路線圖、原作重現版之公式（RCN、DWP）、修改一（RCN＋局部 NCC）之公式、未來規劃（cascade × coarse-to-fine）
→ models\\deck_charts\\1111_p2_roadmap.png、1111_eq_rcn.png、1111_eq_dwp.png、1111_eq_mod.png、1111_p2_future.png

公式依「重現版」之實際設定（手冊 §31.10）：RCN = rcn_torch.py（位移場、全域相關係數、affine stem、×1/n、reg_factor 1、det／ortho 各 0.1）；
DWP = 原作 rethink-reg（位移場、最細 1/2 解析度、雙向、子層 loss 2^-l、λ = 0.5 × loss_mult 2）。10/28 簡報之公式為舊設定（SVF、局部 NCC），不沿用。
記號同 VoxelMorph：(m∘φ)(x) = m(φ(x))。
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from style import OUT, INK, MUTED, PAPER, RULE, W, MODELS, save

M = json.load(open(os.path.join(OUT, '1111_p2_metrics.json'), encoding='utf-8'))


def eq_block(name, eqs, lines, h, fs=19, lfs=12.6, gap=0.3):
    fig = plt.figure(figsize=(W, h), facecolor=PAPER)
    y = 0.36
    for eq, num in eqs:
        fig.text(0.5, 1 - y / h, eq, ha='center', va='center', fontsize=fs, color=INK)
        fig.text(0.985, 1 - y / h, r'$(%d)$' % num, ha='right', va='center', fontsize=16, color=INK)
        y += 0.66
    y -= 0.66 - 0.48
    for i, ln in enumerate(lines):
        fig.text(0.02, 1 - (y + i * gap) / h, ln, ha='left', va='top', fontsize=lfs, color=INK)
    save(fig, name)


IN, SP = '其中　', '　　　'
eq_block('1111_eq_rcn.png',
         [(r'$\Phi_0=\mathcal{A}_{\theta_0}(m,f),\quad u_k=\frac{1}{n}\,g_{\theta_k}(m\circ\Phi_{k-1},\,f),\quad'
           r'\Phi_k=\Phi_{k-1}\circ(\mathrm{Id}+u_k)\quad(k=1,\ldots,n)$', 1),
          (r'$\mathcal{L}=\left[\,1-\rho\,(f,\;m\circ\Phi_n)\,\right]+\lambda\sum_{k=1}^{n}\mathcal{R}(u_k)'
           r'+0.1\,(\mathcal{L}_{\det}+\mathcal{L}_{\mathrm{ortho}})$', 2)],
         [IN + r'$\mathcal{A}_{\theta_0}$' + '：affine stem（可訓練）；' + r'$g_{\theta_k}$' + '：第 k 個 VTN（參數不共享，端對端聯合訓練），本研究 '
          + r'$n=3$' + '；' + r'$\frac{1}{n}$' + '：原作程式之設定',
          SP + r'$\rho$' + '：Pearson 相關係數，於整張影像（含背景）計算一個值（global correlation）；相似度僅計最終輸出',
          SP + r'$\mathcal{R}$' + '：位移場相鄰差平方和（各級皆計），' + r'$\lambda=1$' + '；' + r'$\mathcal{L}_{\det},\,\mathcal{L}_{\mathrm{ortho}}$'
          + '：affine 之行列式與正交性約束'],
         2.4)
eq_block('1111_eq_dwp.png',
         [(r'$u_l=g_l\left(\left[\,F_m^{\,l}\circ\tilde{\Phi}_{l+1},\ F_f^{\,l},\ \mathcal{U}(h_{l+1})\,\right]\right),\quad'
           r'\Phi_l=\tilde{\Phi}_{l+1}\circ(\mathrm{Id}+u_l)\quad(l=4,\ldots,1),\quad\Phi=\mathcal{U}(\Phi_1)$', 3),
          (r'$\mathcal{L}_{m\to f}=-\mathrm{NCC}(f,\,m\circ\Phi)+\lambda\Vert\nabla U_1\Vert^2'
           r'+\sum_{l=2}^{4}2^{-l}\left[-\mathrm{NCC}(f_{\downarrow l},\,m_{\downarrow l}\circ\Phi_l)+\lambda\Vert\nabla U_l\Vert^2\right],'
           r'\quad\mathcal{L}=\frac{1}{2}\left(\mathcal{L}_{m\to f}+\mathcal{L}_{f\to m}\right)$', 4)],
         [IN + r'$F_m^{\,l},\,F_f^{\,l}$' + '：權重共享之 encoder 第 ' + r'$l$' + ' 層特徵（解析度 ' + r'$2^{-l}$' + '）；'
          + r'$\tilde{\Phi}_{l+1}$' + '：上一層形變之 ×2 upsampling；' + r'$h$' + '：decoder feature；' + r'$\Phi_l=\mathrm{Id}+U_l$',
          SP + '最細層為 1/2 解析度，再 upsampling 至原尺寸；NCC 為局部（窗格 ' + r'$9^3$' + '）；' + r'$\lambda=0.5\times2=1$'
          + '（原作程式之 loss_mult）',
          SP + '雙向：同一網路另以 ' + r'$(f,m)$' + ' 對調再算一次；合成時 ' + r'$u_l$' + ' 作為取樣位置之梯度截斷（detach），同原作程式'],
         2.4, fs=17.2)
eq_block('1111_eq_mod.png',
         [(r'$\mathcal{L}_{\mathrm{sim}}=1-\rho\,(f,\;m\circ\Phi_n)\quad\longrightarrow\quad'
           r'\mathcal{L}_{\mathrm{sim}}=-\frac{1}{|\Omega|}\sum_{x\in\Omega}\rho_{N(x)}(f,\;m\circ\Phi_n)^2$', 5)],
         [IN + '左：原作（' + r'$\rho$' + '：整張影像一個相關係數）；右：修改（' + r'$\rho_{N(x)}$' + '：以 ' + r'$x$' + ' 為中心之 ' + r'$9^3$'
          + ' 窗格內之相關係數；同 VoxelMorph 式 (6) 之 local NCC）',
          SP + '僅更換相似度項；網路（affine stem＋VTN × 3）、' + r'$\lambda$' + '、資料、訓練量皆同原作重現版'],
         1.55, fs=19)

# ════ 路線圖 ════
fig = plt.figure(figsize=(W, 3.7), facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, 3.7); ax.axis('off')


def box(x, y, w, h, title, lines, col, status, scol):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc='white', ec=col, lw=1.8))
    ax.text(x + 0.14, y + h - 0.16, title, ha='left', va='top', fontsize=12, fontweight='bold', color=col)
    ax.text(x + 0.14, y + h - 0.52, '\n'.join(lines), ha='left', va='top', fontsize=9.6, color=INK, linespacing=1.45)
    ax.text(x + w - 0.12, y + 0.12, status, ha='right', va='bottom', fontsize=9.5, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc=scol, ec='none'))


GREEN, AMBER, GRAYC = '#3E9B3A', '#D08C2B', '#9A9A93'
v, r, d = M['vxm'], M['rcn'], M['dwp']
box(0.15, 1.15, 2.6, 1.6, '① VoxelMorph', ['原作設定（mix_exp3）', 'U-Net、local NCC、λ = 1',
    'Dice %.4f、%.2f M 參數' % (v['dice'], v['params'] / 1e6)], MODELS['vxm'][2], '完成', GREEN)
box(3.35, 2.15, 2.9, 1.35, '② RCN（cascade）', ['Affine＋VTN × 3、global correlation', 'Dice %.4f、%.1f M 參數' % (r['dice'], r['params'] / 1e6)],
    MODELS['rcn'][2], '完成', GREEN)
box(3.35, 0.4, 2.9, 1.35, '② DWP（coarse-to-fine）', ['Dual-stream pyramid、local NCC', 'Dice %.4f、%.2f M 參數' % (d['dice'], d['params'] / 1e6)],
    MODELS['dwp'][2], '完成', GREEN)
box(6.85, 2.15, 2.5, 1.35, '③ 修改 RCN', ['global correlation → local NCC', '其餘同原作重現版'], '#D08C2B', '訓練中', AMBER)
box(9.95, 1.15, 2.05, 1.6, '④ Cascade × C2F', ['RCN 之每一級改為', 'DWP 之雙流金字塔'], '#7A4FA0', '規劃中', GRAYC)


def arrow(x1, y1, x2, y2, t=None, dy=0.08):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14, color=MUTED, lw=1.3))
    if t:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, t, ha='center', va='bottom', fontsize=8.8, color=MUTED, linespacing=1.25)


arrow(2.78, 2.3, 3.32, 2.8, '原作程式', 0.12)
arrow(2.78, 1.6, 3.32, 1.1)
arrow(6.28, 2.82, 6.82, 2.82, '換相似度')
arrow(9.38, 2.6, 9.92, 2.25)
arrow(6.28, 1.08, 9.92, 1.6, '結合', -0.32)
ax.text(1.45, 0.55, '原作重現：僅改資料、atlas、影像大小、\n訓練量（2.5 萬步）、評估', ha='center', va='center', fontsize=8.8, color=MUTED, linespacing=1.3)
save(fig, '1111_p2_roadmap.png')

# ════ 未來規劃：cascade × coarse-to-fine ════
fig = plt.figure(figsize=(W, 2.9), facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, 2.9); ax.axis('off')


def node(x, y, w, h, t, col=INK, fc='white', fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=col, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.3)


node(0.2, 1.75, 1.3, 0.6, r'Moving $m$')
node(0.2, 0.55, 1.3, 0.6, r'Fixed $f$')
node(2.0, 1.0, 1.6, 0.9, 'Affine stem\n' + r'$\Phi_0$', col=MUTED)
node(4.1, 0.8, 2.2, 1.3, 'Stage 1\nDual-stream pyramid\n' + r'$\phi_1$', col='#0E7C7B', fc='#E8F3F2')
node(7.0, 0.8, 2.2, 1.3, 'Stage 2\nDual-stream pyramid\n' + r'$\phi_2$', col='#0E7C7B', fc='#E8F3F2')
node(9.9, 1.0, 2.0, 0.9, r'$\Phi=\Phi_0\circ\phi_1\circ\phi_2$' + '\n' + r'$m\circ\Phi$', col='#7A4FA0')
for (x1, y1, x2, y2) in ((1.52, 2.05, 1.98, 1.6), (1.52, 0.85, 1.98, 1.3), (3.62, 1.45, 4.08, 1.45), (6.32, 1.45, 6.98, 1.45), (9.22, 1.45, 9.88, 1.45)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, color=MUTED, lw=1.2))
ax.text(6.65, 1.62, r'warp $m$', ha='center', va='bottom', fontsize=9, color=MUTED)
ax.text(6.1, 0.3, '相似度僅計最終輸出（RCN）；各 stage 內含子層 loss（DWP）；相似度採 local NCC（依修改一之結果）', ha='center', va='bottom',
        fontsize=9.5, color=MUTED)
save(fig, '1111_p2_future.png')
