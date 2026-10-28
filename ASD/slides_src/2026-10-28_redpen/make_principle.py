# -*- coding: utf-8 -*-
"""⑥ 架構修改之前：兩篇原作之核心原理（2026-10-28 使用者：「可以在 p11 前加上 cascade 和 pyramid 那兩篇的論文核心原理，
讓老師和我熟悉一下」「結構圖也可以考慮用原文的」）→ models\\deck_charts\\1028_pr_*.png

  1028_pr_rcn_fig.png   RCN（Zhao et al., ICCV 2019）Fig. 3，原文 PDF 第 3 頁裁切
  1028_pr_rcn_eq.png    公式 (1)、(2)：遞迴合成與 loss（含原作程式之 ×1/n）
  1028_pr_dwp_fig.png   DWP（Jian et al., WBIR 2024，"Mamba? Catch The Hype Or Rethink What Really Helps for Image Registration"）
                        Fig. 1，原文 PDF 第 5 頁裁切
  1028_pr_dwp_eq.png    公式 (3)、(4)：逐層 warp、合成與 loss（含原作程式之 detach、子層 loss）

原文 PDF 在 文獻\\reference\\（只留本機，不進版控）；裁出來的圖在 models\\deck_charts（.gitignore），也不會進公開的 repo。
公式之記號同 VoxelMorph（(m∘φ)(x) = m(φ(x))），內容與 ASD/arch.py、ASD/train_arch.py 一致（--rcn、--dwp；手冊 §29）：
  RCN：每級輸出 ×1/n、相似度只算最後、平滑項每級都算；本研究每級為 VoxelMorph（SVF）、n = 2
  DWP：shared encoder 分別抽 m、f 之特徵；每層以上一層形變（×2 upsampling）warp m 之特徵 → decoder → v_l → exp → 合成；
       合成時 φ_l 當取樣位置那條路 detach；子層 l = 2、3、4 另算 2^-l [−NCC(f↓, m↓∘Φ_l) + λ‖∇U_l‖²]
用法：python ASD\\slides_src\\2026-10-28_redpen\\make_principle.py
"""
import os
import pymupdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
REF = os.path.join(ROOT, '文獻', 'reference')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
W, DPI = 12.13, 200
INK, PAPER = '#141A1D', '#FAFAF8'


# ── 原文之架構圖：PyMuPDF 只算裁切範圍（單位 pt；先以 110 dpi 預覽量出像素座標，×72/110 換算）──────────────
def crop(pdf, page, box_px110, name, dpi=300):
    doc = pymupdf.open(os.path.join(REF, pdf))
    k = 72 / 110
    pix = doc[page - 1].get_pixmap(dpi=dpi, clip=pymupdf.Rect(*[v * k for v in box_px110]))
    p = os.path.join(OUT, name)
    pix.save(p)
    print('-> %s（%d × %d）' % (p, pix.width, pix.height))


crop('Recursive Cascaded Networks for Unsupervised Medical Image Registration.pdf', 3, (105, 105, 805, 297), '1028_pr_rcn_fig.png')
crop('Mamba Catch The Hype Or Rethink What Really Helps for Image Registration.pdf', 5, (198, 165, 752, 520), '1028_pr_dwp_fig.png')


# ── 公式區塊（同 2026-10-14_redpen/make_arch.py 的 eq_block：公式置中、編號靠右，下面是「其中」）──────────────
def eq_block(name, eqs, lines, h, gap=0.31, fs=20, lfs=13):
    fig = plt.figure(figsize=(W, h), facecolor=PAPER)
    y = 0.36
    for eq, num in eqs:
        fig.text(0.5, 1 - y / h, eq, ha='center', va='center', fontsize=fs, color=INK)
        fig.text(0.985, 1 - y / h, r'$(%d)$' % num, ha='right', va='center', fontsize=17, color=INK)
        y += 0.66
    y -= 0.66 - 0.48
    for i, ln in enumerate(lines):
        fig.text(0.02, 1 - (y + i * gap) / h, ln, ha='left', va='top', fontsize=lfs, color=INK)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


IN, SP = '其中　', '　　　'
eq_block('1028_pr_rcn_eq.png',
         [(r'$v_k=\frac{1}{n}\,g_{\theta_k}(m\circ\Phi_{k-1},\,f),\quad\phi_k=\exp(v_k),\quad'
           r'\Phi_k=\Phi_{k-1}\circ\phi_k\quad(k=1,\ldots,n;\ \Phi_0=\mathrm{Id})$', 1),
          (r'$\mathcal{L}=-\mathrm{NCC}(f,\,m\circ\Phi_n)+\lambda\sum_{k=1}^{n}\Vert\nabla v_k\Vert^2$', 2)],
         # 10-28 使用者問「p11 p12 公式都是對的嗎」→ 逐項對過 arch.py／train_arch.py 與原文（手冊 §29）：式子與本研究之實作一致；
         # 第 3 行標明哪些是本研究之設定（原文：子網路直接輸出位移場、相似度為 correlation coefficient，§4.1 Implementation）
         [IN + r'$g_{\theta_k}$' + '：第 k 級子網路（原文圖之 ' + r'$f_k$' + '；參數不共享，端對端聯合訓練）；'
          + '相似度僅計最終輸出，平滑項計於各級（同原文）',
          SP + r'$\frac{1}{n}$' + '：原作程式之設定（論文未寫；各級之初始形變較小）；本研究 ' + r'$n=2$' + '，'
          + r'$\exp$' + '（SVF）、局部 NCC、' + r'$\lambda=1$' + ' 同 mix_exp6',
          SP + '原文：子網路直接輸出位移場（無 ' + r'$\exp$' + '），相似度為全域相關係數'],
         2.45, fs=20)
eq_block('1028_pr_dwp_eq.png',
         [(r'$v_l=g_l\left(\left[\,F_m^{\,l}\circ\tilde{\Phi}_{l+1},\ F_f^{\,l},\ \mathcal{U}(h_{l+1})\,\right]\right),\quad'
           r'\phi_l=\exp(v_l),\quad\Phi_l=\tilde{\Phi}_{l+1}\circ\phi_l\quad(l=4,\ldots,0)$', 3),
          (r'$\mathcal{L}=-\mathrm{NCC}(f,\,m\circ\Phi_0)+\lambda\sum_{l=0}^{4}\Vert\nabla v_l\Vert^2'
           r'+\sum_{l=2}^{4}2^{-l}\left[-\mathrm{NCC}(f_{\downarrow l},\,m_{\downarrow l}\circ\Phi_l)'
           r'+\lambda\Vert\nabla U_l\Vert^2\right]$', 4)],
         # 10-28 核對：子層 loss 論文 Implementation Details 有寫（「2^-4 (L_sim + λL_reg)」、[1/16, 1/8, 1/4]），detach 只在程式（vxm_pwc.py 第 97 行）；
         # 原文：位移場（最細一層 1/2 解析度）、λ = 0.5，另有 Dice loss（γ = 0.5，半監督）
         [IN + r'$F_m^{\,l},\,F_f^{\,l}$' + '：shared encoder 第 ' + r'$l$' + ' 層特徵（' + r'$F^{\,0}$'
          + ' 為影像）；' + r'$\tilde{\Phi}_{l+1}=\mathcal{U}(\Phi_{l+1})$' + '：上一層形變之 ×2 upsampling；' + r'$h$' + '：decoder feature；'
          + r'$\Phi_l=\mathrm{Id}+U_l$',
          SP + '子層 loss（' + r'$l=2,3,4$' + '，權重 ' + r'$2^{-l}$' + '；' + r'$f_{\downarrow},\,m_{\downarrow}$'
          + '：average pooling）同原文；合成時 ' + r'$\phi_l$' + ' 作為取樣位置之梯度截斷（detach，前向不變）同原作程式',
          SP + r'$\exp$' + '（每層 SVF）、' + r'$\lambda=1$' + '、非監督為本研究之設定；原文：位移場、' + r'$\lambda=0.5$'
          + '，另加 Dice loss（' + r'$\gamma=0.5$' + '）'],
         2.45, fs=18.5, lfs=12.6)
