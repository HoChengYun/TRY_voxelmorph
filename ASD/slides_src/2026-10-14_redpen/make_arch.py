# -*- coding: utf-8 -*-
"""10/14 簡報 ⑥ 架構修改：架構圖（block diagram）與公式區塊 -> models/deck_charts/1014_arch_*.png。

2026-10-07 使用者：「正式一點、公式 block 都很重要、不要太口語」→ 學術用語、paper 式架構圖、編號公式（同 make_method.py 的樣式）。

  1014_arch_step0_eq.png     Step 0：test-time recursion 的公式 (1)
  1014_arch_cascade.png      Step 1：cascade 架構圖
  1014_arch_cascade_eq.png   Step 1：公式 (2)、(3)
  1014_arch_pyramid.png      Step 2：coarse-to-fine 架構圖
  1014_arch_unet_vs_pyramid.png  Step 2 與 VoxelMorph U-Net 之比較（直式 U 對照，標出三處不同；放在 Step 2 那頁前面）
  1014_arch_pyramid_eq.png   Step 2：公式 (4)、(5)

記號照 VoxelMorph（Balakrishnan et al., TMI 2019）：(m∘φ)(x) = m(φ(x))，φ = Id + u。
先 φ1、再對 m∘φ1 估計 φ2，總形變是 Φ = φ1 ∘ φ2，位移寫法 U(x) = u2(x) + u1(x + u2(x))，跟 ASD/arch.py 一致。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
W, DPI = 12.13, 200                                   # 投影片上的實際寬度（吋）
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, PURPLE, GRAY = '#0E7C7B', '#7F77DD', '#888780'


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


def canvas(h):
    """座標單位 0.1 吋：寬 121.3、高 h*10。"""
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


# ── Step 1：cascade ─────────────────────────────────────────────────────
fig, ax = canvas(2.75)
Y = 14.5                                             # 主軸
ax.add_patch(FancyBboxPatch((9.5, 6.5), 31.5, 17.5, boxstyle='round,pad=0,rounding_size=1', fc=TEAL, alpha=0.07, ec='none'))
ax.add_patch(FancyBboxPatch((42.5, 6.5), 32.5, 17.5, boxstyle='round,pad=0,rounding_size=1', fc=PURPLE, alpha=0.08, ec='none'))
ax.text(10.5, 22.6, 'Stage 1', fontsize=13, fontweight='bold', color=TEAL, va='center')
ax.text(43.5, 22.6, 'Stage 2', fontsize=13, fontweight='bold', color=PURPLE, va='center')
box(ax, 1, 17.5, 6, 6, r'$m$', fc='#F1F0EA')
box(ax, 1, 5.5, 6, 6, r'$f$', fc='#F1F0EA')
box(ax, 12, Y - 3, 9, 6, r'$g_{\theta_1}$', fs=16)
arr(ax, (7, 20.5), (12, Y + 1.5))
arr(ax, (7, 8.5), (12, Y - 1.5))
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
# m 進兩個 ST（上方繞線）、f 進 g_θ2 與 loss（下方繞線）、φ1 進合成（下方繞線）
poly(ax, [(4, 23.5), (4, 25.6), (36.75, 25.6), (36.75, Y + 3)])
poly(ax, [(36.75, 25.6), (86.75, 25.6), (86.75, Y + 3)])
poly(ax, [(4, 5.5), (4, 2.6), (50.5, 2.6), (50.5, Y - 3)])
poly(ax, [(50.5, 2.6), (107.5, 2.6), (107.5, Y - 4.5)])
poly(ax, [(32.25, Y), (32.25, 8.6), (71, 8.6), (71, Y - 2.4)], color=TEAL)
ax.text(33.2, 9.4, r'$\phi_1$', fontsize=13, color=TEAL, va='bottom')
ax.text(121, 0.4, 'ST：spatial transformer（trilinear）　exp：scaling and squaring', ha='right', va='bottom', fontsize=10.5, color=MUTED)
save(fig, '1014_arch_cascade.png')

# ── Step 2：coarse-to-fine（dual-stream encoder + 每層 warp 與 composition）────────────────────
fig, ax = canvas(3.45)
XS = [22, 42, 62, 82, 102]                            # l = 0（原解析度）… 4（1/16）
RES = ['1/1', '1/2', '1/4', '1/8', '1/16']
YT, YM, YB = 28, 17.5, 6.5                            # F_m、decoder、F_f 三排的中心（底下留給註腳）
YW = 22.9                                             # W（warp）圓圈的高度
SZ = [(8, 6.4), (7.2, 5.6), (6.4, 4.8), (5.6, 4.2), (5, 3.6)]   # 越粗的層畫越小
for l, (x, (bw, bh)) in enumerate(zip(XS, SZ)):
    ax.text(x, 33.1, '%s（$l=%d$）' % (RES[l], l), ha='center', va='center', fontsize=11.5, color=MUTED)
    top = r'$m$' if l == 0 else r'$F_m^{\,%d}$' % l
    bot = r'$f$' if l == 0 else r'$F_f^{\,%d}$' % l
    box(ax, x - bw / 2, YT - bh / 2, bw, bh, top, fc='#F1F0EA', fs=14)
    box(ax, x - bw / 2, YB - bh / 2, bw, bh, bot, fc='#F1F0EA', fs=14)
    box(ax, x - 4.5, YM - 3, 9, 6, r'$D_{%d}$' % l, fc='#E7F3F2', ec=TEAL, fs=15)
    arr(ax, (x, YB + bh / 2), (x, YM - 3))                       # F_f^l -> D_l
    if l < 4:                                                     # F_m^l -> W -> D_l（最粗層沒有前一層的形變，直接進）
        ax.add_patch(Circle((x, YW), 1.6, fc='#FFFFFF', ec=INK, lw=1.1, zorder=2))
        ax.text(x, YW, 'W', ha='center', va='center', fontsize=9.5, zorder=3)
        arr(ax, (x, YT - bh / 2), (x, YW + 1.6))
        arr(ax, (x, YW - 1.6), (x, YM + 3))
    else:
        arr(ax, (x, YT - bh / 2), (x, YM + 3))
    if l > 0:                                                     # encoder：步長 2 卷積（兩邊共用權重）
        pw = SZ[l - 1][0]
        arr(ax, (XS[l - 1] + pw / 2, YT), (x - bw / 2, YT), 'E', dy=0.5, fs=11)
        arr(ax, (XS[l - 1] + pw / 2, YB), (x - bw / 2, YB), 'E', dy=0.5, fs=11)
for l in range(4, 0, -1):                                         # decoder：由粗到細，Φ 與 h 放大 2 倍
    x0, x1 = XS[l] - 4.5, XS[l - 1] + 4.5
    arr(ax, (x0, YM), (x1, YM), r'$\mathcal{U}$', dy=0.4, fs=13)
    ax.plot([(x0 + x1) / 2 - 1.5, XS[l - 1] + 1.2], [YM + 0.2, YW - 0.2], color=TEAL, ls=':', lw=1.2, zorder=1)
box(ax, 1, YM - 3, 8.5, 6, r'$\Phi_0$', fc='#FFF6EE', fs=15)
arr(ax, (XS[0] - 4.5, YM), (9.5, YM))
ax.text(1, 0.4, 'E：shared encoder（stride-2 conv，兩路權重共享）　W：warp　'
        + r'$\mathcal{U}$' + '：×2 upsampling（' + r'$\Phi$' + '、' + r'$h$' + '）　'
        + r'$D_l$' + '：式 (4)', ha='left', va='bottom', fontsize=10.5, color=MUTED)
save(fig, '1014_arch_pyramid.png')


# ── Step 2 與 VoxelMorph U-Net 之比較（2026-10-07 使用者：「最後一頁不好想像，和 U-Net 本身架構有點搞混」→ 放進簡報）──
#   左：VoxelMorph 之 U-Net（直式 U，同 Ronneberger 之畫法）；右：上面 1014_arch_pyramid.png 旋轉 90°
#   （上排 m → 左臂、下排 f → 右臂、中排 decoder → 中央），標出三處不同。結構照 ASD/arch.py 的 VxmPyramid
_UY = [5.0, 3.85, 2.7, 1.55, 0.4]                     # l = 0（原解析度）… 4（1/16），由上往下
_UBW, _UBH = 1.1, 0.64
_ENC, _DEC, _OUT, _RED, _GRN, _GRY = '#F1EFE8', '#E6F2F1', '#FDF3EA', '#B03A2E', '#2E8B57', '#A8A8A8'


def _ubox(ax, x, y, t, fc, w=_UBW, fs=13, sub=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - _UBH / 2), w, _UBH, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=INK, lw=1.2))
    ax.text(x, y + (0.1 if sub else 0), t, ha='center', va='center', fontsize=fs, color=INK)
    if sub:
        ax.text(x, y - 0.18, sub, ha='center', va='center', fontsize=9, color=TEAL)


def _uarr(ax, p, q, c, lw=1.5):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, shrinkA=0, shrinkB=0,
                                                    mutation_scale=12))


def _unum(ax, x, y, k):
    ax.add_patch(Circle((x, y), 0.19, fc=_RED, ec='none', zorder=5))
    ax.text(x, y, str(k), ha='center', va='center', fontsize=10, color='white', fontweight='bold', zorder=6)


with plt.rc_context({'font.family': ['Microsoft JhengHei', 'DejaVu Sans']}):   # ×、→ 等缺字用 DejaVu Sans 補
    fig, axes = plt.subplots(1, 2, figsize=(W, 5.0), facecolor=PAPER)
    for ax in axes:
        ax.set_xlim(-1.0, 10.3)
        ax.set_ylim(-1.55, 6.75)
        ax.set_aspect('equal')
        ax.axis('off')
        for l, y in enumerate(_UY):
            ax.text(-0.95, y, '1/%d' % 2 ** l, ha='left', va='center', fontsize=10, color=MUTED)

    # (a) VoxelMorph 之 U-Net：單一 encoder（m、f 於輸入端串接），僅於最高解析度輸出一次
    ax = axes[0]
    EX = [1.0 + 0.6 * l for l in range(5)]
    DX = [8.6 - 0.6 * l for l in range(5)]
    _ubox(ax, EX[0], _UY[0], r'$[m,\ f]$', _ENC, w=1.3, fs=13.5)
    ax.text(EX[0], _UY[0] + 0.5, '串接為 2 通道', ha='center', fontsize=9.5, color=MUTED)
    for l in range(1, 5):
        _ubox(ax, EX[l], _UY[l], r'$E_%d$' % l, _ENC)
        _uarr(ax, (EX[l - 1], _UY[l - 1] - _UBH / 2), (EX[l], _UY[l] + _UBH / 2), _RED)
    for l in range(5):
        _ubox(ax, DX[l], _UY[l], r'$D_%d$' % l, _DEC)
        if l < 4:
            _uarr(ax, (EX[l] + (0.65 if l == 0 else _UBW / 2), _UY[l]), (DX[l] - _UBW / 2, _UY[l]), _GRY, lw=2)
            _uarr(ax, (DX[l + 1], _UY[l + 1] + _UBH / 2), (DX[l], _UY[l] - _UBH / 2), _GRN)
    _uarr(ax, (EX[4] + _UBW / 2, _UY[4]), (DX[4] - _UBW / 2, _UY[4]), INK)
    _ubox(ax, DX[0], 6.25, r'$v\ \to\ \phi=\exp(v)$', _OUT, w=2.8, fs=12.5)
    _uarr(ax, (DX[0], _UY[0] + _UBH / 2), (DX[0], 6.25 - _UBH / 2), INK)
    ax.text(DX[0] - 1.55, 6.25, '僅輸出一次', ha='right', va='center', fontsize=10, color=INK)
    ax.set_title('(a) VoxelMorph U-Net（現行模型之 ' + r'$g_{\theta}$' + '）', fontsize=12.5, fontweight='bold',
                 color=INK, pad=2)
    ax.text(-0.95, -0.6, '紅：stride-2 conv　綠：×2 upsampling　灰：skip connection', fontsize=10, color=MUTED)
    ax.text(-0.95, -1.1, '單一 encoder（m、f 於輸入端串接），於最高解析度輸出一次', fontsize=10, color=MUTED)

    # (b) Step 2：dual-stream encoder、m 之 skip 先 warp、每層 decoder 皆輸出形變
    ax = axes[1]
    MX = [0.9 + 0.55 * l for l in range(5)]
    FX = [8.7 - 0.55 * l for l in range(5)]
    CX, WX = 4.8, 3.65
    for l in range(5):
        _ubox(ax, MX[l], _UY[l], r'$m$' if l == 0 else r'$F_m^{%d}$' % l, _ENC)
        _ubox(ax, FX[l], _UY[l], r'$f$' if l == 0 else r'$F_f^{%d}$' % l, _ENC)
        _ubox(ax, CX, _UY[l], r'$D_%d$' % l, _DEC, sub=r'$\to\ \Phi_%d$' % l)
        if l:
            _uarr(ax, (MX[l - 1], _UY[l - 1] - _UBH / 2), (MX[l], _UY[l] + _UBH / 2), _RED)
            _uarr(ax, (FX[l - 1], _UY[l - 1] - _UBH / 2), (FX[l], _UY[l] + _UBH / 2), _RED)
        _uarr(ax, (FX[l] - _UBW / 2, _UY[l]), (CX + _UBW / 2, _UY[l]), _GRY, lw=2)
        if l < 4:                                         # m 之特徵先以下一層之形變 warp（最粗層 Φ5 = Id，直接進）
            ax.plot([MX[l] + _UBW / 2, WX - 0.2], [_UY[l], _UY[l]], color=_GRY, lw=2)
            ax.add_patch(Circle((WX, _UY[l]), 0.2, fc='white', ec=INK, lw=1.2, zorder=4))
            ax.text(WX, _UY[l], 'W', ha='center', va='center', fontsize=9.5, zorder=5)
            _uarr(ax, (WX + 0.2, _UY[l]), (CX - _UBW / 2, _UY[l]), _GRY, lw=2)
            _uarr(ax, (CX, _UY[l + 1] + _UBH / 2), (CX, _UY[l] - _UBH / 2), _GRN)
            _uarr(ax, (CX - _UBW / 2 + 0.05, _UY[l + 1] + _UBH / 2), (WX + 0.05, _UY[l] - 0.2), TEAL, lw=1.4)
        else:
            _uarr(ax, (MX[l] + _UBW / 2, _UY[l]), (CX - _UBW / 2, _UY[l]), _GRY, lw=2)
    _ubox(ax, CX, 6.25, r'$\Phi_0$', _OUT, fs=13.5)
    _uarr(ax, (CX, _UY[0] + _UBH / 2), (CX, 6.25 - _UBH / 2), INK)
    _unum(ax, MX[0] - 0.2, _UY[0] + 0.62, 1)
    _unum(ax, FX[0] + 0.2, _UY[0] + 0.62, 1)
    _unum(ax, WX, _UY[0] + 0.45, 2)
    _unum(ax, CX + 0.85, _UY[1] + 0.5, 3)
    ax.text(MX[0], _UY[0] + 1.05, 'm 之 encoder', ha='center', fontsize=10, color=INK)
    ax.text(FX[0], _UY[0] + 1.05, 'f 之 encoder', ha='center', fontsize=10, color=INK)
    ax.set_title('(b) Step 2：Coarse-to-fine', fontsize=12.5, fontweight='bold', color=INK, pad=2)
    ax.text(-0.95, -0.5, '① Dual-stream encoder：m、f 分別抽取特徵（權重共享）', fontsize=10, color=INK)
    ax.text(-0.95, -0.95, '② W：以下一層之形變（青色，×2 upsampling）warp m 之特徵', fontsize=10, color=INK)
    ax.text(-0.95, -1.4, '③ 每層 decoder 皆輸出形變，逐層累積至 ' + r'$\Phi_0$', fontsize=10, color=INK)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, wspace=0.03)
    save(fig, '1014_arch_unet_vs_pyramid.png')


# ── 公式區塊（同 make_method.py：公式置中、編號靠右，下面是「其中」）──────────────────
def eq_block(name, eqs, lines, h, gap=0.31, fs=21):
    fig = plt.figure(figsize=(W, h), facecolor=PAPER)
    y = 0.36
    for eq, num in eqs:
        fig.text(0.5, 1 - y / h, eq, ha='center', va='center', fontsize=fs, color=INK)
        fig.text(0.985, 1 - y / h, r'$(%d)$' % num, ha='right', va='center', fontsize=17, color=INK)
        y += 0.66
    y -= 0.66 - 0.48
    for i, ln in enumerate(lines):
        fig.text(0.02, 1 - (y + i * gap) / h, ln, ha='left', va='top', fontsize=13, color=INK)
    save(fig, name)


IN, SP = '其中　', '　　　'
eq_block('1014_arch_step0_eq.png',
         [(r'$\Phi^{(1)}=\exp\left(g_{\theta}(m,\,f)\right),\qquad'
           r'\Phi^{(2)}=\Phi^{(1)}\circ\exp\left(g_{\theta}(m\circ\Phi^{(1)},\,f)\right)$', 1)],
         [IN + r'$g_{\theta}$' + '：已訓練之 mix_exp6，兩次使用相同參數、不重新訓練；'
          + r'$(m\circ\phi)(x)=m(\phi(x))$' + '，' + r'$\phi=\mathrm{Id}+u$',
          SP + '位移表示：' + r'$U^{(2)}(x)=u_2(x)+U^{(1)}(x+u_2(x))$']
         , 1.5, fs=20)
eq_block('1014_arch_cascade_eq.png',
         [(r'$v_k=g_{\theta_k}(m\circ\Phi_{k-1},\,f),\quad\phi_k=\exp(v_k),\quad'
           r'\Phi_k=\Phi_{k-1}\circ\phi_k\quad(k=1,2;\ \Phi_0=\mathrm{Id})$', 2),
          (r'$\mathcal{L}=-\mathrm{NCC}(f,\,m\circ\Phi_2)+\lambda\sum_{k=1}^{2}\Vert\nabla v_k\Vert^2$', 3)],
         [IN + r'$m$' + '：moving image（affine 後）；' + r'$f$' + '：fixed image（MNI152）；'
          + r'$g_{\theta_1},\,g_{\theta_2}$' + '：VoxelMorph U-Net（參數不共享，end-to-end 訓練）',
          SP + r'$\exp$' + '：scaling and squaring（7 steps）；' + r'$\lambda=1$' + '；similarity 僅計算最終輸出，'
          + 'smoothness 計算於各 stage 之 ' + r'$v_k$',
          SP + 'Step 0 即 ' + r'$\theta_1=\theta_2$' + ' 且不重新訓練之特例。參考：Zhao et al., ICCV 2019（RCN）'],
         2.4, fs=20)
eq_block('1014_arch_pyramid_eq.png',
         [(r'$v_l=g_l\left(\left[\,F_m^{\,l}\circ\tilde{\Phi}_{l+1},\ F_f^{\,l},\ \mathcal{U}(h_{l+1})\,\right]\right),\quad'
           r'\phi_l=\exp(v_l),\quad\Phi_l=\tilde{\Phi}_{l+1}\circ\phi_l$', 4),
          (r'$\mathcal{L}=-\mathrm{NCC}(f,\,m\circ\Phi_0)+\lambda\sum_{l=0}^{4}\Vert\nabla v_l\Vert^2$', 5)],
         [IN + r'$F_m^{\,l}=E_l(m)$' + '、' + r'$F_f^{\,l}=E_l(f)$' + '：shared encoder（dual-stream）第 ' + r'$l$'
          + ' 層特徵（' + r'$F^{\,0}$' + ' 為影像本身）；' + r'$[\cdot]$' + '：channel concatenation',
          SP + r'$\mathcal{U}$' + '：×2 upsampling（' + r'$\Phi$' + '：trilinear，位移向量 ×2；' + r'$h$' + '：nearest）；'
          + r'$\tilde{\Phi}_{l+1}=\mathcal{U}(\Phi_{l+1})$' + '，' + r'$\Phi_5=\mathrm{Id}$' + '；'
          + r'$h_{l+1}$' + '：上一層之 decoder feature',
          SP + '通道數同 VoxelMorph；參數 0.41 M。參考：Hu et al., MICCAI 2019（Dual-PRNet）；Jian et al., WBIR 2024（DWP）'],
         2.45, fs=20)
