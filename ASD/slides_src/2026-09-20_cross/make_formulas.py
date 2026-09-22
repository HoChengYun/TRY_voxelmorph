# -*- coding: utf-8 -*-
"""把簡報要用的公式畫成圖（matplotlib 的數學排版，不是純文字）。

輸出到 models/deck_charts/：
    formula_loss.png       總損失 = 相似度 + λ·c·平滑度，兩項各自標中文說明
    formula_terms.png      兩項各自的定義（NCC 與位移場梯度）
    formula_versions.png   位移場版 vs 速度場版的參數化
    formula_metrics.png    Dice 與折疊率（Jacobian 行列式）

中文用微軟正黑體，公式用 matplotlib 的 mathtext。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'models', 'deck_charts')
os.makedirs(OUT, exist_ok=True)

INK, MUTED, TEAL, RUST, PAPER = '#141A1D', '#5F6A6B', '#0E7C7B', '#A34F1B', '#FAFAF8'


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor=PAPER)
    plt.close(fig)
    print('->', p)


# ── 1. 總損失 ─────────────────────────────────────────────────────────
fig, ax = canvas(11, 3.6)
ax.text(.5, .74, r'$\mathcal{L}\;=\;\mathcal{L}_{\mathrm{sim}}(A,\;M\circ\phi)'
                 r'\;+\;\lambda\cdot c\cdot\mathcal{L}_{\mathrm{smooth}}(u)$',
        ha='center', va='center', fontsize=31, color=INK)
# 兩項的底線
ax.plot([.205, .47], [.60, .60], color=TEAL, lw=3, solid_capstyle='round')
ax.plot([.525, .90], [.60, .60], color=RUST, lw=3, solid_capstyle='round')
ax.text(.337, .49, '對得像不像', ha='center', fontsize=17, color=TEAL, fontweight='bold')
ax.text(.337, .37, '捏過的腦跟模板有多接近', ha='center', fontsize=13, color=MUTED)
ax.text(.712, .49, '捏得平不平滑', ha='center', fontsize=17, color=RUST, fontweight='bold')
ax.text(.712, .37, '相鄰的點移動差太多就罰', ha='center', fontsize=13, color=MUTED)
ax.text(.5, .17, r'$A$ 模板　　$M$ 受試者的腦　　$\phi$ 形變場　　$u$ 位移　　'
                 r'$\lambda$ 你下的 --lambda　　$c$ 程式自己乘的 --int-downsize',
        ha='center', fontsize=13, color=MUTED)
ax.text(.5, .05, '模型要讓這個值越小越好：對得越像、又不要捏得太誇張',
        ha='center', fontsize=14, color=INK)
save(fig, 'formula_loss.png')

# ── 2. 兩項的定義 ────────────────────────────────────────────────────
fig, ax = canvas(11, 3.9)
ax.text(.03, .88, '① 對得像不像：局部相關係數（NCC）', fontsize=16, color=TEAL, fontweight='bold')
ax.text(.5, .70, r'$\mathcal{L}_{\mathrm{sim}} \;=\; -\,\frac{1}{|\Omega|}'
                 r'\sum_{p\,\in\,\Omega}\mathrm{CC}(A,\;M\circ\phi)(p)$',
        ha='center', va='center', fontsize=26, color=INK)
ax.text(.5, .555, '每個點取 9×9×9 的小窗格比對，越像越接近 −1',
        ha='center', fontsize=13.5, color=MUTED)
ax.plot([.03, .97], [.48, .48], color='#D9D9D2', lw=1)
ax.text(.03, .40, '② 捏得平不平滑：位移場的梯度平方', fontsize=16, color=RUST, fontweight='bold')
ax.text(.5, .22, r'$\mathcal{L}_{\mathrm{smooth}}(u) \;=\; \frac{1}{3}['
                 r'\overline{(\Delta_x u)^2}+\overline{(\Delta_y u)^2}+\overline{(\Delta_z u)^2}]$',
        ha='center', va='center', fontsize=26, color=INK)
ax.text(.5, .07, r'$\Delta_x u(p)=u(p+e_x)-u(p)$：相鄰兩點的位移差。差越大，這一項越大',
        ha='center', fontsize=13.5, color=MUTED)
save(fig, 'formula_terms.png')

# ── 3. 兩種版本 ──────────────────────────────────────────────────────
fig, ax = canvas(11, 3.4)
ax.text(.25, .90, '位移場版', ha='center', fontsize=18, color=RUST, fontweight='bold')
ax.text(.25, .80, '（論文 Table I 用的）', ha='center', fontsize=12, color=MUTED)
ax.text(.25, .60, r'$\phi \;=\; \mathrm{Id} + u$', ha='center', va='center', fontsize=30, color=INK)
ax.text(.25, .40, '網路直接說「這個點搬到那裡」', ha='center', fontsize=13.5, color=MUTED)
ax.text(.25, .27, '自由，但沒有任何保證', ha='center', fontsize=13.5, color=MUTED)
ax.text(.25, .11, '--int-steps 0', ha='center', fontsize=13, color=INK, family='monospace')

ax.plot([.5, .5], [.06, .95], color='#D9D9D2', lw=1)

ax.text(.75, .90, '速度場版', ha='center', fontsize=18, color=TEAL, fontweight='bold')
ax.text(.75, .80, '（程式的預設）', ha='center', fontsize=12, color=MUTED)
ax.text(.75, .60, r'$\phi \;=\; \exp(v)$', ha='center', va='center', fontsize=30, color=INK)
ax.text(.75, .40, '網路給速度，再沿平滑路線積分 128 步', ha='center', fontsize=13.5, color=MUTED)
ax.text(.75, .27, '連續移動，理論上不會把組織擠穿', ha='center', fontsize=13.5, color=MUTED)
ax.text(.75, .11, '--int-steps 7', ha='center', fontsize=13, color=INK, family='monospace')
save(fig, 'formula_versions.png')

# ── 4. 兩個評估指標 ──────────────────────────────────────────────────
fig, ax = canvas(11, 3.6)
ax.text(.03, .90, '① 對得多準：Dice（30 個結構的平均）', fontsize=16, color=TEAL, fontweight='bold')
ax.text(.5, .70, r'$\mathrm{Dice}(A,B) \;=\; \frac{2\,|A \cap B|}{|A| + |B|}$',
        ha='center', va='center', fontsize=27, color=INK)
ax.text(.5, .555, '兩個結構完全重合是 1，完全不重疊是 0',
        ha='center', fontsize=13.5, color=MUTED)
ax.plot([.03, .97], [.49, .49], color='#D9D9D2', lw=1)
ax.text(.03, .41, '② 有沒有擠爆：形變場的 Jacobian 行列式', fontsize=16, color=RUST, fontweight='bold')
ax.text(.5, .24, r'$J_\phi(p) = \frac{\partial \phi}{\partial p}\;,\qquad$'
                 r'$\%\,|J_\phi|\leq 0 \;=\; \frac{\#\{\,p:\ \det J_\phi(p) \leq 0\,\}}{|\Omega|}$',
        ha='center', va='center', fontsize=23, color=INK)
ax.text(.5, .06, r'$\det J>1$ 體積被撐大　　$0<\det J<1$ 被壓小　　'
                 r'$\det J\leq 0$ 翻面或壓成零體積 ＝ 擠爆',
        ha='center', fontsize=13.5, color=MUTED)
save(fig, 'formula_metrics.png')
