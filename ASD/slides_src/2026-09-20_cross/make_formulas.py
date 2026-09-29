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
import numpy as np
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
ax.text(.5, .555, r'每個點取 9×9×9 的小窗格比對，越像越接近 $-1$',
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
# 2026-09-29：補上 Id / exp 的白話，以及速度場版的 --int-downsize 2（在一半解析度上積分）。
# exp2 -> exp4 的「換版本」其實連解析度一起換了，這張圖要讓人看得出來。
fig, ax = canvas(11, 4.3)
ax.text(.25, .93, '位移場版', ha='center', fontsize=18, color=RUST, fontweight='bold')
ax.text(.25, .855, '（論文 Table I 用的）', ha='center', fontsize=12, color=MUTED)
ax.text(.25, .70, r'$\phi \;=\; \mathrm{Id} + u$', ha='center', va='center', fontsize=30, color=INK)
ax.text(.25, .565, 'Id＝每個點原地不動，u＝往哪移多少', ha='center', fontsize=12.5, color=RUST)
ax.text(.25, .445, '網路直接說「這個點搬到那裡」', ha='center', fontsize=13.5, color=MUTED)
ax.text(.25, .345, '自由，但沒有任何保證', ha='center', fontsize=13.5, color=MUTED)
ax.text(.25, .18, '--int-steps 0   --int-downsize 1', ha='center', fontsize=12.5, color=INK, family='monospace')
ax.text(.25, .075, '全尺寸，不積分', ha='center', fontsize=12.5, color=MUTED)

ax.plot([.5, .5], [.04, .96], color='#D9D9D2', lw=1)

ax.text(.75, .93, '速度場版', ha='center', fontsize=18, color=TEAL, fontweight='bold')
ax.text(.75, .855, '（程式的預設）', ha='center', fontsize=12, color=MUTED)
ax.text(.75, .70, r'$\phi \;=\; \exp(v)$', ha='center', va='center', fontsize=30, color=INK)
ax.text(.75, .565, 'exp＝沿著速度一小步一小步累加（積分）', ha='center', fontsize=12.5, color=TEAL)
ax.text(.75, .445, '網路給速度，再沿平滑路線積分 128 步', ha='center', fontsize=13.5, color=MUTED)
ax.text(.75, .345, '連續移動，理論上不會把組織擠穿', ha='center', fontsize=13.5, color=MUTED)
ax.text(.75, .18, '--int-steps 7   --int-downsize 2', ha='center', fontsize=12.5, color=INK, family='monospace')
ax.text(.75, .075, '先縮小一半再積分，積完放大回原尺寸', ha='center', fontsize=12.5, color=MUTED)
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

# ── 5. 速度場版怎麼算（縮放再平方）────────────────────────────────────
# 右邊的驗算是真的照左邊的公式算出來的（1 維、v(x) = -2x、從 x = 10 出發），不是手打。
T = 7
v1 = lambda x: -2.0 * x
X0 = 10.0
u = lambda x: v1(x) / 2 ** T                        # u0 = v / 2^T
trace = [(1, X0 + u(X0))]
for k in range(T):                                  # u_{k+1}(p) = u_k(p) + u_k(p + u_k(p))
    u = (lambda g: (lambda x: g(x) + g(x + g(x))))(u)
    trace.append((2 ** (k + 1), X0 + u(X0)))

fig, ax = canvas(12, 5.2)
ax.plot([.575, .575], [.04, .96], color='#D9D9D2', lw=1)

ax.text(.03, .92, '① 一小步：速度除以 128', fontsize=15, color=TEAL, fontweight='bold')
ax.text(.28, .80, r'$u_0(p) \;=\; \frac{v(p)}{2^T}$', ha='center', va='center', fontsize=25, color=INK)
ax.text(.03, .64, '② 自己接自己，做 T 次（每次步數翻倍）', fontsize=15, color=TEAL, fontweight='bold')
ax.text(.28, .52, r'$u_{k+1}(p) \;=\; u_k(p) \;+\; u_k(\,p + u_k(p)\,)$',
        ha='center', va='center', fontsize=23, color=INK)
ax.text(.28, .415, '先照舊的走一段，再看「走到的地方」的箭頭走一段', ha='center', fontsize=12.5, color=MUTED)
ax.text(.03, .29, '③ 最後的形變', fontsize=15, color=TEAL, fontweight='bold')
ax.text(.28, .18, r'$\phi(p) \;=\; p + u_T(p)$', ha='center', va='center', fontsize=25, color=INK)
# ⚠️ 減號「−」和上標「⁷」微軟正黑體沒有，會變方框 → 這幾段改用 mathtext（$...$）
ax.text(.28, .06, r'$T = 7$ → 走了 $2^7 = 128$ 小步，但只算 7 次', ha='center', fontsize=12.5, color=MUTED)

ax.text(.79, .92, r'驗算：$v(x) = -2x$，從 $x = 10$ 出發', ha='center', fontsize=15, fontweight='bold', color=INK)
ax.text(.79, .815, r'位移場版：$10 + (-20) = -10$　→ 跑到另一邊', ha='center', fontsize=13.5,
        color=RUST, fontweight='bold')
ax.text(.66, .70, '速度場版', fontsize=13.5, color=TEAL, fontweight='bold')
ax.text(.705, .625, '走了幾步', ha='center', fontsize=12, color=MUTED)
ax.text(.875, .625, '停在', ha='center', fontsize=12, color=MUTED)
for i, (n, pos) in enumerate(trace):
    y = .555 - i * .062
    last = (i == len(trace) - 1)
    ax.text(.705, y, '%d' % n, ha='center', fontsize=13, color=INK, family='monospace',
            fontweight='bold' if last else 'normal')
    ax.text(.875, y, '%.2f' % pos, ha='center', fontsize=13, color=TEAL if last else INK,
            family='monospace', fontweight='bold' if last else 'normal')
ax.text(.79, .02, '越靠近中間走越慢，停在 %.2f，還在同一邊' % trace[-1][1], ha='center', fontsize=12.5, color=TEAL)
save(fig, 'formula_integrate.png')
print('   驗算：', ', '.join('%d步 %.3f' % t for t in trace))

# ── 6. --int-downsize 2：縮小一半 → 積分 → 放大 ─────────────────────────
# 上排：同一塊區域，原本的格子（1 格 = 1 mm）和小地圖（1 格 = 2 mm）並排，箭頭實際長度一樣。
# 下排：一條線上的箭頭長度，縮小再放大之後，平滑的部分不變、1 mm 寬的小彎曲不見了。
from matplotlib.patches import FancyArrowPatch


def mini_grid(ax, n, arrows, title, sub, color):
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    step = 8 / n
    for i in range(n + 1):
        ax.plot([i * step, i * step], [0, 8], color='#C9C9C1', lw=0.8)
        ax.plot([0, 8], [i * step, i * step], color='#C9C9C1', lw=0.8)
    c = (np.arange(n) + 0.5) * step
    gx, gy = np.meshgrid(c, c)
    ux, uy = arrows(gx, gy)
    ax.quiver(gx, gy, ux, uy, angles='xy', scale_units='xy', scale=1, color=color,
              width=0.018 if n <= 4 else 0.012, headwidth=4)
    for sp in ax.spines.values():
        sp.set_color('#9AA3A4')
    ax.set_title(title, fontsize=14, fontweight='bold', color=INK, pad=6)
    ax.text(0.5, -0.09, sub, transform=ax.transAxes, ha='center', va='top', fontsize=11.5, color=MUTED)


pull = lambda x, y: (-0.26 * (x - 4), -0.26 * (y - 4))       # 往中間拉（實際長度，mm）
after = lambda x, y: (-0.2 * (x - 4), -0.2 * (y - 4))        # 積分後的位移（示意）

fig = plt.figure(figsize=(12.5, 6.0), facecolor=PAPER)
panels = [(0.035, 8, pull, '原本的地圖', '1 格 = 1 mm', TEAL),
          (0.285, 4, pull, '小地圖', '1 格 = 2 mm，格子少 8 倍', TEAL),
          (0.535, 4, after, '在小地圖上積分', '那 7 次全部在這裡做', TEAL),
          (0.785, 8, after, '放大回原尺寸', '每個點的箭頭用內插補回來', RUST)]
for x0, n, fn, t, sub, col in panels:
    mini_grid(fig.add_axes([x0, 0.50, 0.18, 0.40]), n, fn, t, sub, col)

top = fig.add_axes([0, 0.45, 1, 0.55])
top.set_xlim(0, 1)
top.set_ylim(0, 1)
top.axis('off')
for x, lab in ((0.25, '① 縮小\n箭頭 × 0.5'), (0.50, '② 積分\n7 次'), (0.75, '③ 放大\n箭頭 × 2')):
    top.add_patch(FancyArrowPatch((x - 0.025, 0.55), (x + 0.025, 0.55), arrowstyle='-|>',
                                  mutation_scale=18, color=INK, lw=1.8))
    top.text(x, 0.70, lab, ha='center', va='bottom', fontsize=12.5, color=INK, fontweight='bold')
top.text(0.5, 0.965, '箭頭 × 0.5、× 2 只是換單位（格數）：同樣 4 mm，原本是 4 格，小地圖上是 2 格，距離沒變',
         ha='center', va='top', fontsize=12.5, color=MUTED)

# 下排：一條線上的箭頭，縮小再放大
ax = fig.add_axes([0.07, 0.08, 0.86, 0.28])
xs = np.arange(0, 41)                                         # 1 mm 一格
prof = 3 * np.sin(xs / 40 * 2 * np.pi)                        # 平滑的大方向
prof = prof + np.where(xs == 21, 2.2, 0.0)                    # 一個 1 mm 寬的小彎曲
coarse_x = xs[::2]                                            # 小地圖只留偶數格
back = np.interp(xs, coarse_x, prof[::2])                     # 放大回來（線性內插）
ax.plot(xs, prof, 'o-', color=INK, lw=1.4, ms=3.5, label='原本（1 mm 一格）')
ax.plot(xs, back, '-', color=RUST, lw=3, alpha=.85, label='縮小再放大之後')
ax.annotate('這個 1 mm 寬的小彎曲不見了', (21, prof[21]), (26, prof[21] + 0.4),
            fontsize=12.5, color=RUST, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RUST, lw=1.6))
ax.set_xlabel('位置（mm）', fontsize=11.5)
ax.set_ylabel('箭頭長度', fontsize=11.5)
ax.legend(loc='lower left', fontsize=11, frameon=False)
ax.set_title('會損失什麼：平滑的地方幾乎沒差，2 mm 以內的細節被抹平', fontsize=14, fontweight='bold', color=INK)
ax.grid(alpha=.3)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.set_facecolor(PAPER)
save(fig, 'formula_downsize.png')
