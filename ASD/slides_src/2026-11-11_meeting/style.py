# -*- coding: utf-8 -*-
"""11/11 簡報共用：路徑、配色、字型、模型清單。各 make_*.py 都 import 這支。"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
J = lambda *p: os.path.join(ROOT, *p)     # noqa: E731
OUT = J('models', 'deck_charts')          # 圖一律輸出成 1111_*.png
for s in (sys.stdout, sys.stderr):
    s.reconfigure(encoding='utf-8', errors='replace')

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
INK, MUTED, PAPER, RULE = '#141A1D', '#5F6A6B', '#FAFAF8', '#D9D9D2'
plt.rcParams['figure.facecolor'] = PAPER         # 跟投影片底色相同，圖片不會多出一塊白框
plt.rcParams['axes.facecolor'] = PAPER
GRAY = '#9A9A93'
W = 12.13                                  # 投影片內容寬（英吋），同 10/28 簡報

# Part 2 的模型路線：VoxelMorph 原作設定 → 原作 cascade／coarse-to-fine → 修改版（訓練中）
# key：(顯示名稱, 副標, 顏色, models 子資料夾, test CSV, surface CSV, val 曲線 CSV, 選定的 checkpoint（步數）)
MODELS = {
    'vxm': ('VoxelMorph', 'mix_exp3：displacement、local NCC、λ = 1', '#4B5D73',
            'mix_exp3', 'dice_0240.csv', 'surface_0240.csv', 'dice_curve_val.csv', 24000),
    'rcn': ('RCN（原作）', 'Affine＋VTN × 3、global correlation', '#2F7FD0',
            'orig_rcn_vtn3', 'dice_023000.csv', 'surface_023000.csv', 'dice_curve_val.csv', 23000),
    'dwp': ('DWP（原作）', 'Dual-stream coarse-to-fine、local NCC', '#0E7C7B',
            'orig_dwp', 'dice_0032.csv', 'surface_0032.csv', 'dice_curve_val_all.csv', 32 * 418),
}
# 修改一（RCN＋local NCC，11-04 帶回）：不放進 MODELS（那幾張三模型的圖照舊），只在指標表、修改一的頁用
NCC = ('RCN＋local NCC', '僅將相似度改為 local NCC', '#B5487A',
       'rcn_vtn3_ncc', 'dice_012000.csv', 'surface_012000.csv', 'dice_curve_val.csv', 12000)
PARAMS = {'vxm': 301411, 'rcn': 99094929, 'dwp': 1886796, 'ncc': 99094929}     # state_dict 的浮點參數數（11-03 從 checkpoint 數；ncc 同 RCN，NCC 沒有參數）
FG_VOX = 1867705                                              # MNI152 atlas 非背景 voxel 數


def load_lut(path=r'D:\MyHome\MRI\FreeSurfer\base_v1\FreeSurferColorLUT.txt'):
    """FreeSurfer 官方標籤顏色：{編號: (r, g, b)}，0～1。"""
    lut = {}
    for line in open(path, encoding='utf-8', errors='ignore'):
        p = line.split()
        if len(p) >= 5 and p[0].isdigit():
            lut[int(p[0])] = (int(p[2]) / 255, int(p[3]) / 255, int(p[4]) / 255)
    return lut


def colorize(lab, lut):
    import numpy as np
    rgb = np.zeros(lab.shape + (3,))
    for v in np.unique(lab):
        if v:
            rgb[lab == v] = lut.get(int(v), (0.5, 0.5, 0.5))
    return rgb


def save(fig, name, dpi=200):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    print('[OK]', p)


def clean(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=RULE, lw=0.6)
        ax.set_axisbelow(True)


# ── 數字欄之底色（11-11，同 p23 熱圖）：同一欄內比較，較佳 = 深、較差 = 淺；粗體 = 該欄最佳 ──
HEAT_LO, HEAT_HI = '#EEF3F8', '#2F5F9E'


def heat_t(vals, better, log=False):
    """vals：數字串列；better：'high'（越大越好）或 'low'；log：差距跨好幾個數量級時（folding）先取 log10。回傳 0～1，1 = 該欄最佳。"""
    import numpy as np
    v = np.log10(np.asarray(vals, float) + 0.1) if log else np.asarray(vals, float)
    lo, hi = v.min(), v.max()
    t = (v - lo) / (hi - lo) if hi > lo else np.full(len(v), 0.5)
    return list(t if better == 'high' else 1 - t)


def heat_cell(fig, x, y, text, t, w=0.062, h=0.075, fontsize=10.5):
    """在 figure 座標 (x, y) 畫一格有底色的數字。"""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch
    cmap = LinearSegmentedColormap.from_list('heat', [HEAT_LO, HEAT_HI])
    fig.patches.append(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.006',
                                      transform=fig.transFigure, fc=cmap(t), ec='none', zorder=0))
    fig.text(x, y, text, ha='center', va='center', fontsize=fontsize, color='white' if t > 0.6 else INK,
             fontweight='bold' if t > 0.999 else 'normal')
