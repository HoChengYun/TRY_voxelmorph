# -*- coding: utf-8 -*-
"""LUMIR 2024（腦部 MRI 對位比賽）測試集成績：長條＝Dice，顏色＝擠爆比例（NDV）。

數字照抄 LUMIR 官方 GitHub 的「Test phase results」表（2026-10-06 抓）：
    https://github.com/JHU-MedImage-Reg/LUMIR_L2R
總排名是官方用 TRE、Dice、HdDist95、NDV 四項綜合算的 Score（README 沒寫公式），所以不完全照 Dice 排。
⚠️ 標籤、資料都跟我們不同，我們的 Dice 不能放進來比。

輸出到同資料夾的 lumir_2024.png。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
INK, MUTED, PAPER = '#141A1D', '#5F6A6B', '#FAFAF8'
TEAL, AMBER, RED = '#0E7C7B', '#D08A1E', '#D62728'

# (隊伍／方法, 測試時是否逐案再最佳化（ISO）, Dice, NDV %, 總排名)
ROWS = [
    ('honkamj', False, 0.7851, 0.0025, 1),
    ('hnuzyx_next-gen-nn', False, 0.7773, 0.0001, 2),
    ('lieweaver', False, 0.7779, 0.0121, 3),
    ('zhuoyuanw210', False, 0.7726, 0.0045, 4),
    ('LYU-zhouhu', False, 0.7776, 0.0150, 5),
    ('Tsubasa025', True, 0.7701, 0.0030, 6),
    ('uniGradICON (w/ ISO 50)', True, 0.7596, 0.0002, 7),
    ('VFA', False, 0.7767, 0.0704, 8),
    ('lukasf', False, 0.7639, 0.2761, 9),
    ('Bailiang', False, 0.7735, 0.0222, 10),
    ('TransMorph', False, 0.7624, 0.3621, 11),
    ('TimH', False, 0.7303, 0.0000, 12),
    ('deedsBCV', True, 0.6958, 0.0002, 13),
    ('uniGradICON', False, 0.7422, 0.0001, 14),
    ('kimjin2510', True, 0.7355, 0.0033, 15),
    ('HongyuLyu', False, 0.7596, 1.1646, 16),
    ('SynthMorph', False, 0.7216, 0.0000, 17),
    ('TS_UKE', True, 0.7603, 0.0475, 18),
    ('ANTsSyN', True, 0.7025, 0.0000, 19),
    ('VoxelMorph', False, 0.7144, 1.2167, 20),
    ('ZeroDisplacement', False, 0.5549, 0.0000, 20),
]
# 有論文名稱的方法換成好讀的標籤；其他參賽隊伍維持隊名、淡色
NAMED = {
    'honkamj': 'SITReg（第 1 名）',
    'VFA': 'VFA',
    'TransMorph': 'TransMorph',
    'uniGradICON': 'uniGradICON（直接用）',
    'uniGradICON (w/ ISO 50)': 'uniGradICON＋逐案微調',
    'SynthMorph': 'SynthMorph',
    'VoxelMorph': 'VoxelMorph（我們用的）',
    'ANTsSyN': 'ANTs SyN（傳統方法）',
    'deedsBCV': 'deedsBCV（傳統方法）',
    'ZeroDisplacement': '完全不變形（起點）',
}
X0 = 0.50   # x 軸起點


def color(ndv):
    if ndv < 0.01:
        return TEAL
    if ndv < 0.1:
        return AMBER
    return RED


def fmt(ndv):
    s = ('%.4f' % ndv).rstrip('0').rstrip('.')
    return s if s not in ('', '0') else '0'


rows = sorted(ROWS, key=lambda r: -r[2])
n = len(rows)
fig, ax = plt.subplots(figsize=(11.5, 9.2))
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.13)

for i, (name, iso, dice, ndv, rank) in enumerate(rows):
    y = n - 1 - i
    named = name in NAMED
    ax.barh(y, dice - X0, left=X0, height=0.7, color=color(ndv), alpha=1.0 if named else 0.38)
    label = NAMED.get(name, name) + (' †' if iso else '')
    ax.text(X0 - 0.004, y, label, ha='right', va='center', clip_on=False,
            fontsize=11.5 if named else 10, color=INK if named else MUTED,
            fontweight='bold' if named else 'normal')
    ax.text(dice + 0.003, y, '%.3f   擠爆 %s%%   總排名 %d' % (dice, fmt(ndv), rank), va='center',
            fontsize=10 if named else 9.5, color=INK if named else MUTED)

ax.set_xlim(X0, 0.875)
ax.set_ylim(-0.7, n - 0.3)
ax.set_yticks([])
ax.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
ax.tick_params(axis='x', colors=MUTED, labelsize=10)
ax.set_xlabel('Dice（越高越好；x 軸從 0.50 開始）', fontsize=11, color=MUTED)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(MUTED)

fig.text(0.03, 0.955, 'LUMIR 2024 測試集：長條＝Dice，顏色＝擠爆比例（NDV）', fontsize=16,
         fontweight='bold', color=INK)
fig.text(0.03, 0.915, '4,014 顆 T1 訓練、590 位測試。總排名是官方綜合 TRE、Dice、HD95、擠爆四項算的，'
         '所以不完全照 Dice 排（VFA 的 Dice 第 5，但擠爆較多，總排名第 8）', fontsize=10.5, color=MUTED)

handles = [Patch(color=TEAL, label='擠爆 < 0.01%'), Patch(color=AMBER, label='0.01～0.1%'),
           Patch(color=RED, label='> 0.1%')]
fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.03, 0.015), ncol=3, frameon=False,
           fontsize=10.5)
fig.text(0.45, 0.04, '† 測試時對每一位再最佳化（比較慢）　淡色＝沒有論文名稱的參賽隊伍\n'
         '資料：github.com/JHU-MedImage-Reg/LUMIR_L2R（Test phase results，2026-10-06）',
         fontsize=9.5, color=MUTED, va='center')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lumir_2024.png')
fig.savefig(out, dpi=115, facecolor=PAPER)
print('寫出', out)
