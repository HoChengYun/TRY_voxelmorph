"""
VoxelMorph 批次測試腳本
========================
自動對 models/ 資料夾裡所有 .pt 模型跑測試，
彙整成表格並畫 Dice vs Epoch 折線圖。

使用方式（在 claude_cheng 根目錄執行）：
    python voxelmorph-code/scripts/torch/batch_test.py
    python voxelmorph-code/scripts/torch/batch_test.py --model-dir models/ --gpu 0
    python voxelmorph-code/scripts/torch/batch_test.py --start 10 --end 100
"""

import os
import sys
import glob
import argparse
import subprocess
import re
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 路徑（相對於本檔案位置，方便跨電腦部署）
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))   # voxelmorph-code/scripts/torch/
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))  # claude_cheng/

TEST_SCRIPT = os.path.join(_HERE, 'test.py')  # 同資料夾的 test.py

# ============================================================
# 參數
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', default='models/',   help='模型資料夾')
parser.add_argument('--gpu',       default='-1',        help='GPU id，-1 用 CPU')
parser.add_argument('--start',     type=int, default=0, help='從第幾個 epoch 開始測（0 = 全部）')
parser.add_argument('--end',       type=int, default=0, help='測到第幾個 epoch（0 = 全部）')
parser.add_argument('--out-dir',   default='models/',   help='結果輸出資料夾')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ============================================================
# 找所有模型檔案，依 epoch 排序
# ============================================================
pt_files = sorted(glob.glob(os.path.join(args.model_dir, '*.pt')))

def get_epoch(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return -1

pt_files = [f for f in pt_files if get_epoch(f) > 0]
pt_files = sorted(pt_files, key=get_epoch)

if args.start > 0:
    pt_files = [f for f in pt_files if get_epoch(f) >= args.start]
if args.end > 0:
    pt_files = [f for f in pt_files if get_epoch(f) <= args.end]

if not pt_files:
    print('找不到符合條件的模型檔案！')
    exit(1)

print(f'找到 {len(pt_files)} 個模型，開始批次測試...')
print()

# ============================================================
# 逐一跑 test.py，解析輸出
# ============================================================
results = []

for pt_path in pt_files:
    epoch    = get_epoch(pt_path)
    csv_path = os.path.join(args.out_dir, f'dice_{epoch:04d}.csv')

    print(f'  Epoch {epoch:04d} : {pt_path}')

    cmd = [
        sys.executable, TEST_SCRIPT,
        '--model',   pt_path,
        '--gpu',     args.gpu,
        '--out-csv', csv_path,
    ]

    proc   = subprocess.run(cmd, capture_output=True)
    output = (proc.stdout or b'').decode(errors='replace') + \
             (proc.stderr  or b'').decode(errors='replace')

    # 從 CSV 讀 Dice（最穩健）
    mean_dice, std_dice = None, None
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        mean_row = next((r for r in rows if r[0] == 'MEAN'), None)
        std_row  = next((r for r in rows if r[0] == 'STD'),  None)
        if mean_row and std_row:
            mean_dice = float(np.mean([float(v) for v in mean_row[1:]]))
            std_dice  = float(np.mean([float(v) for v in std_row[1:]]))

    # 從 TABLE I 輸出抓負 Jacobian % 和推論時間
    neg_pct_mean, neg_pct_std, gpu_sec_mean = None, None, None
    table_match = re.search(
        r'VoxelMorph.*?([\d.]+)\s*\(([\d.]+)\)\s+([\d.]+)\s*\(([\d.]+)\)\s+([\d.]+)\s*\(([\d.]+)\)\s+([\d.]+)\s*\(([\d.]+)\)',
        output
    )
    if table_match:
        try:
            neg_pct_mean = float(table_match.group(7))
            neg_pct_std  = float(table_match.group(8))
            gpu_sec_mean = float(table_match.group(3))
        except Exception:
            pass

    if mean_dice is not None:
        print(f'           Dice = {mean_dice:.4f} ± {std_dice:.4f}', end='')
        if neg_pct_mean is not None:
            print(f'  |  Neg-Jac = {neg_pct_mean:.3f}%', end='')
        print()
    else:
        print('           ⚠ 解析失敗')
        print(output[-500:])

    results.append({
        'epoch':       epoch,
        'model':       pt_path,
        'mean_dice':   mean_dice,
        'std_dice':    std_dice,
        'neg_jac_pct': neg_pct_mean,
        'gpu_sec':     gpu_sec_mean,
        'csv':         csv_path,
    })

# ============================================================
# 彙總表格
# ============================================================
valid = [r for r in results if r['mean_dice'] is not None]

print()
print('=' * 70)
print('  批次測試結果總表')
print('=' * 70)
print(f'  {"Epoch":>6}  {"Dice (mean±std)":>18}  {"Neg-Jac%":>10}  {"sec":>6}')
print(f'  {"-"*6}  {"-"*18}  {"-"*10}  {"-"*6}')

best_epoch, best_dice = None, -1
for r in valid:
    dice_str = f'{r["mean_dice"]:.4f} ± {r["std_dice"]:.4f}' if r["std_dice"] else f'{r["mean_dice"]:.4f}'
    neg_str  = f'{r["neg_jac_pct"]:.3f}%' if r["neg_jac_pct"] is not None else '—'
    sec_str  = f'{r["gpu_sec"]:.3f}'       if r["gpu_sec"]     is not None else '—'
    star = ' ★' if r['mean_dice'] == max(v['mean_dice'] for v in valid) else ''
    print(f'  {r["epoch"]:>6}  {dice_str:>18}  {neg_str:>10}  {sec_str:>6}{star}')
    if r['mean_dice'] > best_dice:
        best_dice  = r['mean_dice']
        best_epoch = r['epoch']

print('=' * 70)
if best_epoch is not None:
    print(f'  最佳模型：Epoch {best_epoch:04d}  Dice = {best_dice:.4f}')
else:
    print('  ⚠ 沒有成功解析的結果，請確認 test.py 有正常執行')
print()

# ============================================================
# 存 summary CSV
# ============================================================
summary_csv = os.path.join(args.out_dir, 'batch_test_summary.csv')
with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'mean_dice', 'std_dice', 'neg_jac_pct', 'gpu_sec'])
    for r in valid:
        writer.writerow([r['epoch'], r['mean_dice'], r['std_dice'],
                         r['neg_jac_pct'], r['gpu_sec']])
print(f'  Summary CSV  → {summary_csv}')

# ============================================================
# 畫 Dice vs Epoch 折線圖
# ============================================================
if not valid:
    print('  沒有有效結果，略過折線圖')
    exit(0)

epochs     = [r['epoch']    for r in valid]
mean_dices = [r['mean_dice'] for r in valid]
std_dices  = [r['std_dice'] if r['std_dice'] else 0 for r in valid]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, mean_dices, 'o-', color='steelblue', linewidth=2,
        markersize=5, label='VoxelMorph (NCC)')
ax.fill_between(epochs,
                [m - s for m, s in zip(mean_dices, std_dices)],
                [m + s for m, s in zip(mean_dices, std_dices)],
                alpha=0.2, color='steelblue')

ax.axhline(y=0.584, color='gray',  linestyle=':',  linewidth=1.2, label='Affine only (0.584)')
ax.axhline(y=0.753, color='green', linestyle='--', linewidth=1.2, label='Paper VoxelMorph-CC (0.753)')

ax.scatter([best_epoch], [best_dice], color='red', zorder=5, s=80)
ax.annotate(f'Best: {best_dice:.4f}\n(epoch {best_epoch})',
            xy=(best_epoch, best_dice),
            xytext=(10, -20), textcoords='offset points',
            fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1))

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Mean Dice Score', fontsize=11)
ax.set_title('Dice Score vs Training Epoch\n(VoxelMorph, Scan-to-Atlas, NCC loss)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=max(0, min(mean_dices) - 0.05))

plt.tight_layout()
draw_dir  = os.path.normpath(os.path.join(_ROOT, 'draw-img'))
os.makedirs(draw_dir, exist_ok=True)
plot_path = os.path.join(draw_dir, 'dice_vs_epoch.png')
fig.savefig(plot_path, dpi=130, bbox_inches='tight')
plt.close(fig)

print(f'  Dice 折線圖  -> {plot_path}')
print()
print('完成！')
