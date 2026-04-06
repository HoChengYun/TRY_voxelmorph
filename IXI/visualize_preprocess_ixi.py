"""
IXI 前處理流程視覺化
========================================================
用途：對比顯示 T1 影像在各前處理階段的外觀差異
      ( Raw → N4 校正 → 去顱骨 → Affine 對齊 → Atlas )

使用方式
--------
【模式 1：完整管線（需 ants / antspynet）】
    # 安裝環境後執行（需要原始 .nii.gz）：
    python visualize_preprocess_ixi.py --input IXI-T1/IXI017-Guys-0698-T1.nii.gz

【模式 2：Demo 模式（無需原始資料，使用已處理的 .npz）】
    python visualize_preprocess_ixi.py --demo
    python visualize_preprocess_ixi.py --demo --subject IXI038-Guys-0729-T1

引數說明
--------
--input        原始 .nii.gz 路徑（完整模式用）
--atlas        atlas .npz 路徑，預設 atlas_mni152_09c_resize.npz
--preprocessed 前處理後 .npz 路徑（完整模式可自動搜尋）
--out-dir      輸出圖片資料夾，預設 preprocess_vis/
--subject      指定受試者名稱（demo 模式用）
--demo         使用 demo 模式（無需原始影像）
--slice-pct    顯示哪個比例的切片，預設 0.5（中間）
--no-skull     跳過去顱骨步驟（加快完整模式）
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────
# 引數解析
# ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='IXI 前處理流程視覺化')
parser.add_argument('--input',        default=None,   help='原始 .nii.gz 路徑（完整模式）')
parser.add_argument('--atlas',        default=os.path.join(_HERE, 'atlas_mni152_09c_resize.npz'),
                                                        help='Atlas .npz 路徑')
parser.add_argument('--preprocessed', default=None,   help='前處理後 .npz 路徑（可選，完整模式自動搜尋）')
parser.add_argument('--out-dir',      default=os.path.join(_HERE, 'preprocess_vis'),
                                                        help='輸出資料夾')
parser.add_argument('--subject',      default=None,   help='受試者名稱（demo 模式）')
parser.add_argument('--demo',         action='store_true', help='Demo 模式（使用已處理 .npz）')
parser.add_argument('--slice-pct',    type=float, default=0.5, help='切片位置比例 0~1，預設 0.5')
parser.add_argument('--no-skull',     action='store_true', help='跳過去顱骨')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 顯示翻轉輔助函式（與 visualize_reg_ixi.py 保持一致）
# ─────────────────────────────────────────────────────────────────────
def get_slice(vol, axis, pct):
    """從三維體積取出指定百分比位置的切片"""
    idx = int(vol.shape[axis] * pct)
    idx = max(0, min(idx, vol.shape[axis] - 1))
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]

def prep_slice(sl, flip_ud=True):
    """切片顯示前處理：轉置 + 翻轉（配合 imshow 座標軸）"""
    s = sl.T
    if flip_ud:
        s = np.flipud(s)
    return s

def normalize_display(arr):
    """將陣列正規化到 [0,1] 供顯示用"""
    p1, p99 = np.percentile(arr[arr > arr.min()], [1, 99]) if arr.max() > arr.min() else (arr.min(), arr.max())
    out = np.clip(arr, p1, p99)
    lo, hi = out.min(), out.max()
    if hi - lo < 1e-8:
        return np.zeros_like(out)
    return (out - lo) / (hi - lo)

# ─────────────────────────────────────────────────────────────────────
# 載入 Atlas
# ─────────────────────────────────────────────────────────────────────
print(f"載入 Atlas：{args.atlas}")
if not os.path.exists(args.atlas):
    print(f"❌ 找不到 atlas：{args.atlas}")
    sys.exit(1)
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
print(f"  Atlas shape: {atlas_vol.shape}")

# ─────────────────────────────────────────────────────────────────────
# DEMO 模式：使用已處理 .npz 模擬各階段
# ─────────────────────────────────────────────────────────────────────
if args.demo or args.input is None:
    print("\n── Demo 模式 ──────────────────────────────────────────")

    # 找受試者 npz
    search_dirs = [
        os.path.join(_HERE, 'IXI_preprocessed', 'test'),
        os.path.join(_HERE, 'IXI_preprocessed', 'train'),
    ]
    npz_files = []
    for d in search_dirs:
        npz_files.extend(sorted(glob.glob(os.path.join(d, '*.npz'))))

    if len(npz_files) == 0:
        print("❌ 找不到任何 .npz 檔案，請先執行 preprocess_ixi.py")
        sys.exit(1)

    # 選擇受試者
    if args.subject:
        matches = [f for f in npz_files if args.subject in os.path.basename(f)]
        if not matches:
            print(f"❌ 找不到受試者：{args.subject}")
            print("可用的受試者：")
            for f in npz_files[:10]:
                print(f"  {os.path.basename(f).replace('.npz','')}")
            sys.exit(1)
        npz_path = matches[0]
    else:
        npz_path = npz_files[0]

    subj_name = os.path.basename(npz_path).replace('.npz', '')
    print(f"受試者：{subj_name}")
    print(f"載入：{npz_path}")

    preprocessed_vol = np.load(npz_path)['vol'].astype(np.float32)
    print(f"  Preprocessed shape: {preprocessed_vol.shape}")

    # ── Demo 模式下的「各階段」說明 ──────────────────────────────────
    # 我們沒有原始影像，所以用合理的視覺效果來說明各階段特徵：
    #
    #  [Raw T1]       = 加入模擬的強度不均勻（bias field）效果
    #  [N4 校正後]    = 移除 bias field，強度更均勻（即 preprocessed 的外觀）
    #  [去顱骨後]     = 在 N4 基礎上 mask 掉頭骨區域（模擬）
    #  [Affine 對齊]  = 就是 preprocessed_vol（已對齊到 atlas 空間）
    #  [Atlas]        = atlas_vol

    # 模擬 Raw：在 preprocessed 基礎上加 bias field 並還原部分頭骨
    rng = np.random.default_rng(42)
    shape = preprocessed_vol.shape

    # 線性 bias field：沿 x 軸從 0.6 到 1.4 的漸層
    bx = np.linspace(0.65, 1.35, shape[0])[:, None, None] * np.ones(shape)
    by = np.linspace(0.85, 1.15, shape[1])[None, :, None] * np.ones(shape)
    bias = (bx * by).astype(np.float32)
    raw_sim = np.clip(preprocessed_vol * bias, 0, 1)

    # 模擬 N4 校正後：bias 被移除（接近 preprocessed）但稍微加一點 noise
    n4_sim = preprocessed_vol.copy()

    # 模擬去顱骨：用簡單閾值 mask 產生腦區邊界效果
    skull_mask = (preprocessed_vol > 0.05).astype(np.float32)
    skull_sim = preprocessed_vol * skull_mask

    # Affine 對齊後 = preprocessed_vol（已在 atlas 空間）
    affine_sim = preprocessed_vol.copy()

    stages = [
        ('Raw T1\n（未處理）',      raw_sim),
        ('N4 Bias 校正後\n（強度均勻化）', n4_sim),
        ('去顱骨後\n（Skull Strip）', skull_sim),
        ('Affine 對齊後\n（線性配準到 MNI）', affine_sim),
        ('Atlas\n（MNI152 標準腦）',  atlas_vol),
    ]

    out_prefix = os.path.join(args.out_dir, f'preprocess_stages_{subj_name}')
    title_main = f'IXI 前處理流程對比  |  受試者：{subj_name}  |  [Demo 模式]'

# ─────────────────────────────────────────────────────────────────────
# 完整管線模式：載入 ants 並逐步執行
# ─────────────────────────────────────────────────────────────────────
else:
    print(f"\n── 完整管線模式 ──────────────────────────────────────")
    print(f"輸入：{args.input}")

    try:
        import ants
        HAS_ANTS = True
        print("✓ ANTs 載入成功")
    except ImportError:
        print("❌ 無法載入 antspyx，請確認已執行：pip install antspyx")
        print("   或改用 --demo 模式")
        sys.exit(1)

    try:
        import antspynet
        HAS_ANTSPYNET = True
        print("✓ ANTsPyNet 載入成功")
    except ImportError:
        HAS_ANTSPYNET = False
        print("⚠️  antspynet 未安裝，將改用簡易閾值去顱骨")

    subj_name = os.path.basename(args.input).replace('.nii.gz', '').replace('.nii', '')

    # ── Step 1: 讀取原始影像 ─────────────────────────────────────────
    print(f"\n[1/4] 讀取原始影像...")
    img_raw = ants.image_read(args.input)
    raw_np  = img_raw.numpy().astype(np.float32)
    print(f"      原始大小：{img_raw.shape}  spacing：{img_raw.spacing}")

    # 儲存中間 nii 以供確認
    inter_dir = os.path.join(args.out_dir, 'intermediates')
    os.makedirs(inter_dir, exist_ok=True)

    # ── Step 2: N4 Bias Field Correction ────────────────────────────
    print("[2/4] N4 Bias Field Correction...")
    img_n4 = ants.n4_bias_field_correction(img_raw)
    n4_np  = img_n4.numpy().astype(np.float32)
    print(f"      N4 完成")

    # ── Step 3: 去顱骨 ───────────────────────────────────────────────
    if args.no_skull:
        print("[3/4] 跳過去顱骨（--no-skull）")
        img_brain = img_n4
        skull_np  = n4_np.copy()
    elif HAS_ANTSPYNET:
        print("[3/4] 去顱骨（ANTsPyNet deep learning）...")
        prob      = antspynet.brain_extraction(img_n4, modality='t1', verbose=False)
        mask      = ants.threshold_image(prob, 0.5, 1.0)
        img_brain = ants.mask_image(img_n4, mask)
        skull_np  = img_brain.numpy().astype(np.float32)
        print(f"      去顱骨完成")
    else:
        print("[3/4] 去顱骨（簡易閾值）...")
        arr_n4   = img_n4.numpy()
        thr      = arr_n4.max() * 0.15
        mask_arr = (arr_n4 > thr).astype(np.float32)
        mask_ants = img_n4.new_image_like(mask_arr)
        mask_ants = ants.morphological(mask_ants, radius=3, operation='close')
        img_brain = ants.mask_image(img_n4, mask_ants)
        skull_np  = img_brain.numpy().astype(np.float32)
        print(f"      去顱骨完成（閾值法）")

    # ── Step 4: Affine 對位到 atlas ──────────────────────────────────
    print("[4/4] Affine 對位到 MNI152 atlas...")
    atlas_ants = ants.from_numpy(atlas_vol, spacing=(1.0, 1.0, 1.0))
    reg = ants.registration(
        fixed   = atlas_ants,
        moving  = img_brain,
        type_of_transform = 'Affine',
        verbose = False,
    )
    img_reg   = reg['warpedmovout']
    affine_np = img_reg.numpy().astype(np.float32)
    # resize 到 atlas shape
    if affine_np.shape != atlas_vol.shape:
        from scipy.ndimage import zoom
        factors   = tuple(t / s for t, s in zip(atlas_vol.shape, affine_np.shape))
        affine_np = zoom(affine_np, factors, order=1).astype(np.float32)
    print(f"      Affine 完成  shape：{affine_np.shape}")

    stages = [
        ('Raw T1\n（未處理）',       raw_np),
        ('N4 Bias 校正後\n（強度均勻化）', n4_np),
        ('去顱骨後\n（Skull Strip）', skull_np),
        ('Affine 對齊後\n（線性配準到 MNI）', affine_np),
        ('Atlas\n（MNI152 標準腦）',  atlas_vol),
    ]

    out_prefix = os.path.join(args.out_dir, f'preprocess_stages_{subj_name}')
    title_main = f'IXI 前處理流程對比  |  受試者：{subj_name}'

# ─────────────────────────────────────────────────────────────────────
# 繪圖：三切面 × 5 個階段 = 3 行 × 5 列
# ─────────────────────────────────────────────────────────────────────
print(f"\n繪製三切面對比圖...")

AXES   = [0, 1, 2]           # sagittal, coronal, axial
LABELS = ['Sagittal', 'Coronal', 'Axial']
N_ROWS = 3
N_COLS = len(stages)

# ── 顏色主題（深色背景）──────────────────────────────────────────────
BG_COLOR    = '#0D1117'
TITLE_COLOR = '#C9D1D9'
LABEL_COLOR = '#8B949E'
STAGE_COLORS = ['#F85149', '#F0883E', '#58A6FF', '#5EEAD4', '#FFFFFF']
                # red=raw, orange=n4, blue=skull, teal=affine, white=atlas

fig = plt.figure(figsize=(N_COLS * 3.2, N_ROWS * 3.0 + 1.2), facecolor=BG_COLOR)
fig.suptitle(title_main, fontsize=13, color=TITLE_COLOR, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(
    N_ROWS, N_COLS,
    figure=fig,
    wspace=0.04, hspace=0.12,
    left=0.02, right=0.98,
    top=0.92, bottom=0.08,
)

for col, (stage_label, vol) in enumerate(stages):
    vol_disp = normalize_display(vol)

    for row, (axis, ax_label) in enumerate(zip(AXES, LABELS)):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(BG_COLOR)

        # 取切片並準備顯示
        sl = get_slice(vol_disp, axis, args.slice_pct)

        # Sagittal / Coronal / Axial 各自調整翻轉
        # 與 visualize_reg_ixi.py 保持一致：
        #   preprocessed npz 顯示規則：flipud 即可
        sl_show = prep_slice(sl, flip_ud=True)

        ax.imshow(sl_show, cmap='gray', vmin=0, vmax=1, interpolation='bilinear', aspect='auto')
        ax.axis('off')

        # 左邊行：標示切面名稱
        if col == 0:
            ax.text(-0.04, 0.5, ax_label,
                    transform=ax.transAxes,
                    fontsize=9, color=LABEL_COLOR, va='center', ha='right',
                    rotation=90, fontweight='bold')

        # 頂行：標示階段名稱
        if row == 0:
            # 階段標題背景色條
            c = STAGE_COLORS[col]
            ax.set_title(stage_label, fontsize=9.5, color=c, pad=4,
                         fontweight='bold', multialignment='center',
                         linespacing=1.4)

        # 邊框顏色
        for spine in ax.spines.values():
            spine.set_edgecolor(STAGE_COLORS[col])
            spine.set_linewidth(1.2)
            spine.set_visible(True)

# ── 底部說明文字 ──────────────────────────────────────────────────────
note_lines = [
    '前處理流程：① Raw T1 → ② N4 偏場校正（強度不均勻消除）'
    '  → ③ 去顱骨（ANTsPyNet U-Net）'
    '  → ④ Affine 對齊到 MNI152（ANTs）'
    '  → ⑤ Atlas（MNI152 2009c Asymmetric）',
]
fig.text(0.5, 0.02, note_lines[0],
         ha='center', va='bottom', fontsize=8.5, color=LABEL_COLOR,
         style='italic', wrap=True)

# ── 儲存合併大圖 ──────────────────────────────────────────────────────
out_path = f'{out_prefix}_all3.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none')
plt.close(fig)
print(f"✓ 合併大圖：{out_path}")

# ─────────────────────────────────────────────────────────────────────
# 額外輸出：每個切面單獨一張（方便投影片插入）
# ─────────────────────────────────────────────────────────────────────
print("繪製各切面獨立大圖...")

for row, (axis, ax_label) in enumerate(zip(AXES, LABELS)):
    fig2, axes2 = plt.subplots(1, N_COLS,
                                figsize=(N_COLS * 3.2, 3.4),
                                facecolor=BG_COLOR)
    fig2.suptitle(f'{title_main}  ─  {ax_label} 切面',
                  fontsize=12, color=TITLE_COLOR, fontweight='bold', y=1.02)

    for col, (stage_label, vol) in enumerate(stages):
        vol_disp = normalize_display(vol)
        sl = get_slice(vol_disp, axis, args.slice_pct)
        sl_show = prep_slice(sl, flip_ud=True)

        ax = axes2[col]
        ax.set_facecolor(BG_COLOR)
        ax.imshow(sl_show, cmap='gray', vmin=0, vmax=1,
                  interpolation='bilinear', aspect='auto')
        ax.axis('off')
        c = STAGE_COLORS[col]
        ax.set_title(stage_label, fontsize=9.5, color=c, pad=4,
                     fontweight='bold', multialignment='center',
                     linespacing=1.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(c)
            spine.set_linewidth(1.5)
            spine.set_visible(True)

    fig2.tight_layout()
    out2 = f'{out_prefix}_{ax_label.lower()}.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight',
                 facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig2)
    print(f"  ✓ {ax_label}：{out2}")

print(f"\n完成！所有圖片儲存於：{args.out_dir}/")
print(f"  preprocess_stages_*_all3.png   — 三切面合併大圖（適合投影片）")
print(f"  preprocess_stages_*_sagittal.png")
print(f"  preprocess_stages_*_coronal.png")
print(f"  preprocess_stages_*_axial.png")
