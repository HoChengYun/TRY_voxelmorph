"""
FreeSurfer recon-all 產物 -> VoxelMorph 訓練用 npz

與 IXI/preprocess_ixi.py 的差異：
  * 不做 N4           —— FreeSurfer 的 nu.mgz 階段已做過（--n4 可強制開啟）
  * 不做去顱骨         —— brain.mgz 已經去過（--brain-extract 可強制開啟）
  * 多搬一份 aseg 標籤  —— 用「與影像完全相同」的 Affine 變換 + 最近鄰內插
  * 切分以「受試者」為單位 —— 避免同一人的多次掃描橫跨 train/test 造成 data leakage

輸入目錄結構（由 FreeSurfer 端 mri_convert 產生）：
    <brain-dir>/<subject>.nii.gz    來自 mri/brain.mgz
    <seg-dir>/<subject>.nii.gz      來自 mri/aseg.mgz

輸出：
    <out-dir>/train/<subject>.npz   keys: vol (float32 [0,1]), seg (int16)
    <out-dir>/test/<subject>.npz
    <out-dir>/nii/                  若有 --save-nii
    <out-dir>/split.json            切分結果（可重現、可複核）

用法：
    # 先看歸戶與切分結果，不動任何影像
    python fs_pipeline\\preprocess_fs.py --brain-dir ... --seg-dir ... ^
        --atlas IXI\\atlas_mni152_09c_v3.nii.gz --out-dir ... --dry-run

    # 先驗證 1 顆（產生 nii 供 ITK-SNAP/Freeview 目視確認標籤與影像疊合）
    python fs_pipeline\\preprocess_fs.py --brain-dir ... --seg-dir ... ^
        --atlas IXI\\atlas_mni152_09c_v3.nii.gz --out-dir fs_check ^
        --only A001 --save-nii

    # 批次（需要 --list-is-final）
    python fs_pipeline\\preprocess_fs.py --brain-dir ... --seg-dir ... ^
        --atlas IXI\\atlas_mni152_09c_v3.nii.gz --out-dir fs_preprocessed_v1 ^
        --subject-list ... --list-is-final --save-nii
"""

import os
import re
import sys
import json
import glob
import random
import argparse
import numpy as np

# FreeSurfer 端已確認不可用的受試者（來源：D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md）
# 這 4 顆本來就不在「跑完 recon-all 的 167 顆」裡（A012/A043 未完成、T085 失敗、
# T065 身分待確認已從暫定清單扣掉），這裡再列一次只是保險。
DEFAULT_EXCLUDE = [
    'A043',   # 影像雜訊過高、灰白對比不足（白質只認出約 5%）
    'T085',   # 只有 120/192 張切片
    'A012',   # 資料夾混了兩次掃描，修好前不要用
    'T065',   # 資料夾名 T065 但 DICOM 病人 ID 是 T056，身分待確認
]

p = argparse.ArgumentParser()
p.add_argument('--brain-dir', required=True, help='brain.mgz 轉出的 .nii.gz 資料夾')
p.add_argument('--seg-dir',   default=None,  help='aseg.mgz 轉出的 .nii.gz 資料夾（不給則只存 vol）')
p.add_argument('--atlas',     required=True, help='對位目標 .nii.gz（帶 header）')
p.add_argument('--out-dir',   required=True)

p.add_argument('--subject-list', default=None,
               help='白名單檔（一行一個 ID，# 開頭為註解）。以 FreeSurfer 端給的清單為準，'
                    '不要靠掃資料夾推。')
p.add_argument('--list-is-final', action='store_true', default=False,
               help='宣告 --subject-list 是「混掃描全面檢查跑完後」的最終版。'
                    '沒有這個旗標時本腳本只做開發驗證，會拒絕批次跑。')
p.add_argument('--exclude', default=','.join(DEFAULT_EXCLUDE),
               help=f'排除清單，逗號分隔（預設：{",".join(DEFAULT_EXCLUDE)}）')
p.add_argument('--group-map', default=None,
               help='受試者歸戶對照表 TSV/CSV：<subject_id><TAB><person_id>。'
                    '沒列到的 ID 用預設規則（去掉結尾的 _<數字>）。')

p.add_argument('--test-frac', type=float, default=0.10)
p.add_argument('--seed', type=int, default=42)

p.add_argument('--n4', action='store_true', default=False,
               help='強制做 N4（預設關閉：FreeSurfer 的 nu.mgz 階段已做過）')
p.add_argument('--brain-extract', action='store_true', default=False,
               help='強制去顱骨（預設關閉：brain.mgz 已去過）')
p.add_argument('--interpolator', default='nearestNeighbor',
               choices=['nearestNeighbor', 'genericLabel'],
               help='標籤內插方式。絕對不可用 linear。')

p.add_argument('--save-nii', action='store_true', default=False)
p.add_argument('--only', default=None, help='只處理這一個受試者（驗證用）')
p.add_argument('--dry-run', action='store_true', default=False,
               help='只印出歸戶與切分結果，不做任何影像處理')
p.add_argument('--no-skip-done', dest='skip_done', action='store_false', default=True)
args = p.parse_args()


# ── 受試者歸戶 ───────────────────────────────────────────────────────
# 預設規則只處理「明確」的情況：結尾底線加數字（A016_1 -> A016）。
#
# A0131 / A0132 這種「數字直接黏在 ID 後面」的無法從字串安全判斷
# （A0131 可能是「A013 的第 1 次掃描」，也可能就是編號 A0131 的獨立受試者，
#  而且 A013 本身也存在），所以預設不自動合併，只發出警告。
# 確認是同一人之後請用 --group-map 明寫。
_SUFFIX_RE = re.compile(r'^(.*?)_\d+$')


def default_person_id(subject_id):
    m = _SUFFIX_RE.match(subject_id)
    return m.group(1) if m else subject_id


def load_group_map(path):
    gm = {}
    if not path:
        return gm
    with open(path, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [x.strip() for x in re.split(r'[\t,]', line) if x.strip()]
            if len(parts) != 2:
                sys.exit(f"[X] --group-map 第 {ln} 行格式錯誤（需要兩欄）：{line}")
            gm[parts[0]] = parts[1]
    return gm


def find_ambiguous(subject_ids):
    """
    找出「可能是同一人但字串規則不敢合併」的組合，交給人判斷。
    兩種型態：
      (1) 只差結尾一個數字      A0131 / A0132
      (2) 某 ID 是別的 ID 的前綴  A013 是 A0131 / A0132 的前綴
    """
    ids = set(subject_ids)
    groups = {}

    for s in subject_ids:
        if re.match(r'^.{4,}\d$', s):
            groups.setdefault(s[:-1], set()).add(s)

    for stem in list(groups):
        if stem in ids:                       # 前綴本身也是一個受試者
            groups[stem].add(stem)

    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


brain_dir = os.path.normpath(args.brain_dir)
seg_dir = os.path.normpath(args.seg_dir) if args.seg_dir else None

if not os.path.isdir(brain_dir):
    sys.exit(f"[X] 找不到 --brain-dir：{brain_dir}")
if seg_dir and not os.path.isdir(seg_dir):
    sys.exit(f"[X] 找不到 --seg-dir：{seg_dir}")

subjects = sorted(
    os.path.basename(f)[:-len('.nii.gz')]
    for f in glob.glob(os.path.join(brain_dir, '*.nii.gz'))
)
if not subjects:
    sys.exit(f"[X] {brain_dir} 裡沒有 .nii.gz")

print(f"掃到 {len(subjects)} 個檔案：{brain_dir}")

# ── 白名單 ───────────────────────────────────────────────────────────
if args.subject_list:
    with open(args.subject_list, encoding='utf-8-sig') as f:
        allow = {ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith('#')}
    missing = sorted(allow - set(subjects))
    if missing:
        print(f"[!] 白名單裡有 {len(missing)} 個 ID 在 --brain-dir 找不到："
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    before = len(subjects)
    subjects = [s for s in subjects if s in allow]
    print(f"    套用白名單：{before} -> {len(subjects)}")
else:
    print("[!] 沒有給 --subject-list：將使用資料夾裡的全部檔案。")

# ── 資料品質閘門 ─────────────────────────────────────────────────────
# FreeSurfer 端的「全資料夾混掃描檢查」尚未完成。混了兩次掃描但總層數沒超過
# 256 的資料夾，recon-all 不會報錯，會安靜跑出一顆「兩個人疊在一起」的腦，
# 餵進訓練會讓模型學到不存在的解剖結構。所以批次跑需要明確宣告清單是最終版。
if not args.list_is_final and not (args.dry_run or args.only):
    print()
    print("=" * 68)
    print("  [X] 拒絕批次執行：受試者清單尚未確認為最終版")
    print("=" * 68)
    print("  FreeSurfer 端的「全資料夾混掃描檢查」尚未完成。在那之前，清單裡")
    print("  可能仍混有『兩次掃描疊在一起』的受試者——recon-all 不會報錯。")
    print()
    print("  現在可以做的：")
    print("    --dry-run          看歸戶與切分結果")
    print("    --only <SUBJECT>   驗證單顆（建議搭 --save-nii 目視確認）")
    print()
    print("  拿到最終清單後，加上 --list-is-final 即可批次執行。")
    print("=" * 68)
    sys.exit(2)

# ── 排除清單 ─────────────────────────────────────────────────────────
excl = {s.strip() for s in args.exclude.split(',') if s.strip()}
hit = sorted(set(subjects) & excl)
if hit:
    print(f"    排除 {len(hit)} 個：{hit}")
    subjects = [s for s in subjects if s not in excl]

# ── 缺 seg 的檢查 ────────────────────────────────────────────────────
if seg_dir:
    no_seg = [s for s in subjects
              if not os.path.exists(os.path.join(seg_dir, s + '.nii.gz'))]
    if no_seg:
        print(f"[!] 有 {len(no_seg)} 個受試者沒有對應的 aseg，將只存 vol："
              f"{no_seg[:10]}{' ...' if len(no_seg) > 10 else ''}")

# ── 歸戶 ─────────────────────────────────────────────────────────────
gmap = load_group_map(args.group_map)
person_of = {s: gmap.get(s, default_person_id(s)) for s in subjects}
persons = {}
for s, pid in person_of.items():
    persons.setdefault(pid, []).append(s)

multi = {k: sorted(v) for k, v in persons.items() if len(v) > 1}
print(f"\n歸戶：{len(subjects)} 個掃描 -> {len(persons)} 位受試者"
      f"（--group-map：{args.group_map or '未提供，使用預設規則'}）")
if multi:
    print(f"  多次掃描 {len(multi)} 位（整組進同一個 split）：")
    for k, v in sorted(multi.items()):
        print(f"    {k}: {v}")

amb = find_ambiguous(subjects)
amb = {k: v for k, v in amb.items()
       if len({person_of[s] for s in v}) > 1}   # 已歸在一起的不必再警告
if amb:
    print("\n[!] 以下 ID 疑似同一人的多次掃描，但目前被當成不同人：")
    for k, v in sorted(amb.items()):
        print(f"      {v}")
    print("    預設規則只合併 `_1`/`_2` 這種底線後綴，不會猜數字直接黏在後面的情況。")
    print("    若確認為同一人，請寫成 --group-map 檔案（一行一組，TAB 分隔）：")
    for k, v in sorted(amb.items()):
        for s in v:
            print(f"        {s}\t{k}")

# ── 以「人」為單位切分 ───────────────────────────────────────────────
rng = random.Random(args.seed)
person_ids = sorted(persons.keys())
rng.shuffle(person_ids)
n_test = max(1, int(round(len(person_ids) * args.test_frac)))
test_persons = set(person_ids[:n_test])

split_of = {}
for pid, ss in persons.items():
    tag = 'test' if pid in test_persons else 'train'
    for s in ss:
        split_of[s] = tag

n_tr = sum(1 for v in split_of.values() if v == 'train')
n_te = sum(1 for v in split_of.values() if v == 'test')
print(f"\n切分（受試者層級，seed={args.seed}，test_frac={args.test_frac}）：")
print(f"  train  {len(persons) - n_test:3d} 人 / {n_tr:3d} 個掃描")
print(f"  test   {n_test:3d} 人 / {n_te:3d} 個掃描")

leak = [pid for pid, ss in persons.items() if len({split_of[s] for s in ss}) > 1]
assert not leak, f"切分有誤，這些人橫跨 train/test：{leak}"
print("  [v] 沒有受試者橫跨 train/test")

if args.only:
    if args.only not in split_of:
        sys.exit(f"[X] --only {args.only} 不在可用清單裡")
    subjects = [args.only]
    print(f"\n[only] 只處理 {args.only}（{split_of[args.only]}）")

if args.dry_run:
    print("\n[dry-run] 不做影像處理，結束。")
    sys.exit(0)


# ── 影像處理 ─────────────────────────────────────────────────────────
try:
    import ants
except ImportError:
    sys.exit("[X] 請先安裝 antspyx：pip install antspyx")

os.makedirs(args.out_dir, exist_ok=True)
for sp in ['train', 'test']:
    os.makedirs(os.path.join(args.out_dir, sp), exist_ok=True)
nii_dir = os.path.join(args.out_dir, 'nii')
if args.save_nii:
    os.makedirs(nii_dir, exist_ok=True)

atlas_ants = ants.image_read(os.path.normpath(args.atlas))
target_shape = atlas_ants.shape
print(f"\nAtlas：{args.atlas}")
print(f"  shape={target_shape}  spacing={tuple(round(s, 4) for s in atlas_ants.spacing)}\n")

with open(os.path.join(args.out_dir, 'split.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'seed': args.seed,
        'test_frac': args.test_frac,
        'list_is_final': args.list_is_final,
        'subject_list': args.subject_list,
        'group_map': args.group_map,
        'excluded': sorted(excl),
        'person_of': person_of,
        'split_of': split_of,
    }, f, indent=2, ensure_ascii=False)

ok = skip = fail = 0
n = len(subjects)

for i, subj in enumerate(subjects, 1):
    split = split_of[subj]
    dst = os.path.join(args.out_dir, split, subj + '.npz')

    if args.skip_done and os.path.exists(dst):
        skip += 1
        print(f"[{i:3d}/{n}] 略過（已存在）：{subj}")
        continue

    print(f"[{i:3d}/{n}] 處理：{subj}  ({split})")
    try:
        img = ants.image_read(os.path.join(brain_dir, subj + '.nii.gz'))
        print(f"        原始：shape={img.shape}  "
              f"spacing={tuple(round(s, 3) for s in img.spacing)}")

        if args.n4:
            print("        N4 bias correction ...")
            img = ants.n4_bias_field_correction(img)
        if args.brain_extract:
            import antspynet
            print("        去顱骨（antspynet）...")
            prob = antspynet.brain_extraction(img, modality='t1', verbose=False)
            img = ants.mask_image(img, ants.threshold_image(prob, 0.5, 1.0))

        # Affine 對位到 atlas
        print("        Affine 對位到 atlas ...")
        reg = ants.registration(
            fixed=atlas_ants,
            moving=img,
            type_of_transform='Affine',
            verbose=False,
        )
        img_reg = reg['warpedmovout']

        img_np = img_reg.numpy().astype(np.float32)
        if img_np.shape != target_shape:
            raise ValueError(f"配準輸出 shape {img_np.shape} != atlas shape {target_shape}")

        # 用「同一個變換」把 aseg 搬過去；最近鄰內插；不正規化
        seg_np = None
        seg_src = os.path.join(seg_dir, subj + '.nii.gz') if seg_dir else None
        if seg_src and os.path.exists(seg_src):
            print(f"        搬 aseg 標籤（{args.interpolator}）...")
            seg = ants.image_read(seg_src)
            seg_reg = ants.apply_transforms(
                fixed=atlas_ants,
                moving=seg,
                transformlist=reg['fwdtransforms'],   # 重用影像的變換
                interpolator=args.interpolator,
            )
            seg_f = seg_reg.numpy()
            # 內插正確的話值必為整數；出現小數代表用錯內插法
            frac = float(np.abs(seg_f - np.round(seg_f)).max())
            if frac > 1e-6:
                raise ValueError(f"標籤出現非整數值（最大偏差 {frac:.4g}）—— 內插法用錯了")
            seg_np = np.round(seg_f).astype(np.int16)
            if seg_np.shape != target_shape:
                raise ValueError(f"標籤 shape {seg_np.shape} != atlas shape {target_shape}")

            labs_before = set(np.unique(seg.numpy()).round().astype(np.int32).tolist())
            labs_after = set(np.unique(seg_np).tolist())
            lost = sorted(labs_before - labs_after)
            print(f"        標籤數：{len(labs_before)} -> {len(labs_after)}"
                  + (f"   [!] 消失：{lost}" if lost else ""))

        # 正規化到 [0,1]（percentile 只看腦內 voxel）
        pos = img_np[img_np > 0]
        if pos.size == 0:
            raise ValueError("影像全是 0")
        p1, p99 = np.percentile(pos, [1, 99])
        img_np = np.clip(img_np, p1, p99)
        img_np = ((img_np - img_np.min()) /
                  (img_np.max() - img_np.min() + 1e-8)).astype(np.float32)

        if seg_np is not None:
            np.savez_compressed(dst, vol=img_np, seg=seg_np)
        else:
            np.savez_compressed(dst, vol=img_np)
        ok += 1
        print(f"        [v] {dst}")
        print(f"            vol {img_np.shape} float32 "
              f"[{img_np.min():.3f}, {img_np.max():.3f}]"
              + (f" | seg {seg_np.shape} int16" if seg_np is not None else " | (無 seg)"))

        if args.save_nii:
            ants.image_write(img_reg.new_image_like(img_np),
                             os.path.join(nii_dir, subj + '.nii.gz'))
            if seg_np is not None:
                ants.image_write(img_reg.new_image_like(seg_np.astype(np.float32)),
                                 os.path.join(nii_dir, subj + '_seg.nii.gz'))
            print(f"        [v] nii -> {nii_dir}")

    except Exception as e:
        fail += 1
        print(f"        [X] 失敗：{e}")
        import traceback
        traceback.print_exc()
    print()

print("=" * 55)
print(f"完成！ 成功={ok} 略過={skip} 失敗={fail}")
print(f"輸出：{args.out_dir}  （切分記錄：split.json）")
