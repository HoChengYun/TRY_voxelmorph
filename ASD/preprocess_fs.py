"""
FreeSurfer recon-all 產物 -> VoxelMorph 訓練用 npz

與 IXI/preprocess_ixi.py 的差異：
  * 不做 N4           —— FreeSurfer 的 nu.mgz 階段已做過（--n4 可強制開啟）
  * 不做去顱骨         —— norm.mgz / brain.mgz 都已去過（--brain-extract 可強制開啟）
  * 多搬一份 aseg 標籤  —— 用「與影像完全相同」的 Affine 變換 + 最近鄰內插
  * 切分以「受試者」為單位 —— 避免同一人的多次掃描橫跨 train/test 造成 data leakage

輸入目錄結構（由 FreeSurfer 端 mri_convert 產生）：
    <img-dir>/<subject>.nii.gz      來自 mri/norm.mgz（或 brain.mgz，全體一致即可）
    <seg-dir>/<subject>.nii.gz      來自 mri/aseg.mgz

輸出：
    <out-dir>/train/<subject>.npz   keys: vol (float32 [0,1]), seg (int16)
    <out-dir>/test/<subject>.npz
    <out-dir>/nii/                  若有 --save-nii
    <out-dir>/split.json            切分結果（可重現、可複核）

用法：
    # 先看歸戶與切分結果，不動任何影像
    python ASD\\preprocess_fs.py --img-dir ... --seg-dir ... ^
        --atlas IXI\\atlas_mni152_09c_v3.nii.gz --out-dir ... --dry-run

    # 先驗證 1 顆（產生 nii 供 ITK-SNAP/Freeview 目視確認標籤與影像疊合）
    python ASD\\preprocess_fs.py --img-dir ... --seg-dir ... ^
        --atlas IXI\\atlas_mni152_09c_v3.nii.gz --out-dir fs_check ^
        --only A001 --save-nii

    # 批次（需要 --list-is-final）
    python ASD\\preprocess_fs.py --img-dir ... --seg-dir ... ^
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

# Windows 主控台預設 cp950，印到 emoji 會 UnicodeEncodeError 直接中斷程式。
# 不要求使用者記得設 PYTHONIOENCODING —— 忘一次就白跑一輪。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 曾經被判定有問題的受試者（來源：D:\MyHome\MRI\FreeSurfer\docs\ASD_資料品質記錄.md）
#
# ⚠️ 這只是「沒有給 --subject-list 時」的安全網，不是權威。
#    有給 --subject-list 時，**清單才是權威**，本表不會拿去砍清單裡的人
#    —— 因為狀態會變。實例：A012 一度因「資料夾混了兩次掃描」被排除，
#    2026-08-23 修復後重跑完成，已列入 FINAL 清單。若當時本表還無腦生效，
#    167 會安靜地變成 166，而且不會有任何錯誤訊息。
DEFAULT_EXCLUDE = {
    'A043': '影像雜訊過高、灰白對比不足（白質只認出約 5%）',
    'T085': '只有 120/192 張切片（缺 0001–0072）',
    'T065': '資料夾名 T065 但 DICOM 病人 ID 是 T056，身分待確認',
    # 'A012': 已於 2026-08-23 修復（分離誤放的 0801 那組後重跑），不再排除
}

p = argparse.ArgumentParser()
# 影像來源用中性名稱：FreeSurfer 端可能給 norm.mgz 或 brain.mgz，兩者都是
# 去顱骨後的 uchar（白質錨在 110），對本腳本沒有差別。舊名保留為別名。
p.add_argument('--img-dir', '--norm-dir', '--brain-dir', dest='img_dir', required=True,
               metavar='DIR',
               help='影像 .nii.gz 資料夾（norm.mgz 或 brain.mgz 轉出的都可以）。'
                    '別名：--norm-dir / --brain-dir')
p.add_argument('--seg-dir',   default=None,  help='aseg.mgz 轉出的 .nii.gz 資料夾（不給則只存 vol）')
p.add_argument('--atlas',     required=True, help='對位目標 .nii.gz（帶 header）')
p.add_argument('--out-dir',   required=True)

p.add_argument('--subject-list', default=None,
               help='白名單檔（一行一個 ID，# 開頭為註解）。以 FreeSurfer 端給的清單為準，'
                    '不要靠掃資料夾推。')
p.add_argument('--list-is-final', action='store_true', default=False,
               help='宣告 --subject-list 是「混掃描全面檢查跑完後」的最終版。'
                    '沒有這個旗標時本腳本只做開發驗證，會拒絕批次跑。')
p.add_argument('--exclude', default=None,
               help='排除清單，逗號分隔。明寫時一律生效。'
                    f'不給時：沒有 --subject-list 才套用內建安全網（{",".join(DEFAULT_EXCLUDE)}）；'
                    '有 --subject-list 時以清單為準，內建表只會提示不會砍人。')
p.add_argument('--group-map', default=None,
               help='受試者歸戶對照表 TSV/CSV：<subject_id><TAB><person_id>。'
                    '明列的一律優先，沒列到的才走 --grouping 的規則。')
p.add_argument('--allow-suffix-names', action='store_true',
               help='略過「同一顆受試者的多種產物在同一目錄」的偵測。'
                    '只有在你確定檔名的後綴真的是不同受試者時才用')
p.add_argument('--show-weak-groups', action='store_true',
               help='把「只差末位數字」的低度懷疑組逐組列出（預設只給一行計數，'
                    '因為補零流水號如 VNT001/VNT002 會大量誤觸）')
p.add_argument('--grouping', default='auto', choices=['auto', 'none'],
               help='沒被 --group-map 明列的 ID 怎麼歸戶。'
                    'auto（預設）＝去掉結尾的 _<數字>（A016_1 與 A016_2 視為同一人）；'
                    'none ＝每個掃描各自成一人，完全不合併。'
                    '選 none 等於假設「沒有任何一組是同一人的重複掃描」，'
                    '假設若錯會造成 data leakage，請在方法學中說明。')

p.add_argument('--split-from', default=None,
               help='沿用別批的 train/test 切分，不重新隨機。'
                    '吃 make_mixed_set.py 的 mixed_manifest.json 或本腳本的 split.json。'
                    '做「同一群人、不同標籤來源」的對照時必須用 —— 否則兩個 arm 各自隨機切，'
                    'test 會是不同的人，Dice 差多少就無法歸因')
p.add_argument('--test-frac', type=float, default=0.10)
p.add_argument('--seed', type=int, default=42)

p.add_argument('--n4', action='store_true', default=False,
               help='強制做 N4（預設關閉：FreeSurfer 的 nu.mgz 階段已做過）')
p.add_argument('--brain-extract', action='store_true', default=False,
               help='強制去顱骨（預設關閉：norm.mgz / brain.mgz 都已去過）')
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

    只回報有**額外佐證**的組，兩種：
      (1) 去掉末位後的字根本身也是一個受試者   A013 是 A0131 / A0132 的字根
      (2) 末位數字前面有分隔符                 A016_1 / A016_2

    🔴 2026-09-07 收緊過。原本只要「ID ≥5 字元且結尾是數字」就歸組，
       結果 VNT001~VNT009 全部變成字根 VNT00 被當成同一人 —— VNT 那包
       68 顆全被歸進 7 組誤報。補零流水號本來就長這樣，那不是證據。
       誤報會訓練人忽略警告，比不報還糟。
    """
    ids = set(subject_ids)
    strong, weak = {}, {}

    for s in subject_ids:
        if not re.match(r'^.{4,}\d$', s):
            continue
        body = s[:-1]
        stem = body.rstrip('_-')              # A016_1 -> A016（不要留下 A016_）
        if not stem:
            continue
        # 有分隔符，或字根本身就是一個受試者 -> 強訊號
        if body != stem or stem in ids:
            strong.setdefault(stem, set()).add(s)
        else:
            weak.setdefault(stem, set()).add(s)

    for stem in list(strong):
        if stem in ids:                       # 字根本身也是一個受試者
            strong[stem].add(stem)

    return ({k: sorted(v) for k, v in strong.items() if len(v) > 1},
            {k: sorted(v) for k, v in weak.items() if len(v) > 1})


img_dir = os.path.normpath(args.img_dir)
seg_dir = os.path.normpath(args.seg_dir) if args.seg_dir else None

if not os.path.isdir(img_dir):
    sys.exit(f"[X] 找不到 --img-dir：{img_dir}")
if seg_dir and not os.path.isdir(seg_dir):
    sys.exit(f"[X] 找不到 --seg-dir：{seg_dir}")

subjects = sorted(
    os.path.basename(f)[:-len('.nii.gz')]
    for f in glob.glob(os.path.join(img_dir, '*.nii.gz'))
)
if not subjects:
    sys.exit(f"[X] {img_dir} 裡沒有 .nii.gz")

print(f"掃到 {len(subjects)} 個檔案：{img_dir}")

# 🔴 受試者 ID 直接來自檔名，所以「一個受試者有多個檔案放在同一個目錄」會被拆成多顆。
#    實例（2026-09-08 實測）：tigerbx 的輸出是 <subject>_tbet / _tbetmask / _aseg / _dgm
#    四個檔案同一層。把 --img-dir 和 --seg-dir 都指過去的話，2 顆會變成 8 個「受試者」，
#    而且因為 seg_dir 同一個目錄、每個都找得到同名檔，連「缺對應 seg」都不會警告
#    —— 等於拿分割圖當訓練影像、自己配自己，一路安靜跑完。
_suffixes = {}
for s in subjects:
    if '_' in s:
        _suffixes.setdefault(s.rsplit('_', 1)[1], []).append(s)
_multi = {k: v for k, v in _suffixes.items() if len(v) > 1}
if len(_multi) > 1 and sum(len(v) for v in _multi.values()) > len(subjects) * 0.5:
    print()
    print("[X] 檔名看起來是「同一顆受試者的多種產物放在同一個目錄」：")
    for k in sorted(_multi)[:6]:
        print(f"      _{k}  ×{len(_multi[k])}   例如 {_multi[k][0]}")
    print("    受試者 ID 是直接取檔名的，這樣會把每一種產物都當成一顆獨立的腦。")
    print("    請先把影像與標籤分到不同目錄、檔名只留受試者 ID，例如：")
    print("        <目標>/img/<subject>.nii.gz")
    print("        <目標>/seg/<subject>.nii.gz")
    print("    確定要照現況跑，加 --allow-suffix-names。")
    if not args.allow_suffix_names:
        sys.exit(2)
    print("    已加 --allow-suffix-names，繼續。")

if seg_dir and os.path.normpath(seg_dir) == os.path.normpath(img_dir):
    print("[!] --img-dir 與 --seg-dir 是同一個目錄 —— 影像與標籤會取到同一個檔案。")

# ── 白名單 ───────────────────────────────────────────────────────────
if args.subject_list:
    with open(args.subject_list, encoding='utf-8-sig') as f:
        allow = {ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith('#')}
    missing = sorted(allow - set(subjects))
    if missing:
        print(f"[!] 白名單裡有 {len(missing)} 個 ID 在 --img-dir 找不到："
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    before = len(subjects)
    subjects = [s for s in subjects if s in allow]
    print(f"    套用白名單：{before} -> {len(subjects)}")
else:
    print("[!] 沒有給 --subject-list：將使用資料夾裡的全部檔案。")

# ── 資料品質閘門 ─────────────────────────────────────────────────────
# 批次跑需要明確宣告「這份清單是資料品質檢查跑完後的最終版」。
#
# 為什麼要這個機制：混了兩次掃描但總層數沒超過 256 的資料夾，recon-all
# 不會報錯，會安靜跑出一顆「兩個人疊在一起」的腦，餵進訓練會讓模型學到
# 不存在的解剖結構。這種錯誤事後幾乎看不出來，所以要在跑之前擋。
#
# ASD 這批：2026-08-23 檢查完成（167 個資料夾、異常 0、167×192=32,064 閉合），
# 可以加 --list-is-final。機制保留給之後的新資料集用。
if not args.list_is_final and not (args.dry_run or args.only):
    print()
    print("=" * 68)
    print("  [X] 拒絕批次執行：受試者清單尚未確認為最終版")
    print("=" * 68)
    print("  資料品質檢查（例如「資料夾是否混了兩次掃描」）若尚未完成，清單裡")
    print("  可能仍混有『兩個人疊在一起』的受試者——recon-all 不會報錯。")
    print()
    print("  現在可以做的：")
    print("    --dry-run          看歸戶與切分結果")
    print("    --only <SUBJECT>   驗證單顆（建議搭 --save-nii 目視確認）")
    print()
    print("  拿到最終清單後，加上 --list-is-final 即可批次執行。")
    print("=" * 68)
    sys.exit(2)

# ── 排除清單 ─────────────────────────────────────────────────────────
# 權威順序：明寫的 --exclude > --subject-list > 內建安全網
if args.exclude is not None:
    excl = {s.strip() for s in args.exclude.split(',') if s.strip()}
    src = '--exclude（明寫）'
elif args.subject_list:
    excl = set()          # 清單就是權威，內建表不砍人
    src = None
    noted = sorted(set(subjects) & set(DEFAULT_EXCLUDE))
    if noted:
        print(f"\n[i] 清單裡有 {len(noted)} 個曾被判定有問題的受試者，"
              f"依 --subject-list 為準予以保留：")
        for s in noted:
            print(f"      {s}：{DEFAULT_EXCLUDE[s]}")
        print("    若確認仍不可用，請明寫 --exclude 覆寫。")
else:
    excl = set(DEFAULT_EXCLUDE)
    src = '內建安全網（未提供 --subject-list）'

if excl:
    hit = sorted(set(subjects) & excl)
    if hit:
        print(f"    排除 {len(hit)} 個（來源：{src}）：{hit}")
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
fallback = default_person_id if args.grouping == 'auto' else (lambda s: s)
person_of = {s: gmap.get(s, fallback(s)) for s in subjects}
persons = {}
for s, pid in person_of.items():
    persons.setdefault(pid, []).append(s)

multi = {k: sorted(v) for k, v in persons.items() if len(v) > 1}
print(f"\n歸戶：{len(subjects)} 個掃描 -> {len(persons)} 位受試者")
print(f"  --grouping  : {args.grouping}"
      + ('（去掉結尾 _<數字>）' if args.grouping == 'auto' else '（每個掃描各自成一人，不合併）'))
print(f"  --group-map : {args.group_map or '未提供'}"
      + (f'（明列 {len(gmap)} 筆）' if gmap else ''))
if multi:
    print(f"  多次掃描 {len(multi)} 位（整組進同一個 split）：")
    for k, v in sorted(multi.items()):
        print(f"    {k}: {v}")

# 疑似同組但目前被當成不同人 —— 一律提醒，不論成因是規則沒抓到還是刻意不合併
amb, weak = find_ambiguous(subjects)
if args.grouping == 'auto':
    # auto 模式下再補上「底線後綴」型態（find_ambiguous 抓不到）
    for s in subjects:
        m = _SUFFIX_RE.match(s)
        if m:
            amb.setdefault(m.group(1), []).append(s)
    amb = {k: sorted(set(v)) for k, v in amb.items() if len(set(v)) > 1}
amb = {k: v for k, v in amb.items()
       if len({person_of[s] for s in v}) > 1}   # 已歸在一起的不必再警告
weak = {k: v for k, v in weak.items()
        if len({person_of[s] for s in v}) > 1}
if amb:
    n_scan = sum(len(v) for v in amb.values())
    print(f"\n[!] 以下 {len(amb)} 組（共 {n_scan} 個掃描）疑似同一人，但目前被當成不同人：")
    for k, v in sorted(amb.items()):
        print(f"      {v}")
    print("    ⚠️ 假設若錯，同一人的掃描可能一個進 train、一個進 test，"
          "造成 data leakage，Dice 會虛高。")
    print("    這個假設必須在方法學中說明，或先向資料提供者確認。")
    print("    確認為同一人後，寫成 --group-map 檔案重跑切分即可（不用改程式）：")
    for k, v in sorted(amb.items()):
        for s in v:
            print(f"        {s}\t{k}")

if weak:
    # 只差末位數字、但字根不是受試者也沒有分隔符 —— 補零流水號長這樣，多半是誤報。
    # 只給一行計數，不列清單，免得淹掉上面真正該看的。
    n_scan = sum(len(v) for v in weak.values())
    print(f"\n[i] 另有 {len(weak)} 組（{n_scan} 個掃描）只是「末位數字不同」"
          f"（如 {sorted(weak.values())[0][0]} / {sorted(weak.values())[0][1]}）。")
    print("    補零流水號本來就長這樣，通常不是同一人。要逐組看的話：--show-weak-groups")
    if args.show_weak_groups:
        for k, v in sorted(weak.items()):
            print(f"      {v}")

# ── 以「人」為單位切分 ───────────────────────────────────────────────
if args.split_from:
    # 沿用別批的切分。做「同一群人、不同標籤來源」的對照時必須這樣做：
    # 各自隨機切的話，兩個 arm 的 test 會是不同的人，Dice 差多少就同時混了
    # 「標籤不同」和「測試對象不同」兩個變因，無法歸因。
    ref = json.load(open(args.split_from, encoding='utf-8'))
    if 'members' in ref:                       # make_mixed_set.py 的 mixed_manifest.json
        ref_split = {k[:-4] if k.endswith('.npz') else k: v['split']
                     for k, v in ref['members'].items()}
    elif 'split_of' in ref:                    # preprocess_fs.py 自己的 split.json
        ref_split = dict(ref['split_of'])
    else:
        sys.exit(f"[X] --split-from 認不得這個檔的格式：{args.split_from}\n"
                 f"    需要 mixed_manifest.json（有 members）或 split.json（有 split_of）")

    missing = sorted(set(subjects) - set(ref_split))
    if missing:
        sys.exit(f"[X] 參考切分裡沒有這 {len(missing)} 位：{missing[:8]}\n"
                 f"    兩批的受試者必須完全相同才能沿用切分。")

    split_of = {s: ref_split[s] for s in subjects}

    # 沿用來的切分也必須是受試者層級的 —— 來源若有 leakage，這裡會照抄
    bad = [pid for pid, ss in persons.items() if len({split_of[s] for s in ss}) > 1]
    if bad:
        sys.exit(f"[X] 沿用的切分讓這些人橫跨 train/test：{bad}\n"
                 f"    來源的切分與本批的 --group-map 不相容，請確認。")

    test_persons = {pid for pid, ss in persons.items() if split_of[ss[0]] == 'test'}
    n_test = len(test_persons)
    print(f"\n[i] 切分沿用 {os.path.relpath(args.split_from, os.path.dirname(os.path.abspath(__file__)))}"
          f" —— 不重新隨機")
else:
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
        'img_dir': img_dir,
        'seg_dir': seg_dir,
        # 這份切分是在什麼歸戶假設下產生的 —— 之後回頭看才知道能不能比較
        'grouping': args.grouping,
        'grouping_note': ('每個掃描各自成一人，完全不合併（假設沒有任何一組是'
                          '同一人的重複掃描）' if args.grouping == 'none'
                          else '去掉結尾 _<數字> 後視為同一人'),
        'group_map': args.group_map,
        'ambiguous_kept_separate': {k: v for k, v in sorted(amb.items())},
        'n_scans': len(subjects),
        'n_persons': len(persons),
        'excluded': sorted(excl),
        'person_of': person_of,
        'split_of': split_of,
    }, f, indent=2, ensure_ascii=False)

ok = skip = fail = moved = 0
n = len(subjects)

for i, subj in enumerate(subjects, 1):
    split = split_of[subj]
    dst = os.path.join(args.out_dir, split, subj + '.npz')

    # 切分改變時（例如補了 --group-map），同一顆的舊檔還躺在另一個資料夾裡。
    # 直接搬過來，不要重跑 —— ANTs 配準有隨機取樣，重跑會得到跟原本略微不同的結果，
    # 那會讓同一份資料集裡的檔案來自兩次不同的處理。
    other = 'test' if split == 'train' else 'train'
    src_old = os.path.join(args.out_dir, other, subj + '.npz')
    if os.path.exists(src_old) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src_old, dst)
        moved += 1
        print(f"[{i:3d}/{n}] 切分改變，{other} -> {split}：{subj}")
        continue
    if os.path.exists(src_old) and os.path.exists(dst):
        os.remove(src_old)      # 兩邊都有 -> 舊的那份是殘留
        print(f"[{i:3d}/{n}] 清掉 {other}/ 的殘留：{subj}")

    if args.skip_done and os.path.exists(dst):
        skip += 1
        print(f"[{i:3d}/{n}] 略過（已存在）：{subj}")
        continue

    print(f"[{i:3d}/{n}] 處理：{subj}  ({split})")
    try:
        img = ants.image_read(os.path.join(img_dir, subj + '.nii.gz'))
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
print(f"完成！ 成功={ok} 略過={skip} 切分改變而搬移={moved} 失敗={fail}")
print(f"輸出：{args.out_dir}  （切分記錄：split.json）")
