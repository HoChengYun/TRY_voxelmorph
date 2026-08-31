# -*- coding: utf-8 -*-
"""產生 meeting 簡報的 .dc.html 投影片。

視覺系統
    墨色 #141A1D 深頁（開場/章節/收尾）夾淺色 #FAFAF8 內容頁
    襯線標題 Noto Serif TC + 黑體內文 Noto Sans TC + 等寬數字 IBM Plex Mono
    檢核章：■ 已驗證 / □ 待確認 —— 貫穿全場的記號
    強調色：#0E7C7B 深青（已驗證的數字）、#A34F1B 鏽橙（風險與保留）
"""
import io
import os

OUT = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720

INK      = '#141A1D'
PAPER    = '#FAFAF8'
SURFACE  = '#EFEFEB'
RULE     = '#D9D9D2'
MUTED    = '#5F6A6B'
TEAL     = '#0E7C7B'
RUST     = '#A34F1B'
ONDARK   = '#E8E6E0'
ONDARKMU = '#8A9294'

HELMET = """<helmet>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@500;700&amp;family=Noto+Sans+TC:wght@400;500;700&amp;family=IBM+Plex+Mono:wght@400;600&amp;display=swap">
  <style>
    body { margin: 0; }
    * { box-sizing: border-box; }
    a { color: %s; text-decoration: none; }
    a:hover { color: %s; }
  </style>
</helmet>""" % (TEAL, RUST)

SERIF = "'Noto Serif TC', 'Songti TC', serif"
SANS  = "'Noto Sans TC', 'PingFang TC', sans-serif"
MONO  = "'IBM Plex Mono', 'Courier New', monospace"


def page(body, bg=PAPER):
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="./support.js"></script>
</head>
<body>
<x-dc>
%s
<div style="width: %dpx; height: %dpx; background: %s; font-family: %s; position: relative; overflow: hidden;">
%s
</div>
</x-dc>
</body>
</html>
""" % (HELMET, W, H, bg, SANS, body)


def kicker(text, color=MUTED):
    return ('<div style="font-family: %s; font-size: 12px; letter-spacing: 0.18em; '
            'text-transform: uppercase; color: %s; font-weight: 600;">%s</div>'
            % (MONO, color, text))


def title(text, color=INK, size=40):
    return ('<div style="font-family: %s; font-size: %dpx; font-weight: 700; color: %s; '
            'line-height: 1.22; margin-top: 10px; text-wrap: pretty;">%s</div>'
            % (SERIF, size, color, text))


def num(n):
    """頁碼。"""
    return ('<div style="position: absolute; right: 56px; bottom: 34px; font-family: %s; '
            'font-size: 12px; color: %s;">%02d</div>' % (MONO, MUTED, n))


def stat(value, label, color=TEAL, vsize=44):
    return ("""<div style="display: flex; flex-direction: column; gap: 4px;">
  <div style="font-family: %s; font-size: %dpx; font-weight: 600; color: %s; line-height: 1;">%s</div>
  <div style="font-size: 13px; color: %s; line-height: 1.4;">%s</div>
</div>""" % (MONO, vsize, color, value, MUTED, label))


def mark(ok=True):
    """檢核章：■ 已驗證 / □ 待確認。"""
    c = TEAL if ok else RUST
    g = 'none' if ok else c
    return ('<span style="display: inline-block; width: 9px; height: 9px; background: %s; '
            'border: 1.5px solid %s; margin-right: 9px; flex: none; margin-top: 6px;">'
            '</span>' % (c if ok else 'transparent', c if ok else g))


def bullets(items, gap=14, fs=15):
    rows = []
    for ok, txt in items:
        rows.append('<div style="display: flex; align-items: flex-start; gap: 0;">%s'
                    '<div style="font-size: %dpx; line-height: 1.62; color: %s; '
                    'text-wrap: pretty;">%s</div></div>' % (mark(ok), fs, INK, txt))
    return ('<div style="display: flex; flex-direction: column; gap: %dpx;">%s</div>'
            % (gap, ''.join(rows)))


def table(head, rows, widths, fs=14):
    ths = ''.join(
        '<div style="font-size: 11.5px; letter-spacing: 0.08em; color: %s; font-weight: 700; '
        'text-transform: uppercase; font-family: %s;">%s</div>' % (MUTED, MONO, h) for h in head)
    out = ['<div style="display: grid; grid-template-columns: %s; gap: 10px 18px; '
           'align-items: start; padding-bottom: 10px; border-bottom: 1px solid %s;">%s</div>'
           % (widths, RULE, ths)]
    for r in rows:
        tds = ''.join('<div style="font-size: %dpx; line-height: 1.55; color: %s;">%s</div>'
                      % (fs, INK, c) for c in r)
        out.append('<div style="display: grid; grid-template-columns: %s; gap: 10px 18px; '
                   'align-items: start; padding: 11px 0; border-bottom: 1px solid %s;">%s</div>'
                   % (widths, RULE, tds))
    return '<div style="display: flex; flex-direction: column;">%s</div>' % ''.join(out)


def figure(src, alt, cap, h=470):
    return ("""<div style="display: flex; flex-direction: column; gap: 10px;">
  <div style="background: %s; border: 1px solid %s; padding: 12px; display: flex; justify-content: center;">
    <img src="%s" alt="%s" style="max-height: %dpx; max-width: 100%%; object-fit: contain; display: block;">
  </div>
  <div style="font-size: 12.5px; color: %s; line-height: 1.5;">%s</div>
</div>""" % (SURFACE, RULE, src, alt, h, MUTED, cap))


def code(text):
    return ('<div style="font-family: %s; font-size: 13px; background: %s; border: 1px solid %s; '
            'padding: 12px 14px; color: %s; line-height: 1.7; white-space: pre;">%s</div>'
            % (MONO, SURFACE, RULE, INK, text))


PAD = 'position: absolute; left: 56px; top: 48px; right: 56px;'
SLIDES = {}

# ── 01 開場 ──────────────────────────────────────────────────────────
SLIDES['Main'] = page("""
<div style="position: absolute; left: 72px; top: 150px; right: 72px;">
  %s
  <div style="font-family: %s; font-size: 56px; font-weight: 700; color: %s; line-height: 1.2; margin-top: 18px;">
    ASD 資料集：從 FreeSurfer<br>到 Dice 驗證
  </div>
  <div style="font-size: 17px; color: %s; margin-top: 24px; line-height: 1.7;">
    167 顆受試者影像的完整處理鏈 —— 資料把關、標籤搬運、模型訓練、重疊度評估
  </div>
  <div style="display: flex; gap: 56px; margin-top: 54px;">
    %s %s %s
  </div>
</div>
<div style="position: absolute; left: 72px; bottom: 46px; font-family: %s; font-size: 12.5px; color: %s;">
  何承運　·　2026 / 08　·　VoxelMorph × MNI152
</div>
""" % (kicker('Meeting Report', ONDARKMU),
       SERIF, ONDARK, ONDARKMU,
       stat('167', '通過品質檢查的受試者', TEAL, 40),
       stat('0.781', 'Dice（30 個解剖結構）', TEAL, 40),
       stat('0.000%', '形變場折疊率', TEAL, 40),
       MONO, ONDARKMU), bg=INK)

# ── 02 報告重點 ──────────────────────────────────────────────────────
agenda = [
    ('01', '資料從哪來、怎麼把關', '170 個資料夾 → 167 顆可用，每一顆被排除都有依據'),
    ('02', '標籤怎麼搬到 atlas 空間', '影像與標籤必須共用同一個變換，已逐 voxel 驗證'),
    ('03', 'atlas 的標籤是自己做的', 'MNI 官方沒有提供，補零策略讓整條鏈零內插'),
    ('04', '訓練與 Dice 結果', 'λ=1.0 是論文建議值，本專案第一次用對尺度'),
    ('05', '誠實邊界與待確認事項', '有兩件事需要老師確認'),
]
rows = []
for n, t, d in agenda:
    rows.append("""<div style="display: grid; grid-template-columns: 46px 1fr; gap: 20px; align-items: start; padding: 17px 0; border-bottom: 1px solid #2A3237;">
  <div style="font-family: %s; font-size: 15px; color: %s; font-weight: 600; padding-top: 2px;">%s</div>
  <div>
    <div style="font-family: %s; font-size: 20px; font-weight: 700; color: %s; line-height: 1.35;">%s</div>
    <div style="font-size: 13.5px; color: %s; margin-top: 5px; line-height: 1.55;">%s</div>
  </div>
</div>""" % (MONO, TEAL, n, SERIF, ONDARK, t, ONDARKMU, d))
SLIDES['Agenda'] = page("""
<div style="%s">
  %s
  <div style="font-family: %s; font-size: 34px; font-weight: 700; color: %s; margin-top: 8px;">本次報告重點</div>
  <div style="margin-top: 26px;">%s</div>
</div>
""" % (PAD, kicker('Agenda', ONDARKMU), SERIF, ONDARK, ''.join(rows)), bg=INK)

# ── 03 整體流程 ──────────────────────────────────────────────────────
steps = [
    ('01', 'recon-all', '167 顆<br>9 天 17 小時', 'FreeSurfer'),
    ('02', '匯出', 'norm.mgz<br>aseg.mgz', '285 MB'),
    ('03', '前處理', 'Affine 對到<br>MNI152', 'ANTs'),
    ('04', '切分', 'train 149<br>test 17', '受試者層級'),
    ('05', '訓練', 'ncc, λ=1.0<br>250 epochs', 'VoxelMorph'),
    ('06', '評估', 'Dice<br>30 個結構', 'vs atlas aseg'),
]
cards = []
for i, (n, t, d, tag) in enumerate(steps):
    arrow = ('<div style="align-self: center; color: %s; font-size: 18px; padding: 0 2px;">→</div>' % RULE) if i else ''
    cards.append(arrow + """<div style="background: %s; border: 1px solid %s; padding: 15px 13px; display: flex; flex-direction: column; gap: 7px; flex: 1;">
  <div style="font-family: %s; font-size: 11px; color: %s; font-weight: 600;">%s</div>
  <div style="font-family: %s; font-size: 17px; font-weight: 700; color: %s;">%s</div>
  <div style="font-size: 12.5px; color: %s; line-height: 1.5;">%s</div>
  <div style="font-family: %s; font-size: 10.5px; color: %s; margin-top: auto; padding-top: 6px;">%s</div>
</div>""" % (SURFACE, RULE, MONO, TEAL, n, SERIF, INK, t, MUTED, d, MONO, MUTED, tag))

SLIDES['Pipeline'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: flex; gap: 6px; margin-top: 30px; height: 176px;">%s</div>
  <div style="margin-top: 34px;">
    <div style="font-family: %s; font-size: 12px; letter-spacing: 0.16em; color: %s; font-weight: 600; text-transform: uppercase;">每一段都有一個「會安靜出錯」的地方</div>
    <div style="margin-top: 16px;">%s</div>
  </div>
</div>
%s
""" % (PAD, kicker('Overview'), title('整體流程'), ''.join(cards), MONO, RUST,
       bullets([
           (True, '<b>混掃描</b>：資料夾若混了兩次掃描但層數沒超標，<code style="font-family:%s;font-size:13.5px">recon-all</code> 不會報錯 → 已全面掃描，異常 0' % MONO),
           (True, '<b>標籤搬運</b>：內插法用錯會插出不存在的標籤值，不會報錯 → 已逐 voxel 驗證'),
           (True, '<b>座標系轉換</b>：位移在 Dice 上只會像「配準不好」，不會報錯 → 已量到 0.003 voxel'),
       ], gap=11), num(3)))

# ── 04 資料把關 ──────────────────────────────────────────────────────
SLIDES['DataQC'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 380px; gap: 46px; margin-top: 28px;">
    <div>
      %s
    </div>
    <div style="display: flex; flex-direction: column; gap: 26px; padding-top: 4px;">
      %s
      %s
      <div style="border-top: 1px solid %s; padding-top: 18px; font-size: 13px; color: %s; line-height: 1.65;">
        每一顆被排除都有 log 或檔案層級的證據，<br>不是憑印象判斷。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Data Quality'), title('資料把關：170 → 166'),
       table(['受試者', '問題', '處置'],
             [['A043', '雜訊過高、灰白對比不足（白質只認出約 5%）', '排除'],
              ['T085', '只有 120/192 張切片，來源即缺', '排除'],
              ['T065', '資料夾名與 DICOM 病人 ID 不符', '身分待確認'],
              ['A012', '資料夾混了兩次掃描（0801 + A012）', '<span style="color:%s">已修復重跑，納入</span>' % TEAL],
              ['A016_1', '皮質面積僅中位數 54%，表面破洞 83/71', '<span style="color:%s">QC 後排除</span>' % RUST]],
             '92px 1fr 130px'),
       stat('167 × 192 = 32,064', '混掃描檢查：資料夾數 × 切片數與實際檔案總數完全閉合'),
       stat('149 / 17', '最終 train / test（A016_1 於 QC 後移出訓練集）'),
       RULE, MUTED, num(4)))

# ── 05 匯出 norm vs brain ────────────────────────────────────────────
SLIDES['Export'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 28px;">
    <div style="background: %s; border: 1px solid %s; padding: 22px;">
      <div style="font-family: %s; font-size: 15px; font-weight: 600; color: %s;">norm.mgz　<span style="color: %s; font-size: 12px;">採用</span></div>
      <div style="font-size: 14px; color: %s; margin-top: 12px; line-height: 1.7;">
        <code style="font-family: %s">mri_ca_normalize</code> 的產物：<br>用 GCA 圖譜當控制點做亮度正規化，再遮罩。
      </div>
    </div>
    <div style="background: %s; border: 1px solid %s; padding: 22px;">
      <div style="font-family: %s; font-size: 15px; font-weight: 600; color: %s;">brain.mgz　<span style="color: %s; font-size: 12px;">未採用</span></div>
      <div style="font-size: 14px; color: %s; margin-top: 12px; line-height: 1.7;">
        再多做一次 <code style="font-family: %s">mri_normalize -aseg</code>，<br>改用該受試者自己的 aseg 當控制點。
      </div>
    </div>
  </div>
  <div style="margin-top: 30px;">%s</div>
</div>
%s
""" % (PAD, kicker('FreeSurfer Export'), title('匯出哪一個影像：norm 還是 brain'),
       SURFACE, TEAL, SERIF, INK, TEAL, INK, MONO,
       PAPER, RULE, SERIF, MUTED, MUTED, MUTED, MONO,
       bullets([
           (True, '<b>兩者都是合理選擇</b>，目前沒有文獻依據說哪個對配準訓練更好。brain 多一道、更貼身的正規化；norm 少一道、離原始更近。'),
           (True, '<b>關鍵是全體一致</b> —— 167 顆全部是 <code style="font-family:%s;font-size:13.5px">norm.mgz</code>，沒有混用。' % MONO),
           (False, '<b>方法學要寫明用的是 norm.mgz。</b> 若日後配準結果不理想，這是該回頭檢視的變因之一。'),
       ]),
       num(5)))

# ── 06 前處理 ────────────────────────────────────────────────────────
SLIDES['Preprocess'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 430px; gap: 44px; margin-top: 26px;">
    <div>
      <div style="font-size: 15px; color: %s; line-height: 1.75;">
        ANTs Affine 把受試者對到 MNI152 atlas，<b>同一組變換</b>再用最近鄰把 aseg 標籤搬過去。
      </div>
      <div style="margin-top: 22px;">%s</div>
      <div style="margin-top: 24px;">%s</div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 20px; padding-top: 2px;">
      %s
      %s
      %s
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Preprocessing'), title('標籤搬運的三個鐵則'), INK,
       bullets([
           (True, '必須用<b>與影像完全相同</b>的變換 —— 不能各自對位一次'),
           (True, '內插必須用<b>最近鄰</b> —— 用 linear 會在標籤 17 和 10 之間插出 13.5 這種不存在的值'),
           (True, '標籤<b>不正規化、不轉 float</b> —— 保持整數'),
       ]),
       code('reg = ants.registration(fixed=atlas, moving=img,\n'
            '                        type_of_transform=&#39;Affine&#39;)\n'
            'seg_reg = ants.apply_transforms(\n'
            '    fixed=atlas, moving=seg,\n'
            '    transformlist=reg[&#39;fwdtransforms&#39;],   # 重用影像的變換\n'
            '    interpolator=&#39;nearestNeighbor&#39;)      # 標籤必須最近鄰'),
       stat('0 / 8,257,536', '影像與標籤逐 voxel 比對，不一致的數量'),
       stat('1,536,912', '反證：改用 linear 會捏造出原標籤集裡沒有的值', RUST),
       stat('0', '批次 167 顆的失敗數與標籤消失數'),
       num(6)))

# ── 07 前處理驗證圖 ──────────────────────────────────────────────────
SLIDES['VerifySeg'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 340px; gap: 40px; margin-top: 24px;">
    %s
    <div style="display: flex; flex-direction: column; gap: 18px; padding-top: 6px;">
      <div style="font-size: 14px; color: %s; line-height: 1.7;">
        目視只能確認「標籤有沒有貼合影像」。但<b>左右翻轉時兩者仍然完美疊合</b>，
        只是整顆腦鏡像了 —— 那種錯誤在 Dice 上只會看起來像「配準效果不好」。
      </div>
      <div style="border: 1px solid %s; padding: 16px; background: %s;">
        <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">左右配對質心（軸 0）</div>
        <div style="font-family: %s; font-size: 12.5px; color: %s; line-height: 1.9; margin-top: 10px;">
          視丘　　L 84.2 ／ R 106.8　Δ −22.6<br>
          海馬迴　L 69.5 ／ R 123.0　Δ −53.5<br>
          殼核　　L 69.4 ／ R 121.7　Δ −52.3
        </div>
        <div style="font-size: 12.5px; color: %s; margin-top: 12px; line-height: 1.6;">
          7 組配對的 Δ <b style="color: %s">全部同號</b>，方向沒有翻轉。
        </div>
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Verification'), title('目視看不出來的錯誤'),
       figure('seg_transfer.png', 'label overlay check',
              '單顆量化驗證 10/10 通過：標籤 100% 落在影像前景內、44 個標籤一個沒少、質心距離 2.29 voxel。', 400),
       INK, RULE, SURFACE, MONO, MUTED, MONO, INK, MUTED, TEAL, num(7)))

# ── 08 切分 ──────────────────────────────────────────────────────────
SLIDES['Split'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 400px; gap: 44px; margin-top: 26px;">
    <div>
      <div style="font-size: 15px; color: %s; line-height: 1.75;">
        同一個人的多次掃描若一個進 train、一個進 test，模型等於看過答案，Dice 會虛高。
        所以切分以<b>受試者</b>為單位，不是檔案層級 shuffle。
      </div>
      <div style="margin-top: 24px;">%s</div>
      <div style="margin-top: 24px;">%s</div>
    </div>
    <div style="background: %s; border: 1px solid %s; padding: 20px;">
      <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">實際落點</div>
      <div style="font-family: %s; font-size: 13px; color: %s; line-height: 2.0; margin-top: 12px;">
        A013　　→ train<br>
        A0131　 → <span style="color: %s; font-weight: 600;">test</span><br>
        A0132　 → train
      </div>
      <div style="font-size: 13px; color: %s; margin-top: 16px; line-height: 1.65; border-top: 1px solid %s; padding-top: 14px;">
        若三者是同一人，這顆 test 的 Dice 會虛高。<br><br>
        <b style="color: %s">但實測 A0131 是 17 顆裡最低的（0.740）</b>，
        排除它之後平均只差 +0.0026 —— 遠小於標準誤 0.0051。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Data Split'), title('切分與 data leakage 風險'), INK,
       bullets([
           (False, '<b>A013 / A0131 / A0132 是不是同一人？</b> 命名規則無法判斷 —— A0131 可能是「A013 的第 1 次」，也可能是獨立受試者，而且 A013 本身也存在。'),
           (True, '已用程式掃過全部 167 個 ID，<b>這種曖昧全批只有這一組</b>，沒有第二處。'),
       ]),
       stat('149 / 17', 'train / test，無受試者橫跨兩邊'),
       SURFACE, RULE, MONO, MUTED, MONO, INK, RUST, MUTED, RULE, TEAL, num(8)))

# ── 09 atlas 缺 seg ──────────────────────────────────────────────────
SLIDES['AtlasGap'] = page("""
<div style="%s">
  %s
  %s
  <div style="font-size: 15px; color: %s; line-height: 1.75; margin-top: 20px; max-width: 940px;">
    Dice 要把受試者的標籤搬到 atlas 空間，再跟 <b>atlas 自己的標籤</b>比。
    但我們的 MNI152 atlas 是自製的，<b>只有影像沒有標籤</b>。
  </div>
  <div style="margin-top: 26px;">%s</div>
  <div style="margin-top: 26px; border-top: 1px solid %s; padding-top: 20px; display: flex; gap: 14px; align-items: flex-start;">
    %s
    <div style="font-size: 14.5px; color: %s; line-height: 1.7;">
      <b>結論：自己跑 FreeSurfer，跟 167 顆同一套方法。</b>
      這也跟論文一致 —— TMI 2019 §V-A-1 明寫受試者與 atlas 都是 FreeSurfer 分割，沒有用現成的標記圖譜。
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('The Missing Piece'), title('atlas 的標籤從哪來'), INK,
       table(['候選來源', '內容', '為什麼不能用'],
             [['MNI 官方發布包', 'gm / wm / csf 組織機率圖（連續值）+ 三張二值遮罩',
               '<span style="color:%s">沒有解剖結構標籤</span>' % RUST],
              ['CerebrA', 'Mindboggle-101 配準 + 人工修，皮質與皮質下標記',
               '定義在 symmetric 空間；標記協定與 aseg 不同'],
              ['Hammers / LPBA40 / AAL', '30–40 位受試者手工描繪，配準平均成機率圖',
               '標記協定不同 → 差異會混進 Dice'],
              ['G-Node 公開資料', '別人對同一模板跑過 recon-all',
               '<span style="color:%s">未標示 FreeSurfer 版本</span>' % RUST]],
             '190px 1fr 250px'),
       RULE, mark(True), INK, num(9)))

# ── 10 補零 ──────────────────────────────────────────────────────────
SLIDES['ZeroPad'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 400px; gap: 44px; margin-top: 24px;">
    <div>
      <div style="font-size: 14.5px; color: %s; line-height: 1.72;">
        <code style="font-family: %s">recon-all</code> 會把輸入轉成 256³ 並保留影像中心。
        MNI152 的三個邊長都是<b>奇數</b>，中心落在半整數體素上 —— 跟目標的 128.0 差半格，<b>會觸發內插</b>。
      </div>
      <div style="margin-top: 20px;">%s</div>
      <div style="margin-top: 20px; font-size: 14.5px; color: %s; line-height: 1.72;">
        先補零成 256³ 之後三軸中心都是 128.0，<b>偏移為零</b>。
        FreeSurfer 的 log 顯示它仍然跑了三線性內插 —— 但在整數座標上，三線性回傳的就是原值。
      </div>
      <div style="margin-top: 20px;">%s</div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 22px; padding-top: 4px;">
      %s
      %s
      %s
      <div style="border-top: 1px solid %s; padding-top: 16px; font-size: 13px; color: %s; line-height: 1.65;">
        對照：若補零沒生效，這裡會是 <b style="color: %s">0.5 voxel</b>。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Zero Interpolation'), title('讓整條轉換鏈零內插'), INK, MONO,
       code('軸 0　193 → 中心 96.5 　目標 128.0　偏移 −31.5　半整數\n'
            '軸 1　229 → 中心 114.5　目標 128.0　偏移 −13.5　半整數\n'
            '軸 2　193 → 中心 96.5 　目標 128.0　偏移 −31.5　半整數'),
       INK,
       code('256³ ──[31:224, 13:242, 31:224]──> (193,229,193)\n'
            '     ──[0:192,   2:226,  0:192 ]──> atlas 空間\n'
            '兩段都是整數切片，voxel 值一個都沒被動過'),
       stat('0.0032', '共用遮罩的強度加權質心位移（voxel）'),
       stat('0.0001', '相位相關次像素估計，整數峰值在 (0,0,0)'),
       stat('0.999977', '轉回後與原圖的相關係數'),
       RULE, MUTED, RUST, num(10)))

# ── 11 atlas seg 結果 ────────────────────────────────────────────────
SLIDES['AtlasSeg'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 320px; gap: 40px; margin-top: 24px;">
    %s
    <div style="display: flex; flex-direction: column; gap: 20px; padding-top: 6px;">
      %s
      %s
      <div style="border-top: 1px solid %s; padding-top: 16px;">
        <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">品質對照</div>
        <div style="font-size: 13px; color: %s; line-height: 1.75; margin-top: 10px;">
          表面破洞 14 / 12，我們 166 顆的中位數是 12 / 10 —— 幾乎就在中位數上。
          原本擔心平均模板邊界模糊會讓拓撲重建失敗，結果沒有。
        </div>
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Atlas Segmentation'), title('atlas 的 aseg：自行產生'),
       figure('atlas_seg.png', 'atlas aseg overlay',
              'FreeSurfer 7.4.1，與 166 顆受試者同一版本。耗時 2 小時 21 分。', 396),
       stat('30 / 30', '評估用的結構全部都在，缺少 0 個'),
       stat('5 / 5', '切到 atlas 空間後的驗證項目全數通過'),
       RULE, MONO, MUTED, MUTED, num(11)))

# ── 12 訓練設定 ──────────────────────────────────────────────────────
SLIDES['Training'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 26px;">
    <div>
      <div style="font-size: 15px; color: %s; line-height: 1.75;">
        同一個 λ 在 MSE 和 NCC 之間<b>意義完全不同</b>，因為兩者損失量級差兩個數量級。
        論文 Fig. 7 分兩張圖畫，x 軸刻度不一樣。
      </div>
      <div style="margin-top: 22px;">%s</div>
    </div>
    <div>
      <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">先前 IXI 實驗的平滑項佔比</div>
      <div style="margin-top: 12px;">%s</div>
      <div style="margin-top: 18px; font-size: 13.5px; color: %s; line-height: 1.7;">
        <b style="color: %s">所有 NCC 實驗的正則化都形同虛設。</b>
        asd_exp1 是本專案第一次把 λ 放到論文建議的尺度。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Training Setup'), title('λ 的尺度取決於 image-loss'), INK,
       table(['image-loss', '損失量級', '論文最佳 λ', 'train.py 預設'],
             [['MSE', '~0.005', '0.01 – 0.02', '0.01　<span style="color:%s">落在最佳區</span>' % TEAL],
              ['NCC', '~1', '<b>≈ 1 – 2</b>', '0.01　<span style="color:%s">小了兩個數量級</span>' % RUST]],
             '96px 90px 108px 1fr'),
       MONO, MUTED,
       table(['實驗', 'λ', '平滑佔比'],
             [['exp4', '0.005', '1.17%'],
              ['exp5', '0.05', '3.26%'],
              ['exp7', '0.01', '1.27%'],
              ['exp6（MSE）', '0.01', '<span style="color:%s">24.53%%</span>' % TEAL]],
             '110px 66px 1fr', fs=13),
       MUTED, RUST, num(12)))

# ── 13 結果：曲線 ────────────────────────────────────────────────────
SLIDES['Curve'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 330px; gap: 38px; margin-top: 22px;">
    %s
    <div style="display: flex; flex-direction: column; gap: 18px; padding-top: 4px;">
      %s
      %s
      %s
      <div style="border-top: 1px solid %s; padding-top: 14px; font-size: 12.5px; color: %s; line-height: 1.65;">
        17 顆測試的標準誤是 0.0051，所以<b>「最佳 epoch = 190」其實是雜訊</b> ——
        epoch 90 之後任何一個都一樣好。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Results'), title('Dice 曲線與基準線'),
       figure('dice_curve.png', 'dice curve',
              'epoch 0 的 Dice 等於基準線 0.676014 —— 兩個用完全不同程式路徑算出的數字一致，證明評估鏈沒有錯。', 392),
       stat('0.6760', '基準線：只有 Affine，模型無貢獻'),
       stat('0.7811', '模型最佳，比基準線 +0.105'),
       stat('0.0000%', '折疊率，全部 26 個檢查點皆為零'),
       RULE, MUTED, num(13)))

# ── 14 結果：輪廓 ────────────────────────────────────────────────────
SLIDES['Contours'] = page("""
<div style="%s">
  %s
  %s
  <div style="margin-top: 20px;">%s</div>
</div>
%s
""" % (PAD, kicker('Results'), title('結構輪廓：配準前 vs 配準後'),
       figure('contours.png', 'structure contours before and after',
              '實線 = atlas 輪廓，虛線 = 受試者。上排只有 Affine，虛線明顯錯開；下排加上 VoxelMorph 之後幾乎重合。'
              ' 受試者 T023，Dice 0.6894 → 0.7931。', 448),
       num(14)))

# ── 15 結果：逐結構 ──────────────────────────────────────────────────
SLIDES['PerStruct'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 350px; gap: 38px; margin-top: 22px;">
    %s
    <div style="display: flex; flex-direction: column; gap: 18px; padding-top: 4px;">
      <div style="font-size: 14px; color: %s; line-height: 1.72;">
        改善的模式很合理，而且不是調出來的：
      </div>
      %s
      <div style="border-top: 1px solid %s; padding-top: 16px; font-size: 13px; color: %s; line-height: 1.7;">
        <b>腦室改善最多</b> —— 形狀個體差異最大，Affine 對不好，正是非線性配準該解決的。<br><br>
        <b>視丘改善最少</b> —— 位於腦中央、位置穩定，Affine 本來就有 0.84。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Results'), title('30 個結構逐一比較'),
       figure('perstruct.png', 'per-structure dice', '灰 = 配準前，藍 = 配準後，依改善幅度排序。', 400),
       INK,
       table(['結構', '前 → 後', '改善'],
             [['右側腦室', '0.622 → 0.863', '<span style="color:%s">+0.241</span>' % TEAL],
              ['左側腦室', '0.681 → 0.868', '<span style="color:%s">+0.187</span>' % TEAL],
              ['左大腦皮質', '0.550 → 0.719', '+0.169'],
              ['左視丘', '0.840 → 0.874', '+0.035']],
             '92px 118px 1fr', fs=13),
       RULE, MUTED, num(15)))

# ── 16 誠實邊界 ──────────────────────────────────────────────────────
SLIDES['Caveats'] = page("""
<div style="%s">
  %s
  <div style="font-family: %s; font-size: 34px; font-weight: 700; color: %s; margin-top: 8px;">誠實邊界與待確認</div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 30px;">
    <div>
      <div style="font-family: %s; font-size: 11px; letter-spacing: 0.14em; color: %s; font-weight: 600;">要寫進方法學的限制</div>
      <div style="margin-top: 16px; display: flex; flex-direction: column; gap: 15px;">
        %s
      </div>
    </div>
    <div>
      <div style="font-family: %s; font-size: 11px; letter-spacing: 0.14em; color: %s; font-weight: 600;">需要老師確認</div>
      <div style="margin-top: 16px; display: flex; flex-direction: column; gap: 18px;">
        <div style="border: 1px solid #2A3237; padding: 18px; background: #1A2125;">
          <div style="font-family: %s; font-size: 16px; font-weight: 700; color: %s;">A013 / A0131 / A0132 是同一人嗎？</div>
          <div style="font-size: 13.5px; color: %s; margin-top: 8px; line-height: 1.65;">
            目前採「三個不同的人」的暫定假設。這是全批唯一的命名曖昧，確認後改一份設定檔就能重跑切分，不用改程式。
          </div>
        </div>
        <div style="border: 1px solid #2A3237; padding: 18px; background: #1A2125;">
          <div style="font-family: %s; font-size: 16px; font-weight: 700; color: %s;">這批的年齡分布？</div>
          <div style="font-size: 13.5px; color: %s; margin-top: 8px; line-height: 1.65;">
            eTIV 中位數 1.29 L，比健康成人低約 20%%，且厚度異常者全部偏厚 —— 這批可能包含兒童。
            會影響能否與 IXI（20–86 歲成人）對照。
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('What We Cannot Claim', ONDARKMU), SERIF, ONDARK, MONO, ONDARKMU,
       ''.join(
           '<div style="display: flex; gap: 11px; align-items: flex-start;">'
           '<span style="display:inline-block;width:9px;height:9px;border:1.5px solid %s;margin-top:6px;flex:none;"></span>'
           '<div style="font-size: 14px; line-height: 1.65; color: %s;">%s</div></div>' % (RUST, ONDARK, t)
           for t in [
               '<b>系統性天花板</b>：MNI152 是 152 顆腦非線性平均，結構邊界比個體模糊，即使配準完美 Dice 也達不到 1.0。絕對值會低於論文，方法間的相對比較仍有效。',
               '<b>沒有 validation set</b>：epoch 是依 test 曲線挑的，報出的數字會樂觀偏高。論文的做法是另外切一份 validation。',
               '<b>基準線較高</b>：我們的 Affine only 是 0.676，論文是 0.584，所以模型的可進步空間較小。要看的是「比基準線高多少」，不是絕對值。',
           ]),
       MONO, ONDARKMU, SERIF, TEAL, ONDARKMU, SERIF, TEAL, ONDARKMU,
       num(16)), bg=INK)

# ── 寫檔 ─────────────────────────────────────────────────────────────
for name, html in SLIDES.items():
    with io.open(os.path.join(OUT, name + '.dc.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print('  %-14s %6.1f KB' % (name + '.dc.html', len(html.encode('utf-8')) / 1024))

# ── canvas.json ──────────────────────────────────────────────────────
import json
order = ['Main', 'Agenda', 'Pipeline', 'DataQC', 'Export', 'Preprocess', 'VerifySeg',
         'Split', 'AtlasGap', 'ZeroPad', 'AtlasSeg', 'Training', 'Curve', 'Contours',
         'PerStruct', 'Caveats']
GAP_X, GAP_Y, COLS = 160, 220, 4
boards = []
for i, n in enumerate(order):
    boards.append({'file': n + '.dc.html',
                   'x': (i % COLS) * (W + GAP_X),
                   'y': (i // COLS) * (H + GAP_Y),
                   'w': W, 'h': H})
canvas = {'artboards': boards, 'launch': {'view': 'focused', 'file': 'Main.dc.html'}}
with io.open(os.path.join(OUT, 'canvas.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(canvas, indent=2, ensure_ascii=False))
print('\n  canvas.json  %d 個投影片，%d 欄排列' % (len(order), COLS))
