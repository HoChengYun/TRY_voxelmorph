# -*- coding: utf-8 -*-
"""擴充簡報：Affine 細節、MNI152 前處理、VoxelMorph 原理、更多結果圖。

⚠️ 這支腳本是「加法」的：
    · 既有投影片從 slides_live/（線上最新版）複製過來，內容不動
      —— 使用者在畫布上改過 Main 的標題與頁腳，不能被蓋掉
    · 只有新投影片由這裡產生
    · 頁碼統一重算（唯一會動到既有檔案的地方）
"""
import io
import os
import re
import json
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(os.path.dirname(HERE), 'slides_live')
OUT = os.path.join(HERE, 'v2')
W, H = 1280, 720

INK, PAPER, SURFACE, RULE = '#141A1D', '#FAFAF8', '#EFEFEB', '#D9D9D2'
MUTED, TEAL, RUST = '#5F6A6B', '#0E7C7B', '#A34F1B'
ONDARK, ONDARKMU = '#E8E6E0', '#8A9294'

SERIF = "'Noto Serif TC', 'Songti TC', serif"
SANS = "'Noto Sans TC', 'PingFang TC', sans-serif"
MONO = "'IBM Plex Mono', 'Courier New', monospace"

HELMET = """<helmet>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@500;700&amp;family=Noto+Sans+TC:wght@400;500;700&amp;family=IBM+Plex+Mono:wght@400;600&amp;display=swap">
  <style>
    body { margin: 0; }
    * { box-sizing: border-box; }
    a { color: %s; text-decoration: none; }
    a:hover { color: %s; }
  </style>
</helmet>""" % (TEAL, RUST)

PAD = 'position: absolute; left: 56px; top: 48px; right: 56px;'


def page(body, bg=PAPER):
    return ('<!doctype html>\n<html>\n<head>\n  <meta charset="utf-8">\n'
            '  <script src="./support.js"></script>\n</head>\n<body>\n<x-dc>\n%s\n'
            '<div style="width: %dpx; height: %dpx; background: %s; font-family: %s; '
            'position: relative; overflow: hidden;">\n%s\n</div>\n</x-dc>\n</body>\n</html>\n'
            % (HELMET, W, H, bg, SANS, body))


def kicker(t, c=MUTED):
    return ('<div style="font-family: %s; font-size: 12px; letter-spacing: 0.18em; '
            'text-transform: uppercase; color: %s; font-weight: 600;">%s</div>' % (MONO, c, t))


def title(t, c=INK, s=40):
    return ('<div style="font-family: %s; font-size: %dpx; font-weight: 700; color: %s; '
            'line-height: 1.22; margin-top: 10px; text-wrap: pretty;">%s</div>' % (SERIF, s, c, t))


def stat(v, l, c=TEAL, s=44):
    return ('<div style="display: flex; flex-direction: column; gap: 4px;">'
            '<div style="font-family: %s; font-size: %dpx; font-weight: 600; color: %s; '
            'line-height: 1;">%s</div><div style="font-size: 13px; color: %s; '
            'line-height: 1.4;">%s</div></div>' % (MONO, s, c, v, MUTED, l))


def mark(ok=True):
    c = TEAL if ok else RUST
    return ('<span style="display: inline-block; width: 9px; height: 9px; background: %s; '
            'border: 1.5px solid %s; margin-right: 9px; flex: none; margin-top: 6px;"></span>'
            % (c if ok else 'transparent', c))


def bullets(items, gap=14, fs=15, color=INK):
    rows = ['<div style="display: flex; align-items: flex-start;">%s<div style="font-size: %dpx; '
            'line-height: 1.62; color: %s; text-wrap: pretty;">%s</div></div>' % (mark(ok), fs, color, t)
            for ok, t in items]
    return ('<div style="display: flex; flex-direction: column; gap: %dpx;">%s</div>'
            % (gap, ''.join(rows)))


def table(head, rows, widths, fs=14):
    ths = ''.join('<div style="font-size: 11.5px; letter-spacing: 0.08em; color: %s; '
                  'font-weight: 700; text-transform: uppercase; font-family: %s;">%s</div>'
                  % (MUTED, MONO, h) for h in head)
    out = ['<div style="display: grid; grid-template-columns: %s; gap: 10px 18px; align-items: start; '
           'padding-bottom: 10px; border-bottom: 1px solid %s;">%s</div>' % (widths, RULE, ths)]
    for r in rows:
        tds = ''.join('<div style="font-size: %dpx; line-height: 1.55; color: %s;">%s</div>'
                      % (fs, INK, c) for c in r)
        out.append('<div style="display: grid; grid-template-columns: %s; gap: 10px 18px; '
                   'align-items: start; padding: 11px 0; border-bottom: 1px solid %s;">%s</div>'
                   % (widths, RULE, tds))
    return '<div style="display: flex; flex-direction: column;">%s</div>' % ''.join(out)


def figure(src, alt, cap, h=470):
    return ('<div style="display: flex; flex-direction: column; gap: 10px;">'
            '<div style="background: %s; border: 1px solid %s; padding: 12px; display: flex; '
            'justify-content: center;"><img src="%s" alt="%s" style="max-height: %dpx; '
            'max-width: 100%%; object-fit: contain; display: block;"></div>'
            '<div style="font-size: 12.5px; color: %s; line-height: 1.5;">%s</div></div>'
            % (SURFACE, RULE, src, alt, h, MUTED, cap))


def code(t, fs=13):
    return ('<div style="font-family: %s; font-size: %dpx; background: %s; border: 1px solid %s; '
            'padding: 12px 14px; color: %s; line-height: 1.7; white-space: pre;">%s</div>'
            % (MONO, fs, SURFACE, RULE, INK, t))


NEW = {}

# ── MNI152 → atlas ───────────────────────────────────────────────────
NEW['AtlasBuild'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 400px; gap: 42px; margin-top: 24px;">
    <div>
      <div style="font-size: 14.5px; color: %s; line-height: 1.7;">
        下載 <b>ICBM 2009c Nonlinear Asymmetric</b>（Fonov et al., 2009）——152 位受試者反覆迭代非線性平均，
        <code style="font-family: %s">(193, 229, 193)</code>、精確 1mm、RAS。
      </div>
      <div style="margin-top: 20px;">%s</div>
      <div style="margin-top: 20px;">%s</div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 18px; padding-top: 2px;">
      %s
      %s
      <div style="border-top: 1px solid %s; padding-top: 16px; font-size: 13px; color: %s; line-height: 1.68;">
        <b style="color: %s">為什麼要改尺寸</b><br>
        U-Net 有 4 層下採樣，三個維度都要能被 16 整除。<br>
        193÷16 = 12.06 ✗　229÷16 = 14.31 ✗<br>
        192÷16 = 12 ✓　　224÷16 = 14 ✓
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Atlas Preparation'), title('MNI152 怎麼變成訓練用的 atlas'), INK, MONO,
       table(['步驟', '做什麼', '實測'],
             [['① 去顱骨', '套 MNI 官方附的二值腦罩（ANTs 只做相乘，沒有判斷）',
               '非零 8,529,878 → 1,886,574'],
              ['② clip', '1–99 百分位，去掉極端值', 'p1 = 26.09　p99 = 91.47'],
              ['③ 正規化', 'min-max 到 [0, 1]', '—'],
              ['④ 裁切', '(193,229,193) → (192,224,192)',
               '<span style="color:%s">切掉部分最大值 0.0000</span>' % TEAL],
              ['⑤ 存檔', '.nii.gz 帶 header 給 ANTs／.npz 給 train.py', '—']],
             '82px 1fr 190px', fs=13.5),
       bullets([
           (True, '<b>三個軸切掉的部分最大值都是 0</b> —— 全是背景，一個腦組織 voxel 都沒動到。'),
           (True, '用 <code style="font-family:%s;font-size:13.5px">crop</code> 而不是 resample：<b>spacing 維持精確 1mm</b>，voxel 值零損失。resample 會把體素拉長成 ≈1.005mm。' % MONO),
       ], gap=12, fs=14),
       stat('0', '重現時與現存 atlas 的逐 voxel 差異'),
       stat('96.0 → 134.0', '軸 1 的 origin 隨裁切位移（切掉前面 2 個 voxel）'),
       RULE, MUTED, INK, '__NUM__'))

# ── Affine 做了什麼 ──────────────────────────────────────────────────
NEW['AffineWhat'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 22px;">
    <div>
      <div style="font-size: 14.5px; color: %s; line-height: 1.7;">
        <code style="font-family: %s">ants.registration(type_of_transform='Affine')</code>
        —— <b>12 個自由度</b>，三個軸的縮放可以各自不同（非等向）。
      </div>
      <div style="margin-top: 18px;">%s</div>
      <div style="margin-top: 18px;">%s</div>
    </div>
    <div>
      <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">受試者 A001 的實際變換</div>
      <div style="margin-top: 12px;">%s</div>
      <div style="margin-top: 18px;">%s</div>
      <div style="margin-top: 16px; font-size: 13px; color: %s; line-height: 1.68; border-top: 1px solid %s; padding-top: 14px;">
        <b style="color: %s">⚠️ 配準後不能拿來量真實體積。</b>
        voxel 一直是 1mm³，改變的是腦蓋住幾格 —— 這個人的腦沒有變大，是被拉去貼合 atlas 的。
        FreeSurfer 報的 eTIV、海馬體積都是在<b>原生空間</b>算的，那才是真實的。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Linear Registration'), title('線性 Affine 到底做了哪些動作'), INK, MONO,
       table(['自由度', '動作', 'A001 實測'],
             [['平移 ×3', '移位置', '[−2.1, +14.3, −11.4] mm'],
              ['旋轉 ×3', '轉角度', '[−17.8°, +0.4°, 0.0°]'],
              ['縮放 ×3', '改大小（三軸各自獨立）', '見右'],
              ['剪切 ×3', '傾斜', '—']],
             '82px 1fr 178px', fs=13.5),
       bullets([
           (True, '<b>前處理只做線性，非線性刻意留給 VoxelMorph 學。</b> 如果前處理就把腦扭到跟 atlas 一模一樣，模型就沒東西可學了。'),
           (True, 'ANTs 底層是 ITK / C++；metric 用 Mattes 互資訊（32 bins、隨機取樣 20%），四層金字塔 6×4×2×1。'),
       ], gap=12, fs=14),
       MONO, MUTED,
       code('腦組織體積（兩邊都是 1mm 等向）\n\n'
            '　　　　配準前　　　配準後　　　倍率\n'
            'A001　1,432,053　2,162,512　　1.510\n'
            'A003　1,640,622　2,156,836　　1.315\n'
            'T023　1,434,004　2,153,835　　1.502', 12.5),
       stat('×1.44', '三顆平均的體積倍率 —— 受試者被放大去貼合 atlas'),
       MUTED, RULE, RUST, '__NUM__'))

# ── VoxelMorph 架構 ──────────────────────────────────────────────────
NEW['VxmArch'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 420px; gap: 42px; margin-top: 22px;">
    <div>
      <div style="font-size: 14.5px; color: %s; line-height: 1.7;">
        傳統配準（ANTs SyN）對<b>每一對</b>影像都跑一次迭代優化，一對要幾十分鐘。
        VoxelMorph 改成訓練一個 U-Net 直接輸出形變場 —— 論文稱為 <i>amortized optimization</i>：
        把每對各自優化的成本，攤提到一次性的訓練裡。
      </div>
      <div style="margin-top: 20px;">%s</div>
      <div style="margin-top: 20px;">%s</div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 18px; padding-top: 2px;">
      %s
      %s
      <div style="border-top: 1px solid %s; padding-top: 16px; font-size: 13px; color: %s; line-height: 1.7;">
        <b style="color: %s">關鍵：取樣這一步可微分</b><br>
        SpatialTransformer 用 <code style="font-family: %s">grid_sample</code> 取值，
        誤差能反傳回 U-Net —— <b>所以沒有 ground truth 形變場也能訓練</b>。
        這是整篇論文的技術核心。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('How VoxelMorph Works　1 / 3'), title('架構：U-Net 直接輸出形變場'), INK,
       code('輸入　moving + fixed 串成 2 channel\n'
            '　↓\n'
            'U-Net　encoder [16, 32, 32, 32]\n'
            '　　　　decoder [32, 32, 32, 32, 32, 16, 16]\n'
            '　↓\n'
            '輸出　3 channel = 每個 voxel 的位移向量 u(p)\n'
            '　↓\n'
            '形變場　φ = Id + u\n'
            '　↓\n'
            'SpatialTransformer：到 moving 的 p + u(p) 取值', 12.5),
       bullets([
           (True, '取樣位置通常不是整數座標 → 要內插。<b>影像用線性、標籤必須用最近鄰。</b>'),
       ], gap=12, fs=14),
       stat('< 1 秒', '訓練完之後對新影像推論的時間'),
       stat('數十分鐘', '傳統方法對每一對影像的迭代優化', RUST),
       RULE, MUTED, INK, MONO, '__NUM__'))

# ── 損失函數 ─────────────────────────────────────────────────────────
NEW['VxmLoss'] = page("""
<div style="%s">
  %s
  %s
  <div style="margin-top: 20px;">%s</div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 24px;">
    <div>
      <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">L_sim　影像相似度</div>
      <div style="margin-top: 12px; font-size: 14px; color: %s; line-height: 1.72;">
        兩種選擇：<br><br>
        <b>局部 CC</b>（論文 Eq. 6）—— 在每個 voxel 周圍 <b>9³</b> 的窗格裡算相關係數的平方，再取負。
        對亮度差異魯棒。<br><br>
        <b>MSE</b> —— 逐 voxel 平方差。適用於亮度分布相近的影像。
      </div>
    </div>
    <div>
      <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">L_smooth　形變場平滑度</div>
      <div style="margin-top: 12px; font-size: 14px; color: %s; line-height: 1.72;">
        位移場空間梯度的 L2（論文 Eq. 7）：<br><br>
        <code style="font-family: %s; font-size: 15px;">Σ ‖∇u(p)‖²</code><br><br>
        沒有它的話，網路會把影像扭到完全一樣，但形變場亂七八糟、不符合解剖。
      </div>
    </div>
  </div>
  <div style="margin-top: 26px; border-top: 1px solid %s; padding-top: 18px;">%s</div>
</div>
%s
""" % (PAD, kicker('How VoxelMorph Works　2 / 3'), title('損失函數：沒有標準答案也能訓練'),
       code('L = L_sim( f , m∘φ )  +  λ · L_smooth( φ )        ← 論文 Eq. 4\n'
            '     └─ warp 完之後跟 fixed 像不像      └─ 形變場平不平滑', 15),
       MONO, MUTED, INK, MONO, MUTED, INK, MONO,
       RULE,
       bullets([
           (True, '<b>訓練完全不需要 ground truth 形變場</b> —— 損失只看「warp 完像不像」，而這個相似度對形變場可微。'),
           (True, '<b>λ 就是這兩項的平衡</b>。它的合適尺度取決於用哪個 L_sim —— 這正是下一頁要講的。'),
       ], gap=12, fs=14.5),
       '__NUM__'))

# ── 微分同胚 ─────────────────────────────────────────────────────────
NEW['VxmDiffeo'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 420px; gap: 42px; margin-top: 22px;">
    <div>
      <div style="font-size: 14.5px; color: %s; line-height: 1.7;">
        打開 <code style="font-family: %s">int_steps=7</code> 之後，
        網路輸出的<b>不再是位移場，而是速度場 v</b>，再用 <b>scaling and squaring</b> 積分成形變場。
      </div>
      <div style="margin-top: 18px;">%s</div>
      <div style="margin-top: 18px; font-size: 14px; color: %s; line-height: 1.7;">
        原理是 φ = exp(v)。小位移下 exp(v/2ⁿ) ≈ Id + v/2ⁿ，再平方 n 次還原 ——
        這樣得到的形變場<b>理論上保證可逆、不折疊</b>。
      </div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 18px; padding-top: 2px;">
      %s
      <div style="border: 1px solid %s; padding: 16px; background: %s;">
        <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">一個要講清楚的落差</div>
        <div style="font-size: 13px; color: %s; line-height: 1.7; margin-top: 10px;">
          <b>論文 Table I 的主結果是「非微分同胚」版本</b>（純位移場，<code style="font-family: %s">int_steps=0</code>）。
          repo 預設的 <code style="font-family: %s">int_steps=7</code> 來自 Dalca 等人的機率式微分同胚版本。<br><br>
          所以「跑 repo 預設」跟「複現論文表格」<b>嚴格說不是同一件事</b>。
        </div>
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('How VoxelMorph Works　3 / 3'), title('微分同胚：讓形變場不折疊'), INK, MONO,
       code('vec = vec * (1 / 2**7)          # 先縮小 128 倍\n'
            'for _ in range(7):              # 自我複合 7 次\n'
            '    vec = vec + transformer(vec, vec)', 13),
       INK,
       stat('0.0000%', 'asd_exp1 全部 26 個檢查點的折疊率'),
       RULE, SURFACE, MONO, RUST, INK, MONO, MONO, '__NUM__'))

# ── 結果圖 ───────────────────────────────────────────────────────────
NEW['ResultTriplanar'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: grid; grid-template-columns: 1fr 300px; gap: 36px; margin-top: 22px;">
    %s
    <div style="display: flex; flex-direction: column; gap: 18px; padding-top: 4px;">
      %s
      %s
      <div style="border-top: 1px solid %s; padding-top: 14px; font-size: 13px; color: %s; line-height: 1.68;">
        四格由左至右：受試者原影像、atlas、配準後、差異圖。
        差異圖越暗表示對得越好。
      </div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Results'), title('三平面視圖：配準後 vs atlas'),
       figure('triplanar.png', 'triplanar view', '受試者 T023，asd_exp1 epoch 190。', 404),
       stat('0.9750', 'NCC（全域相關，含背景）'),
       stat('0.9198', 'SSIM（結構相似度）'),
       RULE, MUTED, '__NUM__'))

NEW['ResultLabels'] = page("""
<div style="%s">
  %s
  %s
  <div style="margin-top: 18px;">%s</div>
</div>
%s
""" % (PAD, kicker('Results'), title('標籤重疊：Dice 實際在算什麼'),
       figure('labels.png', 'label overlap',
              '<b style="color: %s">紅</b> = 只有 atlas 有（漏掉了）　'
              '<b style="color: %s">綠</b> = 只有受試者有（多出來了）　'
              '<b>黃</b> = 兩者都有（對上了）。Dice 就是在算黃色佔的比例。'
              '上排只有 Affine，邊緣一圈明顯的紅綠；下排幾乎全黃，只剩皮質腦溝還有殘留 ——'
              '那正是個體差異最大、配準最難的地方。' % ('#c62828', '#2e7d32'), 452),
       '__NUM__'))

NEW['ResultChecker'] = page("""
<div style="%s">
  %s
  %s
  <div style="display: flex; flex-direction: column; gap: 22px; margin-top: 22px;">
    <div>
      <div style="font-family: %s; font-size: 11.5px; letter-spacing: 0.12em; color: %s; font-weight: 600; margin-bottom: 8px;">Checkerboard —— 交替方格分別來自兩張影像</div>
      <div style="background: %s; border: 1px solid %s; padding: 10px; display: flex; justify-content: center;">
        <img src="checker.png" alt="checkerboard" style="max-height: 205px; max-width: 100%%; object-fit: contain; display: block;">
      </div>
    </div>
    <div>
      <div style="font-family: %s; font-size: 11.5px; letter-spacing: 0.12em; color: %s; font-weight: 600; margin-bottom: 8px;">Deformation Grid —— 模型實際做的變形</div>
      <div style="background: %s; border: 1px solid %s; padding: 10px; display: flex; justify-content: center;">
        <img src="grid.png" alt="deformation grid" style="max-height: 205px; max-width: 100%%; object-fit: contain; display: block;">
      </div>
    </div>
  </div>
  <div style="margin-top: 16px; font-size: 13px; color: %s; line-height: 1.6;">
    棋盤格看<b>方格交界處腦輪廓有沒有接續</b>——對準了會像一顆完整的腦。
    網格應該是<b>平滑的彎曲</b>，不該有打結、交叉或擠成一團的地方。
  </div>
</div>
%s
""" % (PAD, kicker('Results'), title('棋盤格與形變網格'),
       MONO, MUTED, SURFACE, RULE, MONO, MUTED, SURFACE, RULE, MUTED, '__NUM__'))

NEW['ResultJacobian'] = page("""
<div style="%s">
  %s
  %s
  <div style="background: %s; border: 1px solid %s; padding: 12px; display: flex; justify-content: center; margin-top: 22px;">
    <img src="jacobian.png" alt="jacobian determinant map" style="max-height: 232px; max-width: 100%%; object-fit: contain; display: block;">
  </div>
  <div style="display: grid; grid-template-columns: 1fr 420px; gap: 40px; margin-top: 24px;">
    <div>
      <div style="font-size: 14px; color: %s; line-height: 1.7;">
        Jacobian determinant 描述形變場在每個 voxel 的局部性質：
        <b>&gt; 1 是擴張、&lt; 1 是收縮、≤ 0 代表空間被翻摺</b>。
        折疊表示解剖上不可能的變形 —— 兩塊組織被疊到同一個位置。
      </div>
      <div style="margin-top: 18px;">%s</div>
    </div>
    <div>
      <div style="font-family: %s; font-size: 11px; letter-spacing: 0.12em; color: %s; font-weight: 600;">論文 Table I 的折疊率</div>
      <div style="margin-top: 12px;">%s</div>
    </div>
  </div>
</div>
%s
""" % (PAD, kicker('Results'), title('Jacobian：形變場有沒有折疊'), SURFACE, RULE, INK,
       bullets([
           (True, '<b>只看 Dice 會被騙</b> —— 亂折疊也可以把 Dice 衝高，所以要同時報折疊率。'),
       ], gap=10, fs=14),
       MONO, MUTED,
       table(['方法', 'Dice', '%|J|≤0'],
             [['ANTs SyN (CC)', '0.749', '0.185%'],
              ['NiftyReg (CC)', '0.755', '0.793%'],
              ['VoxelMorph (CC)', '0.753', '0.366%'],
              ['<b>本專案 asd_exp1</b>', '<b>0.781</b>',
               '<b style="color:%s">0.000%%</b>' % TEAL]],
             '1fr 62px 74px', fs=13),
       '__NUM__'))

# ── Agenda 改寫（原本沒有使用者編輯）────────────────────────────────
agenda = [
    ('01', '資料從哪來、怎麼把關', '170 個資料夾 → 167 顆可用，每一顆被排除都有依據'),
    ('02', 'atlas 與線性對位', 'MNI152 怎麼變成 atlas，Affine 到底做了哪些動作'),
    ('03', '標籤怎麼搬到 atlas 空間', '影像與標籤必須共用同一個變換，已逐 voxel 驗證'),
    ('04', 'VoxelMorph 的原理', '架構、損失函數、微分同胚積分'),
    ('05', '訓練與 Dice 結果', 'λ=1.0 是論文建議值，本專案第一次用對尺度'),
    ('06', '誠實邊界與待確認事項', '有兩件事需要老師確認'),
]
rows = ''.join(
    '<div style="display: grid; grid-template-columns: 46px 1fr; gap: 20px; align-items: start; '
    'padding: 14px 0; border-bottom: 1px solid #2A3237;">'
    '<div style="font-family: %s; font-size: 15px; color: %s; font-weight: 600; padding-top: 2px;">%s</div>'
    '<div><div style="font-family: %s; font-size: 19px; font-weight: 700; color: %s; line-height: 1.35;">%s</div>'
    '<div style="font-size: 13px; color: %s; margin-top: 4px; line-height: 1.55;">%s</div></div></div>'
    % (MONO, TEAL, n, SERIF, ONDARK, t, ONDARKMU, d) for n, t, d in agenda)
NEW['Agenda'] = page("""
<div style="%s">
  %s
  <div style="font-family: %s; font-size: 34px; font-weight: 700; color: %s; margin-top: 8px;">本次報告重點</div>
  <div style="margin-top: 22px;">%s</div>
</div>
""" % (PAD, kicker('Agenda', ONDARKMU), SERIF, ONDARK, rows), bg=INK)

# ═════════════════════════════════════════════════════════════════════
ORDER = ['Main', 'Agenda', 'Pipeline', 'DataQC', 'Export',
         'AtlasBuild', 'AffineWhat', 'Preprocess', 'VerifySeg', 'Split',
         'AtlasGap', 'ZeroPad', 'AtlasSeg',
         'VxmArch', 'VxmLoss', 'VxmDiffeo', 'Training',
         'Curve', 'ResultTriplanar', 'Contours', 'ResultLabels',
         'ResultChecker', 'ResultJacobian', 'PerStruct', 'Caveats']

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)

NUM_RE = re.compile(
    r'<div style="position: absolute; right: 56px; bottom: 34px; font-family: [^"]*; '
    r'font-size: 12px; color: [^"]*;">\d+</div>')


def numdiv(n):
    return ('<div style="position: absolute; right: 56px; bottom: 34px; font-family: %s; '
            'font-size: 12px; color: %s;">%02d</div>' % (MONO, MUTED, n))


print('  來源　　　　　　　頁碼')
for i, name in enumerate(ORDER, 1):
    f = name + '.dc.html'
    if name in NEW:
        html = NEW[name].replace('__NUM__', numdiv(i) if name not in ('Main', 'Agenda') else '')
        src = '新增' if name != 'Agenda' else '改寫'
    else:
        html = io.open(os.path.join(LIVE, f), encoding='utf-8').read()
        if NUM_RE.search(html):
            html = NUM_RE.sub(numdiv(i), html)
        src = '線上版'
    io.open(os.path.join(OUT, f), 'w', encoding='utf-8').write(html)
    print('  %-6s %-20s %02d' % (src, f, i))

for img in ('seg_transfer.png', 'atlas_seg.png', 'dice_curve.png', 'contours.png',
            'perstruct.png', 'triplanar.png', 'checker.png', 'grid.png',
            'jacobian.png', 'labels.png'):
    shutil.copy(os.path.join(HERE, img), os.path.join(OUT, img))

GAP_X, GAP_Y, COLS = 160, 220, 5
boards = [{'file': n + '.dc.html', 'x': (i % COLS) * (W + GAP_X),
           'y': (i // COLS) * (H + GAP_Y), 'w': W, 'h': H}
          for i, n in enumerate(ORDER)]
io.open(os.path.join(OUT, 'canvas.json'), 'w', encoding='utf-8').write(
    json.dumps({'artboards': boards, 'launch': {'view': 'focused', 'file': 'Main.dc.html'}},
               indent=2, ensure_ascii=False))
print('\n  共 %d 頁，%d 欄排列 -> %s' % (len(ORDER), COLS, OUT))
