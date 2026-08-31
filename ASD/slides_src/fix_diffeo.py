# -*- coding: utf-8 -*-
"""重寫「微分同胚」那一頁。

原版的問題：直接從 int_steps=7 講起，沒有先建立「折疊是什麼、
為什麼位移場會折疊」，而 φ = exp(v) 對沒碰過的人等於天書。

改成三段推進：
    ① 折疊是什麼（用格線圖，箭頭交叉 vs 不交叉）
    ② 位移場為什麼會折疊（一步到位，兩個點可能落到同一處）
    ③ 速度場怎麼避免（分成很多小步，每步都小到不可能翻過去）
"""
import io
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v2', 'VxmDiffeo.dc.html')

INK, PAPER, SURFACE, RULE = '#141A1D', '#FAFAF8', '#EFEFEB', '#D9D9D2'
MUTED, TEAL, RUST = '#5F6A6B', '#0E7C7B', '#A34F1B'
SERIF = "'Noto Serif TC', 'Songti TC', serif"
SANS = "'Noto Sans TC', 'PingFang TC', sans-serif"
MONO = "'IBM Plex Mono', 'Courier New', monospace"


def arrows_svg(cross):
    """一維示意：上排是原始位置，下排是搬過去之後。

    cross=True  箭頭交叉 -> 兩點的先後順序被翻轉 = 折疊
    cross=False 箭頭不交叉 -> 順序保持 = 沒有折疊
    """
    xs = [30, 70, 110, 150, 190, 230]
    if cross:
        # 第 3、4 個點互換位置 —— 這就是折疊
        tgt = [26, 66, 152, 104, 196, 236]
    else:
        tgt = [24, 62, 104, 152, 196, 238]
    col = RUST if cross else TEAL
    parts = ['<svg viewBox="0 0 260 100" style="width: 100%; height: auto; display: block;">']
    # 上下兩條基線
    parts.append('<line x1="10" y1="20" x2="250" y2="20" stroke="%s" stroke-width="1"/>' % RULE)
    parts.append('<line x1="10" y1="82" x2="250" y2="82" stroke="%s" stroke-width="1"/>' % RULE)
    for x, t in zip(xs, tgt):
        parts.append('<circle cx="%d" cy="20" r="3.5" fill="%s"/>' % (x, MUTED))
        parts.append('<line x1="%d" y1="26" x2="%d" y2="76" stroke="%s" stroke-width="1.6" '
                     'opacity="0.85"/>' % (x, t, col))
        parts.append('<circle cx="%d" cy="82" r="3.5" fill="%s"/>' % (t, col))
    parts.append('</svg>')
    return ''.join(parts)


CROSS = arrows_svg(True)
NOCROSS = arrows_svg(False)

html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@500;700&amp;family=Noto+Sans+TC:wght@400;500;700&amp;family=IBM+Plex+Mono:wght@400;600&amp;display=swap">
  <style>
    body {{ margin: 0; }}
    * {{ box-sizing: border-box; }}
    a {{ color: {TEAL}; text-decoration: none; }}
    a:hover {{ color: {RUST}; }}
  </style>
</helmet>
<div style="width: 1280px; height: 720px; background: {PAPER}; font-family: {SANS}; position: relative; overflow: hidden;">
<div style="position: absolute; left: 56px; top: 40px; right: 56px;">

  <div style="font-family: {MONO}; font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase; color: {MUTED}; font-weight: 600;">How VoxelMorph Works　3 / 3</div>
  <div style="font-family: {SERIF}; font-size: 35px; font-weight: 700; color: {INK}; line-height: 1.22; margin-top: 6px;">「折疊」是什麼，怎麼避免</div>

  <div style="font-size: 14.5px; color: {INK}; line-height: 1.65; margin-top: 11px; max-width: 1010px;">
    形變場要把受試者的每個位置搬到 atlas 的對應位置。
    <b>折疊</b>＝兩塊組織被搬到同一個地方，或前後順序被翻轉 —— 解剖上不可能發生的變形。
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 15px;">

    <div style="border: 1px solid {RUST}; background: {SURFACE}; padding: 13px 16px;">
      <div style="font-family: {MONO}; font-size: 11px; letter-spacing: 0.1em; color: {RUST}; font-weight: 600;">位移場　一步到位</div>
      <div style="font-size: 13.5px; color: {INK}; line-height: 1.58; margin-top: 7px;">
        直接說「這個點搬到那裡」。<b>各點各自搬，互不相干</b> ——
        所以兩條路徑可能交叉，兩個點可能落到同一處。
      </div>
      <div style="margin-top: 9px;">{CROSS}</div>
      <div style="font-size: 12.5px; color: {RUST}; line-height: 1.5; margin-top: 4px;">
        中間兩個點的順序被翻轉了 —— 這就是折疊。
      </div>
    </div>

    <div style="border: 1px solid {TEAL}; background: {SURFACE}; padding: 13px 16px;">
      <div style="font-family: {MONO}; font-size: 11px; letter-spacing: 0.1em; color: {TEAL}; font-weight: 600;">速度場　分成很多小步</div>
      <div style="font-size: 13.5px; color: {INK}; line-height: 1.58; margin-top: 7px;">
        改成說「這個點往哪個方向流」，像水流一樣。
        <b>順著流走，每一步都極小</b> —— 小到不可能翻過旁邊的點。
      </div>
      <div style="margin-top: 9px;">{NOCROSS}</div>
      <div style="font-size: 12.5px; color: {TEAL}; line-height: 1.5; margin-top: 4px;">
        順序始終保持 —— 不會折疊。
      </div>
    </div>

  </div>

  <div style="display: grid; grid-template-columns: 1fr 430px; gap: 38px; margin-top: 18px; border-top: 1px solid {RULE}; padding-top: 16px;">
    <div>
      <div style="font-family: {MONO}; font-size: 11px; letter-spacing: 0.12em; color: {MUTED}; font-weight: 600;">怎麼「分成很多小步」——scaling and squaring</div>
      <div style="font-size: 14px; color: {INK}; line-height: 1.68; margin-top: 8px;">
        <code style="font-family: {MONO}">int_steps=7</code> 的意思是分成 <b>2⁷ = 128 步</b>。
        但不用真的算 128 次 —— 先把整個速度場縮小 128 倍（每一步都極小），
        然後<b>自我複合 7 次</b>，每複合一次步數就翻倍：1 → 2 → 4 → … → 128。
      </div>
      <div style="font-family: {MONO}; font-size: 12.5px; background: {SURFACE}; border: 1px solid {RULE}; padding: 10px 12px; color: {INK}; line-height: 1.62; white-space: pre; margin-top: 10px;">vec = vec / 128            # 縮小成極小的一步
for _ in range(7):         # 複合 7 次，步數翻 7 倍
    vec = vec + warp(vec, vec)</code></div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 12px;">
      <div style="display: flex; flex-direction: column; gap: 3px;">
        <div style="font-family: {MONO}; font-size: 34px; font-weight: 600; color: {TEAL}; line-height: 1;">0.0000%</div>
        <div style="font-size: 13px; color: {MUTED}; line-height: 1.4;">asd_exp1 全部 26 個檢查點的折疊率</div>
      </div>
      <div style="border: 1px solid {RULE}; background: {SURFACE}; padding: 12px 14px;">
        <div style="font-size: 12.5px; color: {INK}; line-height: 1.62;">
          <b style="color: {RUST}">要講清楚的落差：</b>
          論文 Table I 的主結果是<b>非</b>微分同胚版本（<code style="font-family: {MONO}">int_steps=0</code>，純位移場）。
          repo 預設的 7 來自後續的機率式微分同胚版本 —— 所以「跑 repo 預設」跟「複現論文表格」不是同一件事。
        </div>
      </div>
    </div>
  </div>

</div>
<div style="position: absolute; right: 56px; bottom: 34px; font-family: {MONO}; font-size: 12px; color: {MUTED};">16</div>
</div>
</x-dc>
</body>
</html>
""".format(**locals())

io.open(OUT, 'w', encoding='utf-8').write(html)
print('已重寫 %s（%.1f KB）' % (os.path.basename(OUT), len(html.encode('utf-8')) / 1024))
