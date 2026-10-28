# -*- coding: utf-8 -*-
"""報告用講稿（印出來或用手機看）：講稿.md ＋ 投影片縮圖 → 講稿_10-28.pdf。2026-10-28。

使用者：「我報告可以另外寫講稿嗎，這次沒有雙螢幕可以看稿」→ 不能用 PowerPoint 的簡報者檢視，備忘稿看不到，另外出一份 PDF。
內容就是 講稿.md（build.js 寫進備忘稿的同一份），只是重新排版：
  - 每頁：頁碼、段落、預估時間；左邊投影片縮圖、右邊要講的話（字大）
  - 【備忘：…】＝被問到時再講，放在下方灰框（字小）；其他【…】開頭的段落照樣要講，標籤粗體
  - 第一頁：流程總覽（每頁標題與時間、總時間）

用法（先重建 deck.pptx，再把每頁輸出成縮圖，最後跑這支）：
    node ASD\\slides_src\\2026-10-28_redpen\\build.js
    （PowerPoint 輸出縮圖到 ASD\\slides_src\\2026-10-28_redpen\\script_thumbs\\slide-01.png …，1600×900；
      做法同 ..\\2026-10-14_redpen\\render.ps1，OutDir 換成 script_thumbs）
    python ASD\\slides_src\\2026-10-28_redpen\\make_script.py
PDF 用 Edge headless 列印（A4；要從 PowerShell／cmd 執行，Claude 的 bash 沙箱裡 Edge 開不起來）。
縮圖轉成 JPEG（寬 1280）再放進 PDF，檔案小、手機開得快。縮圖與 PDF 都不進版控（.gitignore 的 /ASD/**/*）。
"""
import os
import re
import html
import shutil
import tempfile
import subprocess

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '講稿.md')
THUMBS = os.path.join(HERE, 'script_thumbs')
OUT_HTML = os.path.join(HERE, 'script_1028.html')          # Edge 的命令列用 ASCII 檔名比較保險，印完再複製成中文檔名
OUT_PDF_TMP = os.path.join(HERE, 'script_1028.pdf')
OUT_PDF = os.path.join(HERE, '講稿_10-28.pdf')
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

# ── 讀 講稿.md ─────────────────────────────────────────────────────────
HEAD = re.compile(r'^## 第 (\d+) 頁\s*(\S+)\s+(.+?)\s*<!--\s*(\w+)\s*-->\s*$')
TIME = re.compile(r'^（約 (\d+) 秒(?:；(.+))?）$')
pages = []
cur = None
for line in open(SRC, encoding='utf-8').read().split('\n'):
    m = HEAD.match(line)
    if m:
        cur = dict(no=int(m.group(1)), sec=m.group(2), title=m.group(3), key=m.group(4), secs=None, tag='', say=[], memo=[])
        pages.append(cur)
        continue
    if line.startswith('#'):
        cur = None
        continue
    if cur is None or not line.strip():
        continue
    t = TIME.match(line.strip())
    if t and cur['secs'] is None:
        cur['secs'], cur['tag'] = int(t.group(1)), (t.group(2) or '')
        continue
    (cur['memo'] if line.startswith('【備忘') else cur['say']).append(line.strip())
assert [p['no'] for p in pages] == list(range(1, len(pages) + 1)), '講稿.md 的頁碼不連續'
for p in pages:
    th = os.path.join(THUMBS, 'slide-%02d.png' % p['no'])
    assert os.path.exists(th), '少了縮圖 %s（先把 deck.pptx 輸出成圖）' % th
    im = Image.open(th).convert('RGB')
    im.resize((1280, round(1280 * im.height / im.width)), Image.LANCZOS).save(th[:-4] + '.jpg', quality=85)
    assert p['secs'] is not None, '第 %d 頁沒有（約 N 秒）' % p['no']


def para(t):
    """【標籤】開頭的段落：標籤粗體；其餘照原文（跳脫 HTML）。"""
    m = re.match(r'^【([^】]+)】(.*)$', t)
    if m:
        return '<p><b>%s</b>%s</p>' % (html.escape('【' + m.group(1) + '】'), html.escape(m.group(2)))
    return '<p>%s</p>' % html.escape(t)


def memo(t):
    m = re.match(r'^【備忘：?([^】]*)】(.*)$', t)
    lab, body = (m.group(1), m.group(2)) if m else ('', t)
    return '<li><b>%s</b>%s</li>' % (html.escape(lab + '：') if lab else '', html.escape(body))


main = [p for p in pages if p['sec'] != '附']
appx = [p for p in pages if p['sec'] == '附']
tmin = lambda ps: sum(p['secs'] for p in ps) / 60

# ── HTML ───────────────────────────────────────────────────────────────
CSS = """
@page { size: A4; margin: 11mm 12mm 12mm 12mm; }
* { box-sizing: border-box; }
body { font-family: 'Microsoft JhengHei', 'Noto Sans TC', sans-serif; color: #141A1D; margin: 0; font-size: 12.5pt; line-height: 1.7; }
h1 { font-size: 20pt; margin: 0 0 2mm; }
.sub { color: #5F6A6B; font-size: 11pt; margin-bottom: 5mm; }
table.toc { width: 100%; border-collapse: collapse; font-size: 10.5pt; line-height: 1.35; }
table.toc th { text-align: left; color: #5F6A6B; font-weight: normal; border-bottom: 1.2pt solid #141A1D; padding: 1.4mm 2mm; }
table.toc td { border-bottom: 0.5pt solid #D9D9D2; padding: 0.9mm 2mm; vertical-align: top; }
table.toc small { color: #5F6A6B; }
table.toc td.n, table.toc td.s { white-space: nowrap; }
table.toc tr.appx td { color: #5F6A6B; }
.legend { margin: 0 0 4mm; font-size: 10.5pt; color: #5F6A6B; }
.page { break-before: page; }
.slide { margin-bottom: 7mm; break-inside: avoid; }          /* 一頁投影片的講稿盡量不跨頁，報告時不用中途翻頁 */
.hd { display: flex; align-items: baseline; gap: 3mm; border-bottom: 1.5pt solid #0E7C7B; padding-bottom: 1mm; margin-bottom: 2.5mm; break-after: avoid; }
.hd .no { font-size: 17pt; font-weight: bold; color: #0E7C7B; }
.hd .ttl { font-size: 13pt; font-weight: bold; flex: 1; }
.hd .tm { font-size: 10.5pt; color: #5F6A6B; white-space: nowrap; }
.row { display: flex; gap: 4mm; align-items: flex-start; }
.row img { width: 66mm; border: 0.5pt solid #D9D9D2; flex: none; break-inside: avoid; }
.say p { margin: 0 0 2mm; }
.say p.hint { color: #A34F1B; font-size: 11pt; }
.memo { break-inside: avoid; margin-top: 2.5mm; background: #EFEFEB; border-radius: 1.5mm; padding: 1.8mm 3mm; font-size: 9.5pt; line-height: 1.5; color: #3A4446; }
.memo .cap { font-weight: bold; color: #5F6A6B; margin-bottom: 0.8mm; }
.memo ul { margin: 0; padding-left: 4.5mm; }
.memo li { margin-bottom: 0.8mm; }
.appx .hd { border-bottom-color: #9A9A93; }
.appx .hd .no { color: #5F6A6B; }
"""
out = ['<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>講稿 10-28</title><style>%s</style></head><body>' % CSS]
out.append('<h1>報告講稿：10/14 會議意見之回覆</h1>')
out.append('<div class="sub">2026-10-28 meeting｜共 %d 頁｜正文約 %.0f 分鐘、附錄約 %.0f 分鐘（依講稿各頁之預估）</div>'
           % (len(pages), tmin(main), tmin(appx)))
out.append('<div class="legend">每頁：左邊是投影片縮圖，右邊是要講的內容；灰框的「備忘」是被問到時再補充的內容，不用主動講。'
           '附錄（第 %d 頁起）視時間與提問再講。</div>' % (appx[0]['no'] if appx else 0))
out.append('<table class="toc"><tr><th>頁</th><th>段落</th><th>內容</th><th>時間</th></tr>')
acc = 0
for p in pages:
    acc += p['secs']
    out.append('<tr class="%s"><td class="n">%d</td><td class="s">%s</td><td>%s%s</td><td class="s">%d 秒（累計 %d:%02d）</td></tr>'
               % ('appx' if p['sec'] == '附' else '', p['no'], html.escape(p['sec']), html.escape(p['title']),
                  ('　<small>%s</small>' % html.escape(p['tag'].split('，')[0])) if p['tag'] else '', p['secs'], acc // 60, acc % 60))
out.append('</table>')
first = True
for p in pages:
    cls = 'slide appx' if p['sec'] == '附' else 'slide'
    out.append('<div class="%s%s">' % (cls, ' page' if first else ''))
    first = False
    tag = p['tag'].split('，')                                     # 標題旁只放第一段（試作／附錄），其餘放在講稿第一行
    out.append('<div class="hd"><span class="no">第 %d 頁</span><span class="ttl">%s %s</span><span class="tm">約 %d 秒%s</span></div>'
               % (p['no'], html.escape(p['sec']), html.escape(p['title']), p['secs'],
                  ('｜' + html.escape(tag[0])) if p['tag'] else ''))
    hint = ('<p class="hint">（%s）</p>' % html.escape('，'.join(tag[1:]))) if len(tag) > 1 else ''
    out.append('<div class="row"><img src="script_thumbs/slide-%02d.jpg"><div class="say">%s%s</div></div>'
               % (p['no'], hint, ''.join(para(t) for t in p['say'])))
    if p['memo']:
        out.append('<div class="memo"><div class="cap">備忘（被問到時）</div><ul>%s</ul></div>' % ''.join(memo(t) for t in p['memo']))
    out.append('</div>')
out.append('</body></html>')
open(OUT_HTML, 'w', encoding='utf-8').write('\n'.join(out))
print('->', OUT_HTML)

# ── PDF（Edge headless）─────────────────────────────────────────────────
url = 'file:///' + OUT_HTML.replace('\\', '/')
if os.path.exists(OUT_PDF_TMP):
    os.remove(OUT_PDF_TMP)
prof = tempfile.mkdtemp(prefix='edge_script_')                    # 另開一個 Edge 設定檔，不會接到使用者正在用的 Edge
subprocess.run([EDGE, '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--user-data-dir=' + prof,
                '--no-pdf-header-footer', '--print-to-pdf=' + OUT_PDF_TMP, url], check=True, timeout=180,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
shutil.rmtree(prof, ignore_errors=True)
assert os.path.exists(OUT_PDF_TMP), 'Edge 沒有產生 PDF'
shutil.copyfile(OUT_PDF_TMP, OUT_PDF)
os.remove(OUT_PDF_TMP)
print('-> %s（正文約 %.0f 分鐘、附錄約 %.0f 分鐘）' % (OUT_PDF, tmin(main), tmin(appx)))
