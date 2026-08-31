# -*- coding: utf-8 -*-
"""把 25 頁疊成一個檢查頁，量每頁在「Google Fonts 沒載入」時會不會爆版。

匯出 PDF 時 Google Fonts 嵌不進去，會退回系統字型。所以真正該量的是
fallback 生效時的版面，而不是開發時看到的版面。

用 --fallback 產生的檢查頁會把 Noto / IBM Plex 從字型堆疊裡拿掉，
模擬匯出時的情況。
"""
import io
import os
import re
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, 'v2')
FALLBACK = '--fallback' in sys.argv

ORDER = ['Main', 'Agenda', 'Pipeline', 'DataQC', 'Export', 'AtlasBuild', 'AffineWhat',
         'Preprocess', 'VerifySeg', 'Split', 'AtlasGap', 'ZeroPad', 'AtlasSeg',
         'VxmArch', 'VxmLoss', 'VxmDiffeo', 'Training', 'Curve', 'ResultTriplanar',
         'Contours', 'ResultLabels', 'ResultChecker', 'ResultJacobian', 'PerStruct', 'Caveats']

blocks = []
link = ''
for i, name in enumerate(ORDER, 1):
    p = os.path.join(V2, name + '.dc.html')
    s = io.open(p, encoding='utf-8').read()
    if not link:
        m = re.search(r'<link[^>]*fonts\.googleapis[^>]*>', s)
        link = m.group(0) if m else ''
    body = re.search(r'<x-dc>(.*?)</x-dc>', s, re.S).group(1)
    body = re.sub(r'<helmet>.*?</helmet>', '', body, flags=re.S)
    if FALLBACK:
        # 拿掉 Google Fonts 的字族，只留 fallback —— 模擬匯出時的狀況
        body = body.replace("'Noto Serif TC', ", '').replace("'Noto Sans TC', ", '')
        body = body.replace("'IBM Plex Mono', ", '')
    blocks.append('<div class="slot" data-name="%s" data-idx="%d">%s</div>' % (name, i, body))

head = '' if FALLBACK else link
html = """<!doctype html><html><head><meta charset="utf-8">%s
<style>
  body { margin: 0; background: #333; }
  .slot { position: relative; margin: 0 auto 8px; width: 1280px; }
</style></head><body>
%s
<script>
window.__report = () => {
  const out = [];
  document.querySelectorAll('.slot').forEach(slot => {
    const root = slot.firstElementChild;
    const rr = root.getBoundingClientRect();
    let worst = 0, who = '';
    root.querySelectorAll('*').forEach(el => {
      const r = el.getBoundingClientRect();
      if (r.height === 0 || r.width === 0) return;
      const ov = Math.max(r.bottom - rr.bottom, r.right - rr.right);
      if (ov > worst) { worst = ov; who = (el.textContent || '').trim().slice(0, 26); }
    });
    out.push({ n: slot.dataset.idx, name: slot.dataset.name,
               overflow: Math.round(worst), where: who });
  });
  return out.filter(o => o.overflow > 1);
};
</script></body></html>""" % (head, '\n'.join(blocks))

out = os.path.join(HERE, 'check_%s.html' % ('fallback' if FALLBACK else 'webfont'))
io.open(out, 'w', encoding='utf-8').write(html)
print('%s（%d 頁）' % (os.path.basename(out), len(ORDER)))
