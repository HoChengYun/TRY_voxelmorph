// 下次 meeting 簡報（暫定 10/28）：回覆 10/14 老師的紅字。
// 2026-10-13 使用者：「先做這一 part 就好」→ 目前只有 ③ 殘留體積（p12、p16、p18 的紅字）；10-14 加第 3 頁「不同模型」（p18）。其他項目做完再加頁。
// 原本第 3 頁「敏感度分析與顱頂厚度之比較」使用者說字太多、拿掉，半徑只用 5 mm。
// 版面、配色、講稿寫進備忘稿的做法沿用 2026-10-14_redpen。投影片上的數字一律從 deck_data.json 讀（make_resvol.py 產生），不手打。
const path = require('path');
const fs = require('fs');
const pptxgen = require(require.resolve('pptxgenjs', {
  paths: [path.join(__dirname, '..', '2026-09_mix_tigerbx', 'node_modules')],
}));

const HERE = __dirname;
const OUT = process.argv[2] || path.join(HERE, 'deck.pptx');
const D = JSON.parse(fs.readFileSync(path.join(HERE, 'deck_data.json'), 'utf8'));
const MROOT = path.join(HERE, '..', '..', '..', 'models');

const C = {
  INK: '141A1D', PAPER: 'FAFAF8', SURF: 'EFEFEB', RULE: 'D9D9D2', MUTED: '5F6A6B',
  TEAL: '0E7C7B', TEAL_L: '5BB8B6', RUST: 'A34F1B', RUST_L: 'D9895A', WHITE: 'FFFFFF',
};
const F = { SANS: 'Microsoft JhengHei', MONO: 'Consolas' };
const f3 = (x) => x.toFixed(3);
const sg = (x) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(2);
const pval = (p) => (p < 0.001 ? '< 0.001' : p.toFixed(3));

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'VoxelMorph：10/14 會議意見之回覆';
pres.author = 'HoChengYun';

const W = 13.333, M = 0.6;
let page = 0;
const ORDER = ['rv_def', 'rv_res', ...(D.models ? ['rv_models'] : [])];

function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function txt(s, text, o) {
  s.addText(text, Object.assign({ fontFace: F.SANS, color: C.INK, margin: 0, isTextBox: true, valign: 'top', lang: 'zh-TW' }, o));
}
// 講稿（講稿.md）→ 每頁的備忘稿。標題行「## 第 N 頁 … <!-- key -->」的 key 對應 ORDER
const NOTES = {};
{
  const f = path.join(__dirname, '講稿.md');
  if (fs.existsSync(f)) {
    let key = null;
    for (const line of fs.readFileSync(f, 'utf8').split(/\r?\n/)) {
      const h = line.match(/^## 第 (\d+) 頁.*<!--\s*(\w+)\s*-->/);
      if (h) { key = h[2]; NOTES[key] = { page: +h[1], text: [] }; continue; }
      if (/^#{1,2} /.test(line)) { key = null; continue; }
      if (key) NOTES[key].text.push(line.replace(/\*\*/g, ''));
    }
  }
}
const noteOf = (k) => (NOTES[k] ? NOTES[k].text.join('\n').trim() : '');

function base(eyebrow, title) {
  const s = pres.addSlide();
  page += 1;
  const notes = noteOf(ORDER[page - 1]);
  s.background = { color: C.PAPER };
  txt(s, eyebrow, { x: M, y: 0.42, w: 11.5, h: 0.28, fontSize: 12, bold: true, color: C.MUTED, charSpacing: 2 });
  txt(s, title, { x: M, y: 0.74, w: 12.1, h: 0.7, fontSize: 26, bold: true });
  txt(s, String(page).padStart(2, '0'), { x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.28, fontFace: F.MONO, fontSize: 11, color: C.MUTED, align: 'right' });
  if (notes) s.addNotes(notes);
  return s;
}
function table(s, rows, o) {
  const head = rows[0].map((t) => ({ text: t, options: { bold: true, color: C.MUTED, fontSize: 12, fill: { color: C.PAPER } } }));
  const body = rows.slice(1).map((r) => r.map((c) => (typeof c === 'object' ? c : { text: String(c) })));
  s.addTable([head].concat(body), Object.assign({
    fontFace: F.SANS, fontSize: 13, color: C.INK, valign: 'middle', lang: 'zh-TW',
    border: { type: 'solid', pt: 0.75, color: C.RULE }, fill: { color: C.PAPER },
    margin: [0.06, 0.1, 0.06, 0.1],
  }, o));
}
function bullets(s, items, o) {
  const arr = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    if (Array.isArray(it)) {
      it.forEach((run, j) => arr.push({ text: run.text, options: Object.assign({ bullet: j === 0 ? { indent: 14 } : undefined, breakLine: j === it.length - 1 && !last }, run.options) }));
    } else arr.push({ text: it, options: { bullet: { indent: 14 }, breakLine: !last } });
  });
  s.addText(arr, Object.assign({ fontFace: F.SANS, fontSize: 15, color: C.INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 8, lang: 'zh-TW' }, o));
}
function fitImage(s, p, x, y, maxW, maxH, alt) {
  if (!fs.existsSync(p)) throw new Error('找不到圖片 ' + p);
  const z = pngSize(p);
  let w = maxW, h = maxW * z.h / z.w;
  if (h > maxH) { h = maxH; w = maxH * z.w / z.h; }
  s.addImage({ path: p, x: x + (maxW - w) / 2, y, w, h, altText: alt || '' });
  return { w, h };
}
const CH = (n) => path.join(MROOT, 'deck_charts', n);

const EB = '③ 去顱骨殘留：改以體積量測（10/14 意見 p12、p16、p18）';
const R5 = D.radii['5'], E = R5.near;

{
  // ─────────────────────────────────────────────── 殘留體積之定義（示意圖＋式 (1)(2)）
  const s = base(EB, '殘留體積之定義：FreeSurfer 標記腦組織以外之非零 voxel');
  fitImage(s, CH('1028_resvol_demo.png'), M, 1.35, 12.13, 2.95, '殘留體積示意（' + D.subject + '）');
  txt(s, D.subject + '：黃線＝FreeSurfer 標記之腦組織 S；藍線＝B（S 補起腦溝與大腦縱裂）；紅色＝殘留，共 '
    + D.demo_cm3.toFixed(0) + ' cm³', { x: M, y: 4.33, w: 12.13, h: 0.3, fontSize: 12, color: C.MUTED, align: 'center' });
  fitImage(s, CH('1028_resvol_eq.png'), M, 4.7, 12.13, 2.3, '式 (1)(2)');
}

{
  // ─────────────────────────────────────────────── 結果：散佈圖＋Dice 數值
  const s = base(EB, '殘留體積與 Dice（n = ' + E.n + '）：配準後之 Dice 與殘留無關');
  fitImage(s, CH('1028_resvol_result.png'), M, 1.35, 12.13, 4.35, '殘留體積與 ΔDice、Dice 數值');
  bullets(s, [
    [{ text: '配準後 Dice 與殘留體積無關（r = ' + sg(E.r_after) + '，p = ' + pval(E.p_after) + '）；ΔDice 之負相關主要來自 affine Dice 隨殘留略高（r = '
      + sg(E.r_before) + '，p = ' + pval(E.p_before) + '）' }],
    [{ text: '殘留體積中位數 ' + E.median.toFixed(0) + ' cm³（' + R5.range[0].toFixed(0) + '～' + R5.range[1].toFixed(0)
      + '）；亮度達皮質一半以上者約 ' + (100 * R5.bright_share).toFixed(0) + '%，其餘主要為腦周圍之腦脊髓液' }],
    [{ text: 'Dice：殘留鄰近結構（' + [...new Set(R5.labs_name.split('、').map((n) => n.replace(/^[左右]/, '')))].join('、')
      + '，左右）之平均；' + D.model + '；test ' + E.test.n + '、val ' + E.val.n
      + '、MRS ' + E.MRS.n + '，均未參與訓練' }],
  ], { x: M, y: 5.8, w: 12.13, h: 1.15, fontSize: 13.5, paraSpaceAfter: 5 });
}

if (D.models) {
  // ─────────────────────────────────────────────── 不同模型（10/14 紅字 p18：「想看不同 model 做出來的效果」）
  const Ms = Object.values(D.models);
  const rng = (k) => sg(Math.min(...Ms.map((m) => m[k]))) + '～' + sg(Math.max(...Ms.map((m) => m[k])));
  const s = base(EB, '不同配準模型之比較：配準後之 Dice 皆與殘留體積無關');
  fitImage(s, CH('1028_resvol_models.png'), M, 1.3, 12.13, 4.75, '五個模型：殘留體積與 ΔDice、Dice（每位受試者一點）');
  bullets(s, [
    [{ text: '五個模型結論一致：ΔDice 隨殘留略減（r = ' + rng('r') + '），配準後 Dice 與殘留無關（r = ' + rng('r_after')
      + '，p > ' + (Math.floor(Math.min(...Ms.map((m) => m.p_after)) * 10) / 10).toFixed(1) + '）' }],
    [{ text: 'Dice：殘留鄰近結構（大腦皮質、小腦皮質，左右）之平均，非 30 個結構；受試者（n = ' + E.n + '）、殘留體積與前頁相同，僅更換配準模型' }],
  ], { x: M, y: 6.15, w: 12.13, h: 0.8, fontSize: 13.5, paraSpaceAfter: 5 });
}

if (page !== ORDER.length) throw new Error('頁數對不上：做了 ' + page + ' 頁，ORDER 排了 ' + ORDER.length + ' 頁');
ORDER.forEach((k, i) => {
  if (!NOTES[k]) console.warn('[!] 講稿.md 沒有 ' + k + '（第 ' + (i + 1) + ' 頁）');
  else if (NOTES[k].page !== i + 1) console.warn('[!] 講稿.md 的 ' + k + ' 寫第 ' + NOTES[k].page + ' 頁，實際是第 ' + (i + 1) + ' 頁');
});
pres.writeFile({ fileName: OUT }).then(() => console.log('ok -> ' + path.basename(OUT) + ' ｜' + page + ' 頁'));
