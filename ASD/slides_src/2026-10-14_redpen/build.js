// 2026-10-14 meeting 簡報：回覆 09-30 老師的紅字（p18、p22、p25）
// 版面與配色沿用 2026-09-20_cross。投影片上的數字一律從 deck_data.json 讀，不手打。
// 還沒跑完的實驗（mix_exp6、mix_exp7、mix_wide_vel）顯示「跑中」；結果帶回來後重跑 gather.py、make_charts.py、這支。
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
const pct = (x) => x.toFixed(3) + '%';
const sgn = (x, d) => (x >= 0 ? '+' : '-') + Math.abs(x).toFixed(d === undefined ? 2 : d);
const pval = (p) => (p < 0.001 ? 'p < 0.001' : 'p = ' + p.toFixed(2));

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'VoxelMorph：09-30 老師紅字的回覆';
pres.author = 'HoChengYun';

const W = 13.333, M = 0.6;
let page = 0;

function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function txt(s, text, o) {
  s.addText(text, Object.assign({ fontFace: F.SANS, color: C.INK, margin: 0, isTextBox: true, valign: 'top', lang: 'zh-TW' }, o));
}
function base(eyebrow, title, notes) {
  const s = pres.addSlide();
  page += 1;
  s.background = { color: C.PAPER };
  txt(s, eyebrow, { x: M, y: 0.42, w: 11.5, h: 0.28, fontFace: F.SANS, fontSize: 12, bold: true, color: C.MUTED, charSpacing: 2 });
  txt(s, title, { x: M, y: 0.74, w: 12.1, h: 0.7, fontSize: 26, bold: true });
  txt(s, String(page).padStart(2, '0'), { x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.28, fontFace: F.MONO, fontSize: 11, color: C.MUTED, align: 'right' });
  if (notes) s.addNotes(notes);
  return s;
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || C.SURF }, line: { color: C.RULE, width: 0.75 } });
}
function stat(s, big, small, x, y, w, color) {
  txt(s, big, { x, y, w, h: 0.66, fontFace: F.MONO, fontSize: 32, bold: true, color: color || C.TEAL, align: 'center' });
  txt(s, small, { x, y: y + 0.72, w, h: 0.6, fontSize: 12.5, color: C.MUTED, align: 'center' });
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
const hl = (t, color) => ({ text: t, options: { color: color || C.TEAL, bold: true } });
const dim = (t) => ({ text: t, options: { color: C.MUTED } });
function bullets(s, items, o) {
  const arr = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    if (Array.isArray(it)) {
      it.forEach((run, j) => arr.push({ text: run.text, options: Object.assign({ bullet: j === 0 ? { indent: 14 } : undefined, breakLine: j === it.length - 1 && !last }, run.options) }));
    } else arr.push({ text: it, options: { bullet: { indent: 14 }, breakLine: !last } });
  });
  s.addText(arr, Object.assign({ fontFace: F.SANS, fontSize: 15, color: C.INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 10, lang: 'zh-TW' }, o));
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
const FC = (n) => path.join(MROOT, 'folding_check', n);

const m = D.models, P = D.paired, FD = D.folding, R = D.residue, DL = D.dilution, TC = D.top_check;
const done = (e) => m[e].status === 'done';
const score = (e) => (done(e) ? f3(m[e].mean) : '跑中');
const fold = (e) => (done(e) ? pct(m[e].jneg) : '跑中');
const RUN = { text: '跑中', options: { color: C.RUST, bold: true } };

// ───────────────────────────────────────────────────────── 01 封面
{
  const s = pres.addSlide();
  page += 1;
  s.background = { color: '1A2125' };
  txt(s, '2026-10-14   MEETING', { x: M, y: 2.2, w: 11, h: 0.3, fontFace: F.MONO, fontSize: 12, color: '8A9294', charSpacing: 4 });
  txt(s, '上次老師交代的五件事', { x: M, y: 2.7, w: 11.5, h: 1.0, fontSize: 40, bold: true, color: C.WHITE });
  txt(s, '擠爆的位置、速度場調平滑權重、只算殘留旁邊的 Dice、後腦杓、加寬改速度場',
    { x: M, y: 3.75, w: 11.5, h: 0.5, fontSize: 18, color: 'D9DEDF' });
  txt(s, 'VoxelMorph 腦部影像配準', { x: M, y: 4.8, w: 11, h: 0.4, fontSize: 16, color: '8A9294' });
}

// ───────────────────────────────────────────────────────── 02 一頁看完
{
  const s = base('SUMMARY', '一頁看完：五件事做到哪');
  const lam = done('mix_exp6')
    ? '權重 2：' + score('mix_exp5') + '｜1：' + score('mix_exp6') + '｜0.5：' + score('mix_exp7')
    : '權重 2：' + score('mix_exp5') + '、不擠爆；權重 1、0.5 跑中';
  const wide = done('mix_wide_vel') ? score('mix_wide_vel') + '（加寬位移場 ' + score('mix_wide') + '）' : 'AI 上跑中';
  table(s, [
    ['老師寫的', '做了什麼', '結果'],
    ['① 確認擠爆的位置（p18）', '51 位 test，找出每個擠爆點在哪', '一小團一小團，沿著腦溝，在皮質和白質裡'],
    ['② 速度場 λ 去調一下（p18）', '平滑權重 2、1、0.5 各跑一顆', lam],
    ['③ Dice 只算沒切乾淨附近（p22）', '只平均殘留旁邊的結構', hl('頭頂殘留越多，皮質對得越差（' + DL.n + ' 人）', C.RUST)],
    ['④ 後腦杓也去看（p22）', '多掃後腦杓的殘留', '有殘留，但不影響配準'],
    ['⑤ 加寬改看看速度場（p25）', '加寬 2 倍＋速度場', wide],
  ], { x: M, y: 1.65, w: 12.13, colW: [3.7, 3.75, 4.68], fontSize: 13.5, rowH: 0.62 });
  txt(s, '③ 另外確認了：頭頂那層「殘留」不是 FreeSurfer 把腦畫太小，是真的沒切乾淨（第 12 頁）。',
    { x: M, y: 6.0, w: 12.13, h: 0.45, fontSize: 14.5, color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 03 ① 擠爆在哪
const F3 = FD.mix_exp3;
{
  const s = base('① 擠爆的位置（p18）', '擠爆的點：散在皮質和白質裡，每個人擠的位置不一樣');
  fitImage(s, CH('1014_folding_where.png'), M, 1.45, 12.13, 3.45, '51 位擠爆點疊在模板上');
  txt(s, '擠爆＝形變把空間捏到翻過去。圖是平滑權重 1 的位移場（mix_exp3），51 位疊在模板上，只標 3 位以上在同一點擠爆的地方，越紅越多人。'
        + '速度場版完全不擠爆，所以只看位移場。',
    { x: M, y: 4.98, w: 12.13, h: 0.45, fontSize: 12.5, color: C.MUTED, align: 'center' });
  const w = (12.13 - 0.4 * 2) / 3;
  [[F3.any1.toFixed(0) + '%', '腦裡的點，至少 1 位在那裡擠爆過', C.TEAL],
   [F3.any5.toFixed(1) + '%', '5 位以上都在同一點擠爆 → 每人位置不同', C.RUST],
   [(F3.points_med / FD.mix_exp4.points_med).toFixed(1) + ' 倍', '平滑權重 1 比權重 2 多的擠爆點\n（落點分布差不多，加寬也沒變多）', C.TEAL]].forEach((it, i) => {
    const x = M + i * (w + 0.4);
    card(s, x, 5.45, w, 1.45);
    stat(s, it[0], it[1], x, 5.55, w, it[2]);
  });
}

// ───────────────────────────────────────────────────────── 04 ① 哪些區域、多深
{
  const s = base('① 擠爆的位置（p18）', '八成落在大腦皮質和白質，深部結構幾乎沒有');
  fitImage(s, CH('1014_folding_regions.png'), M, 1.45, 12.13, 3.6, '擠爆點落在哪些區域');
  const w = (12.13 - 0.4 * 2) / 3;
  [[F3.ctx_wm.toFixed(0) + '%', '擠爆點落在大腦皮質＋白質', C.RUST],
   [F3.depth_med.toFixed(0) + ' mm', '離腦表面往內（中位數）\n也就是腦溝凹進去的那一段', C.TEAL],
   [F3.share['深部灰質・海馬・杏仁核'].toFixed(1) + '%', '深部灰質、海馬、杏仁核', C.TEAL]].forEach((it, i) => {
    const x = M + i * (w + 0.4);
    card(s, x, 5.2, w, 1.6);
    stat(s, it[0], it[1], x, 5.32, w, it[2]);
  });
}

// ───────────────────────────────────────────────────────── 05 ① 放大一團
{
  const s = base('① 擠爆的位置（p18）', '放大一團來看：沿著一條腦溝，格子被捏到翻過去');
  fitImage(s, FC('folding_zoom_T054.png'), M, 1.45, 12.13, 4.55, 'T054 最大的一團擠爆點');
  bullets(s, [
    [{ text: '黃線＝原本方正的格子被形變拉成的樣子，紅點＝擠爆的點', options: { color: C.MUTED } }],
    [{ text: '紅點排成一條線、沿著腦溝；格線在那裡交叉', options: { bold: true } },
     { text: '　→ 模型為了把腦溝對到模板，把那裡的空間捏到翻過去' }],
  ], { x: M, y: 6.1, w: 12.13, h: 0.9, fontSize: 14, paraSpaceAfter: 6 });
}

// ───────────────────────────────────────────────────────── 06 ② 上次的結論
{
  const s = base('② 速度場的 λ（p18）', '上次的結論：同樣條件下，速度場比位移場好，而且不擠爆');
  fitImage(s, CH('ablation.png'), M, 1.45, 12.13, 4.4, '四顆模型一次只改一件事');
  bullets(s, [
    [{ text: '同樣全尺寸、平滑權重 2：速度場 ' + score('mix_exp5') + ' ＞ 位移場 ' + score('mix_exp4'), options: { bold: true, color: C.TEAL } },
     { text: '　擠爆 ' + fold('mix_exp5') + ' vs ' + fold('mix_exp4') }],
    [{ text: '位移場最好的 ' + score('mix_exp3') + ' 是平滑權重 1。', options: { bold: true } },
     { text: '老師的問題：速度場也把權重降下來，會怎樣？' }],
  ], { x: M, y: 6.0, w: 12.13, h: 1.0, fontSize: 14.5, paraSpaceAfter: 6 });
}

// ───────────────────────────────────────────────────────── 07 ② λ 掃描
{
  const s = base('② 速度場的 λ（p18）', '速度場的平滑權重：2 → 1 → 0.5');
  fitImage(s, CH('1014_lambda.png'), M, 1.45, 12.13, 4.55, '平滑權重與 Dice、擠爆');
  const items = [
    [{ text: '權重 2（mix_exp5）：' + score('mix_exp5') + '，擠爆 ' + fold('mix_exp5'), options: { bold: true } }],
  ];
  ['mix_exp6', 'mix_exp7'].forEach((e) => {
    items.push(done(e)
      ? [{ text: '權重 ' + m[e].weight + '（' + e + '）：' + score(e) + '，擠爆 ' + fold(e), options: { bold: true } }]
      : [{ text: '權重 ' + m[e].weight + '（' + e + '）：AI 上跑中', options: { bold: true, color: C.RUST } }]);
  });
  bullets(s, items, { x: M, y: 6.1, w: 12.13, h: 0.9, fontSize: 14, paraSpaceAfter: 4 });
}

// ───────────────────────────────────────────────────────── 08 ③ 老師的做法
{
  const s = base('③ 只算殘留旁邊的 Dice（p22）', '老師的做法：不要 30 個結構全部平均，只平均殘留旁邊的');
  txt(s, '殘留的每一點，找離它最近的是哪個結構；佔殘留點 5% 以上、而且在 30 個評估結構裡的，才拿來平均。',
    { x: M, y: 1.55, w: 12.13, h: 0.45, fontSize: 15 });
  table(s, [
    ['殘留在哪', '殘留旁邊的結構（佔殘留點）', '只平均這些的 Dice'],
    ['頭頂', R.top.near_shares_pooled, R.top.labels_pooled],
    ['顱底', R.base.near_shares_pooled, R.base.labels_pooled],
    ['後腦杓', R.back.near_shares_pooled, R.back.labels_pooled],
  ], { x: M, y: 2.25, w: 12.13, colW: [1.25, 5.55, 5.33], fontSize: 13 });
  card(s, M, 4.55, 12.13, 0.8, 'FFF3E8');
  txt(s, [
    { text: '為什麼：', options: { bold: true } },
    { text: '殘留貼在腦的外面，旁邊幾乎都是大腦皮質。視丘、海馬迴這些離殘留很遠的結構也一起平均的話，影響會被稀釋掉。' },
  ], { x: M + 0.3, y: 4.74, w: 11.5, h: 0.45, fontSize: 15, fontFace: F.SANS, lang: 'zh-TW', color: C.INK, margin: 0 });
  const oc = R.base.near_shares_pooled.match(/視交叉 ([0-9.]+%)/);
  if (oc) {
    txt(s, '視交叉（顱底 ' + oc[1] + '）不在 30 個評估結構裡，所以沒算進去。',
      { x: M, y: 5.75, w: 12.13, h: 0.4, fontSize: 12.5, color: C.MUTED });
  }
}

// ───────────────────────────────────────────────────────── 09 ③ 結果
{
  const s = base('③ 只算殘留旁邊的 Dice（p22）', '頭頂殘留越多，皮質對得越差；30 個結構一起平均就看不出來');
  fitImage(s, CH('1014_dilution.png'), M, 1.45, 12.13, 4.6, '30 個結構一起平均 vs 只平均殘留旁邊的結構');
  const T = R.top;
  bullets(s, [
    [{ text: '每個點是一個人，共 ' + DL.n + ' 人（test 50、val 50、第五包 MRS 70，三批都沒進過訓練）', options: { color: C.MUTED } }],
    [{ text: '殘留最多的 1/4：模型貢獻 +' + f3(T.dirty_gain) + '；最少的 1/4：+' + f3(T.clean_gain), options: { bold: true } },
     { text: '　三批分開算，方向都一樣', options: { color: C.MUTED } }],
  ], { x: M, y: 6.15, w: 12.13, h: 0.9, fontSize: 13.5, paraSpaceAfter: 4 });
}

// ───────────────────────────────────────────────────────── 10 ③④ 三個位置
{
  const s = base('③④ 三個位置一起看（p22）', '只有頭頂有影響，顱底和後腦杓沒有');
  fitImage(s, CH('1014_regions.png'), M, 1.45, 12.13, 3.3, '三個位置的殘留與模型貢獻');
  const row = (k, n) => [n, f3(R[k].dirty_gain), f3(R[k].clean_gain),
    R[k].pooled_p < 0.05 ? hl(sgn(R[k].pooled_r) + '（' + pval(R[k].pooled_p) + '）', C.RUST)
                         : sgn(R[k].pooled_r) + '（' + pval(R[k].pooled_p) + '）'];
  table(s, [
    ['位置', '殘留最多 1/4 的模型貢獻', '殘留最少 1/4 的模型貢獻', '相關（' + R.top.pooled_n + ' 人）'],
    row('top', '頭頂'), row('base', '顱底'), row('back', '後腦杓'),
  ], { x: M, y: 4.95, w: 12.13, colW: [1.6, 3.6, 3.6, 3.33], fontSize: 13.5 });
}

// ───────────────────────────────────────────────────────── 11 ④ 後腦杓
{
  const s = base('④ 後腦杓（p22）', '後腦杓也有殘留，但跟配準好不好沒有關係');
  fitImage(s, CH('1014_back_example.png'), M, 1.45, 12.13, 3.7, '後腦杓殘留的例子');
  const B = D.back;
  bullets(s, [
    [{ text: '紅色＝FreeSurfer 標到的腦的最後面，再往後還亮著的組織', options: { color: C.MUTED } }],
    [{ text: 'test ' + B.n + ' 位：後腦杓殘留中位數 ' + B.median.toFixed(2) + ' mm（' + B.min.toFixed(2) + '～' + B.max.toFixed(2) + '）', options: { bold: true } }],
    [{ text: '殘留多寡跟模型貢獻：r = ' + sgn(R.back.pooled_r) + '（' + pval(R.back.pooled_p) + '，' + R.back.pooled_n + ' 人）→ 沒有關係', options: { bold: true, color: C.TEAL } }],
    [{ text: '每個人都有左右腦中間、大腦和小腦之間的兩片腦膜，所以這個數字有一個大家共同的底', options: { color: C.MUTED, fontSize: 13 } }],
  ], { x: M, y: 5.3, w: 12.13, h: 1.7, fontSize: 14, paraSpaceAfter: 5 });
}

// ───────────────────────────────────────────────────────── 12 ③ 是不是 FreeSurfer 畫錯
{
  const s = base('③ 只算殘留旁邊的 Dice（p22）', '會不會是 FreeSurfer 把腦畫太小？不是，是真的沒切乾淨');
  txt(s, '程式把「FreeSurfer 畫的腦外面還亮的東西」當成殘留。如果其實是 FreeSurfer 畫太小、漏掉一塊皮質，'
        + '那 Dice 低就不是殘留害的。看圖：綠色（皮質）一樣完整，紅色都在綠色外面。',
    { x: M, y: 1.45, w: 12.13, h: 0.7, fontSize: 14, color: C.MUTED });
  fitImage(s, CH('1014_top_example.png'), M, 2.2, 12.13, 3.05, '殘留多的人 vs 乾淨的人，頭頂放大');
  const w = (12.13 - 0.4 * 2) / 3;
  [[TC.ctx_thick.dirty.toFixed(1) + ' vs ' + TC.ctx_thick.clean.toFixed(1), '白質到腦頂幾格：殘留多 vs 少\n→ 皮質一樣厚，沒有缺一塊', C.TEAL],
   [(100 * TC.top_ctx_min).toFixed(1) + '%', '腦的最上面一格是皮質\n（' + TC.n + ' 人每個都至少這麼多）', C.TEAL],
   [TC.cont_int.dirty.toFixed(1) + ' 倍', '那層東西的亮度只有皮質的 ' + TC.cont_int.dirty.toFixed(1) + ' 倍\n→ 比皮質暗，是腦膜這類東西', C.RUST]].forEach((it, i) => {
    const x = M + i * (w + 0.4);
    card(s, x, 5.4, w, 1.5);
    stat(s, it[0], it[1], x, 5.5, w, it[2]);
  });
}

// ───────────────────────────────────────────────────────── 13 ⑤ 加寬＋速度場
{
  const s = base('⑤ 加寬改速度場（p25）', done('mix_wide_vel') ? '加寬＋速度場：' + score('mix_wide_vel') : '加寬＋速度場：AI 上跑中');
  const cell = (e) => (done(e) ? { text: e + '\nDice ' + score(e) + '　擠爆 ' + fold(e) }
                               : { text: e + '\n跑中', options: { color: C.RUST, bold: true } });
  table(s, [
    ['全尺寸、平滑權重 1', '預設寬度', '加寬 2 倍'],
    [{ text: '位移場', options: { bold: true } }, cell('mix_exp3'), cell('mix_wide')],
    [{ text: '速度場', options: { bold: true } }, cell('mix_exp6'), cell('mix_wide_vel')],
  ], { x: M, y: 1.65, w: 8.2, colW: [2.2, 3.0, 3.0], fontSize: 14, rowH: 0.85 });
  const X0 = 9.2, WW = 3.53;
  bullets(s, [
    [{ text: '橫著比：只差寬度', options: { bold: true } },
     { text: '\n位移場加寬 +' + (P.width_disp ? f3(P.width_disp.mean) : '?') + '（' + (P.width_disp ? P.width_disp.win + '/' + P.width_disp.n : '') + ' 位變好）',
       options: { color: C.MUTED, fontSize: 13 } }],
    [{ text: '直著比：只差版本', options: { bold: true } },
     { text: '\n加寬之後，速度場還是比較好、又不擠爆嗎？', options: { color: C.MUTED, fontSize: 13 } }],
  ], { x: X0, y: 1.7, w: WW, h: 2.6, fontSize: 15, paraSpaceAfter: 12 });
  card(s, M, 4.6, 12.13, 1.9, 'FFF3E8');
  txt(s, [
    { text: '順便發現加寬版為什麼比預估慢：', options: { bold: true } },
    { text: '\nPyTorch 會先多佔一些顯存備用，加寬那顆想佔約 33 GB，超過 AI 的 24 GB，多的部分拿一般記憶體頂，所以變慢。' },
    { text: '\n這次訓練前多設一行（限制最多佔 85%），只管記憶體、不改計算，預計約 19 小時跑完（mix_wide 當時 26 小時）。',
      options: { color: C.MUTED } },
  ], { x: M + 0.3, y: 4.8, w: 11.5, h: 1.55, fontSize: 14.5, fontFace: F.SANS, lang: 'zh-TW', color: C.INK, margin: 0, paraSpaceAfter: 4 });
}

// ───────────────────────────────────────────────────────── 14 下一步
{
  const s = base('NEXT', '下一步');
  bullets(s, [
    [{ text: 'mix_exp6、mix_exp7、mix_wide_vel 結果回來', options: { bold: true } },
     { text: '\n　補進第 7 頁（平滑權重）和第 13 頁（加寬＋速度場）', options: { color: C.MUTED } }],
    [{ text: '新資料：第五包 MRS（' + D.mrs.n + ' 人）已經前處理好', options: { bold: true } },
     { text: '\n　等其他包到齊，一起併進來重新切分，當成新的一版資料', options: { color: C.MUTED } }],
    [{ text: '順帶看到：模型拿去對從沒看過的 MRS 研究，Dice ' + f3(D.mrs.after) + '（起點 ' + f3(D.mrs.before) + '）', options: { bold: true } },
     { text: '\n　跟原本 test 的 ' + score('mix_exp3') + ' 一樣 → 換一個研究的資料也能用', options: { color: C.MUTED } }],
  ], { x: M, y: 1.8, w: 12.13, h: 4.2, fontSize: 16, paraSpaceAfter: 16 });
}

pres.writeFile({ fileName: OUT }).then(() => console.log('ok ->', OUT, '｜' + page + ' 頁'));
