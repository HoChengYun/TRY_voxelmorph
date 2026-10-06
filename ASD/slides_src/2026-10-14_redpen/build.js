// 2026-10-14 meeting 簡報：回覆 09-30 老師的紅字（p18、p22、p23、p25）
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
const f4 = (x) => x.toFixed(4);
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

const m = D.models, P = D.paired, FD = D.folding, R = D.residue, DL = D.dilution, TC = D.top_check, TT = D.train_time || {};
const MM = D.residue_mm;            // ③④ 原始數值版（mm／mm³、Dice 進步不扣平均、用 mm 分組），2026-10-05 起第 12、14、15 頁用這個
const done = (e) => m[e].status === 'done';
const score = (e) => (done(e) ? f3(m[e].mean) : '跑中');
// 速度場權重 1、0.5 只有零星幾個點（平均 0.000002%、0.0001%），印 0.000% 會被看成完全沒有
const jfmt = (x) => (x === 0 ? '0%' : x < 0.001 ? '< 0.001%' : pct(x));
const fold = (e) => (done(e) ? jfmt(m[e].jneg) : '跑中');
const HAS_LAM = done('mix_exp6') && done('mix_exp7');
const HAS_PARAMS = fs.existsSync(FC('folding_params.png'));    // check_folding.py --views（速度場兩顆也算過之後）
const HAS_WCURVE = done('mix_wide_vel') && fs.existsSync(CH('curve_wide.png'));   // 2026-09-20_cross\make_compare.py --set wide
const HAS_SIX = fs.existsSync(CH('1014_six.png'));                                // make_charts.py（2026-10-05 加）
const HAS_BASE6 = fs.existsSync(CH('1014_six_base.png'));                         // make_charts.py（2026-10-05 加，顱底）
// ⑤ 加寬改速度場的補充（2026-10-06 使用者：「1、2、3 項，也可以放訓練 loss 和一些視覺化比較」）：[頁, 要先有的圖]
const WIDE_MORE = [['wide_struct', [CH('1014_wide_struct.png')]], ['wide_diff', [CH('1014_wide_difficulty.png')]],
  ['wide_loss', [CH('1014_wide_loss.png')]], ['wide_full', [FC('folding_full_pair_T054.png')]],
  ['wide_vis', [FC('folding_zoom_pair_T054.png')]]];
const HAS_WM = Object.fromEntries(WIDE_MORE.map(([k, f]) => [k, f.every((x) => fs.existsSync(x))]));

// 頭頂殘留怎麼量（四頁，每頁一張圖＋一個公式區塊）：make_method.py（2026-10-06 加）
const METHOD = [1, 2, 3, 4].flatMap((k) => ['1014_method_' + k + '.png', '1014_method_eq' + k + '.png']);
const HAS_METHOD = METHOD.every((f) => fs.existsSync(CH(f)));

// 頁碼：第 2 頁的表、最後一頁的「下一步」會引用後面的頁，所以先排好順序再算（最後會檢查有沒有對上）
const ORDER = ['cover', 'summary', 'fold_where', ...(HAS_PARAMS ? ['fold_params'] : []), 'fold_regions', 'fold_zoom', 'lam_prev', 'lam',
  ...(HAS_LAM ? ['lam_struct', 'lam_grid'] : []),
  'res_method', ...(HAS_METHOD ? ['res_m1', 'res_m2', 'res_m3', 'res_m4'] : []), 'res_result', ...(HAS_SIX ? ['res_six'] : []), 'res_regions', 'back',
  ...(HAS_BASE6 ? ['base6'] : []), 'res_check',
  'wide', ...(HAS_WCURVE ? ['wide_curve'] : []), ...WIDE_MORE.filter(([, f]) => f.every((x) => fs.existsSync(x))).map(([k]) => k),
  'next'];
const PG = Object.fromEntries(ORDER.map((k, i) => [k, i + 1]));

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
  const lam = HAS_LAM
    ? '2 → 1：' + sgn(P.lam_vel_1.mean, 4) + '；1 → 0.5：' + sgn(P.lam_vel_05.mean, 4) + '（沒再變好）；都幾乎不擠爆'
    : '權重 2：' + score('mix_exp5') + '、不擠爆；權重 1、0.5 跑中';
  const wide = done('mix_wide_vel')
    ? 'Dice ' + score('mix_wide_vel') + '，跟加寬位移場打平；幾乎不擠爆'
    : 'AI 上跑中';
  table(s, [
    ['老師寫的', '做了什麼', '結果'],
    ['① 確認擠爆的位置（p18）', '51 位 test，找出每個擠爆點在哪', '一小團一小團，沿著腦溝，在皮質和白質裡'],
    ['② 速度場 λ 去調一下（p18）', '平滑權重 2、1、0.5 各跑一顆', lam],
    ['③ Dice 只算沒切乾淨附近（p23）', '只平均殘留旁邊的結構', hl('頭頂殘留越多，皮質對得越差（' + DL.n + ' 人）', C.RUST)],
    ['④ 後腦杓也去看（p22）', '多掃後腦杓的殘留', '有殘留，但不影響配準'],
    ['⑤ 加寬改看看速度場（p25）', '加寬 2 倍＋速度場', wide],
  ], { x: M, y: 1.65, w: 12.13, colW: [3.7, 3.75, 4.68], fontSize: 13.5, rowH: 0.62 });
  txt(s, '③ 另外確認了：頭頂那層「殘留」不是 FreeSurfer 把腦畫太小，是真的沒切乾淨（第 ' + PG.res_check + ' 頁）。',
    { x: M, y: 6.0, w: 12.13, h: 0.45, fontSize: 14.5, color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 03 ① 擠爆在哪
const F3 = FD.mix_exp3;
{
  const s = base('① 擠爆的位置（p18）', '擠爆的點：散在皮質和白質裡，每個人擠的位置不一樣');
  fitImage(s, FC('folding_views.png'), M, 1.42, 7.75, 5.5, '軸狀、冠狀、矢狀各切 4 刀');
  const X0 = 8.6, WW = 4.13;
  txt(s, '擠爆＝形變把空間捏到翻過去。圖是平滑權重 1 的位移場（mix_exp3），51 位疊在模板上，只標 3 位以上在同一點擠爆的地方，越紅越多人。',
    { x: X0, y: 1.5, w: WW, h: 1.05, fontSize: 12.5, color: C.MUTED });
  [[F3.any1.toFixed(0) + '%', '腦裡的點，至少 1 位擠爆過', C.TEAL],
   [F3.any5.toFixed(1) + '%', '5 位以上都在同一點擠爆\n→ 每個人位置不同', C.RUST],
   [(F3.points_med / FD.mix_exp4.points_med).toFixed(1) + ' 倍', '平滑權重 1 比 2 多的擠爆點', C.TEAL]].forEach((it, i) => {
    const y = 2.65 + i * 1.42;
    card(s, X0, y, WW, 1.3);
    stat(s, it[0], it[1], X0, y + 0.06, WW, it[2]);
  });
}

if (HAS_PARAMS) {
  // ─────────────────────────────────────────────────────── ① 不同設定的擠爆位置
  const s = base('① 擠爆的位置（p18）', '換不同設定：位移場權重越小擠爆越多，速度場幾乎沒有');
  fitImage(s, FC('folding_params.png'), M, 1.42, 12.13, 5.0, '不同設定 × 三個方向的擠爆位置');
  txt(s, '每一欄是一顆模型、每一列是一個方向，一樣只標 3 位以上在同一點擠爆的地方。'
        + '速度場那幾欄是空的：點本來就很少，而且每個人散在不同地方。',
    { x: M, y: 6.5, w: 12.13, h: 0.45, fontSize: 13, color: C.MUTED, align: 'center' });
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
  const s = base('② 速度場的 λ（p18）', HAS_LAM ? '速度場：平滑權重 2 → 1 有幫助，再降到 0.5 就沒再變好'
                                                : '速度場的平滑權重：2 → 1 → 0.5');
  fitImage(s, CH('1014_lambda.png'), M, 1.45, 12.13, 4.4, '平滑權重與 Dice、擠爆');
  let items;
  if (HAS_LAM) {
    const L1 = P.lam_vel_1, L05 = P.lam_vel_05, VW = P.version_w1;
    items = [
      [{ text: '權重 2 → 1：' + sgn(L1.mean, 4) + '（51 位裡 ' + L1.win + ' 位變好）；1 → 0.5：' + sgn(L05.mean, 4) + '（沒差）',
         options: { bold: true } }],
      [{ text: '速度場最好的是權重 1（' + score('mix_exp6') + '），比位移場權重 1（' + score('mix_exp3') + '）少 ' + f4(VW.mean),
         options: { bold: true } },
       { text: '　但擠爆從 ' + fold('mix_exp3') + ' 變成 ' + fold('mix_exp6'), options: { bold: true, color: C.TEAL } }],
    ];
  } else {
    items = [[{ text: '權重 2（mix_exp5）：' + score('mix_exp5') + '，擠爆 ' + fold('mix_exp5'), options: { bold: true } }]];
    ['mix_exp6', 'mix_exp7'].forEach((e) => {
      items.push(done(e)
        ? [{ text: '權重 ' + m[e].weight + '（' + e + '）：' + score(e) + '，擠爆 ' + fold(e), options: { bold: true } }]
        : [{ text: '權重 ' + m[e].weight + '（' + e + '）：AI 上跑中', options: { bold: true, color: C.RUST } }]);
    });
  }
  bullets(s, items, { x: M, y: 6.0, w: 12.13, h: 0.95, fontSize: 14.5, paraSpaceAfter: 6 });
}

if (HAS_LAM) {
  // ─────────────────────────────────────────────────────── ② 權重 0.5 為什麼沒再變好
  const S = D.struct, df = (n) => S.mix_exp7[n] - S.mix_exp5[n];
  const s = base('② 速度場的 λ（p18）', '權重 0.5 為什麼沒再變好：大結構變好、小結構變差');
  fitImage(s, CH('1014_lambda_struct.png'), M, 1.45, 7.7, 5.45, '各結構跟權重 2 比變多少');
  const X0 = 8.55, WW = 4.18;
  bullets(s, [
    [{ text: '大結構一路變好', options: { bold: true, color: C.TEAL } },
     { text: '\n大腦皮質 ' + sgn(df('大腦皮質'), 3) + '、白質 ' + sgn(df('大腦白質'), 3), options: { color: C.MUTED, fontSize: 13.5 } }],
    [{ text: '小結構變差', options: { bold: true, color: C.RUST } },
     { text: '\n脈絡叢 ' + sgn(df('脈絡叢'), 3) + '、腦脊髓液 ' + sgn(df('腦脊髓液'), 3), options: { color: C.MUTED, fontSize: 13.5 } }],
    [{ text: 'Dice 是 30 個結構「一樣重」的平均', options: { bold: true } },
     { text: '\n大小結構互相抵掉，平均就打平', options: { color: C.MUTED, fontSize: 13.5 } }],
    [{ text: '權重 0.5 的皮質 ' + f3(S.mix_exp7['大腦皮質']), options: { bold: true } },
     { text: '\n比位移場權重 1 的 ' + f3(S.mix_exp3['大腦皮質']) + ' 還高', options: { color: C.MUTED, fontSize: 13.5 } }],
  ], { x: X0, y: 1.65, w: WW, h: 5.2, fontSize: 15, paraSpaceAfter: 14 });
}

if (HAS_LAM) {
  // ─────────────────────────────────────────────────────── ② 形變網格：四顆對照
  const s = base('② 速度場的 λ（p18）', '平滑權重越小，形變捏得越細');
  fitImage(s, CH('grid_lambda.png'), M, 1.5, 12.13, 4.0, '四顆的形變網格');
  txt(s, '同一位受試者（T054）、同一個切面。黃線＝原本方正的格子被形變拉成的樣子，越往右越扭。',
    { x: M, y: 5.65, w: 12.13, h: 0.4, fontSize: 14.5, align: 'center' });
  txt(s, '速度場三顆都幾乎不擠爆；最右邊的位移場一樣扭得很細，但擠爆 ' + fold('mix_exp3') + '。',
    { x: M, y: 6.1, w: 12.13, h: 0.4, fontSize: 13.5, align: 'center', color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 08 ③ 老師的做法
{
  const s = base('③ 只算殘留旁邊的 Dice（p23）', '老師的做法：不要 30 個結構全部平均，只平均殘留旁邊的');
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

if (HAS_METHOD) {
  // ─────────────────────────────────────────────────────── ③ 頭頂殘留怎麼量（四頁）
  // 2026-10-06 使用者：「放進簡報取代第 12 頁、一定要放公式、希望可以和 paper 一樣好閱讀」。
  // 每一步一頁：上面是圖、下面是編號公式＋「其中」符號說明（公式用 LaTeX 字型畫成圖，都是 make_method.py 產生的）。
  // 圖和公式區塊就是投影片上的大小（寬 12.13 吋），第 4 頁公式多一行（25 mm 的說明），所以圖矮一點
  // 第 2、4 頁的公式說明各多一行（換門檻、換 25 mm 結論都一樣），圖相對矮一點
  const STEP = [['找腦的頂邊', 3.6, 1.85], ['「亮」的門檻', 3.55, 2.02], ['每根吸管數幾格', 3.6, 1.85], ['頭頂那一塊取平均', 2.95, 2.66]];
  STEP.forEach(([name, hf, he], i) => {
    const s = base('③ 只算殘留旁邊的 Dice（p23）', '頭頂殘留厚度怎麼量（' + (i + 1) + '/4）：' + name);
    fitImage(s, CH('1014_method_' + (i + 1) + '.png'), M, 1.38, 12.13, hf, '第 ' + (i + 1) + ' 步的圖');
    fitImage(s, CH('1014_method_eq' + (i + 1) + '.png'), M, 1.38 + hf + 0.04, 12.13, he, '第 ' + (i + 1) + ' 步的公式');
  });
}

// ───────────────────────────────────────────────────────── 09 ③ 結果
{
  const s = base('③ 只算殘留旁邊的 Dice（p23）', '頭頂殘留越多，皮質對得越差；30 個結構一起平均就看不出來');
  fitImage(s, CH('1014_dilution.png'), M, 1.4, 12.13, 3.55, '頭頂殘留厚度 vs Dice 進步多少：30 個結構一起平均 vs 只平均殘留旁邊的結構');
  // 2026-10-05 使用者要標 Dice 數值：殘留最多／最少 1/4 的「起點 → 配準後」。起點一定要一起寫（只寫配準後會被起點騙）。
  // 分組直接用 mm 切（170 人的 1/4、3/4 分位數，gather.py 的 residue_mm）
  // ⚠️「起點就高」「進步差不多」「起點差不多」這幾個字是照現在的數字寫的
  const T = MM.top;
  const ba = (b, a) => f3(b) + ' → ' + f3(a) + '（' + sgn(a - b, 3) + '）';
  table(s, [
    ['頭頂殘留厚度', '30 個結構一起平均：Dice 起點 → 配準後', '只算殘留旁邊（大腦皮質）：Dice 起點 → 配準後'],
    [T.hi.toFixed(2) + ' mm 以上（' + T.dirty_n + ' 人）', ba(T.all_dirty_b, T.all_dirty_a), ba(T.dirty_b, T.dirty_a)],
    [T.lo.toFixed(2) + ' mm 以下（' + T.clean_n + ' 人）', ba(T.all_clean_b, T.all_clean_a), ba(T.clean_b, T.clean_a)],
    [{ text: '怎麼看', options: { bold: true } },
     '起點就高 ' + f3(T.all_dirty_b - T.all_clean_b) + '，進步差不多 → 看不出來',
     hl('起點差不多，配準後低 ' + f3(T.clean_a - T.dirty_a) + '、進步少 ' + f3((T.clean_a - T.clean_b) - (T.dirty_a - T.dirty_b)), C.RUST)],
  ], { x: M, y: 5.12, w: 12.13, colW: [2.75, 4.65, 4.73], fontSize: 13 });
  txt(s, '每個點是一個人，共 ' + T.n + ' 人（test 50、val 50、第五包 MRS 70，三批都沒進過訓練；三批分開算，方向都一樣）',
    { x: M, y: 6.72, w: 11.0, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

if (HAS_SIX) {
  // ─────────────────────────────────────────────────────── ③ 同樣 6 位：兩種算法
  const s = base('③ 只算殘留旁邊的 Dice（p23）', '同樣 6 位：30 個結構一起平均分不出來，只算皮質就分得出來');
  // 公式 2026-10-06 一度放這頁右邊，使用者說分開 → 獨立一頁（res_formula），這頁恢復整頁寬的圖
  fitImage(s, CH('1014_six.png'), M, 1.45, 12.13, 4.95, '頭頂殘留最多 3 位與最少 3 位的皮質 Dice');
  txt(s, 'test 裡頭頂殘留最多、最少各 3 位（跟之前殘留對照圖同一批人），紅色＝殘留。大字是皮質 Dice 的「起點 → 配準後」，下面是進步多少。',
    { x: M, y: 6.5, w: 12.13, h: 0.45, fontSize: 13, color: C.MUTED, align: 'center' });
}

// ───────────────────────────────────────────────────────── 10 ③④ 三個位置
{
  const s = base('③④ 三個位置一起看（p22、p23）', '只有頭頂有影響，顱底和後腦杓沒有');
  fitImage(s, CH('1014_regions.png'), M, 1.38, 12.13, 3.15, '三個位置：殘留量 vs 只算殘留旁邊結構的 Dice 進步');
  // 2026-10-05 起：上面是三張散佈圖（r 寫在圖上），表格放 Dice（起點 → 配準後），分組直接用 mm／mm³ 切（gather.py 的 residue_mm）
  const labs = (k) => [...new Set(R[k].labels_pooled.split('、').map((x) => x.replace(/^[左右]/, '')))].join('、');
  const ba2 = (b, a) => f3(b) + ' → ' + f3(a) + '（' + sgn(a - b, 3) + '）';
  const amt = (k, v) => (k === 'base' ? Math.round(v).toLocaleString('en-US') + ' mm³' : v.toFixed(2) + ' mm');
  // 2026-10-06 使用者：「只算這些結構」要寫參考了哪些 FreeSurfer 結構 → 一個結構一行：中文名稱＋FreeSurferColorLUT 的
  // 正式名稱與標籤編號（gather.py 的 fs_pairs，左右合併）。Dice 那兩欄拆成「門檻」＋「起點 → 配準後」兩行
  const fsCell = (k) => ({ text: MM[k].fs_pairs.flatMap(([zh, fs], i, arr) => [
    { text: zh + '　' },
    { text: fs, options: { fontSize: 10.5, color: C.MUTED, breakLine: i < arr.length - 1 } }]) });
  const two = (a, b) => ({ text: [{ text: a, options: { breakLine: true } }, { text: b }] });
  const row = (k, n) => [n, fsCell(k),
                         two(amt(k, MM[k].hi) + ' 以上', ba2(MM[k].dirty_b, MM[k].dirty_a)),
                         two(amt(k, MM[k].lo) + ' 以下', ba2(MM[k].clean_b, MM[k].clean_a))];
  table(s, [
    ['位置', '只算這些 FreeSurfer 結構（標籤編號，左右合併）', '殘留最多 1/4：Dice 起點 → 配準後', '殘留最少 1/4：Dice 起點 → 配準後'],
    row('top', '頭頂'), row('base', '顱底'), row('back', '後腦杓'),
  ], { x: M, y: 4.68, w: 12.13, colW: [0.95, 5.3, 2.94, 2.94], fontSize: 12 });
}

// ───────────────────────────────────────────────────────── 11 ④ 後腦杓
{
  const s = base('④ 後腦杓（p22）', '後腦杓也有殘留，但跟配準好不好沒有關係');
  // 2026-10-05 使用者：第 13、15 頁在講同一件事，圖要統一 → 改成跟第 13 頁一樣的 3 對 3（原本是 1 對、五個切面、不標 Dice）
  fitImage(s, CH('1014_six_back.png'), M, 1.42, 12.13, 4.55, '後腦杓殘留最多 3 位與最少 3 位的皮質 Dice');
  const B = D.back, K = MM.back;
  const bl = [...new Set(R.back.labels_pooled.split('、').map((x) => x.replace(/^[左右]/, '')))].join('、');
  txt(s, 'test 裡後腦杓殘留最多、最少各 3 位，紅色＝殘留；只算殘留旁邊的' + bl + '。看「進步」：兩組交錯在一起，分不出誰殘留多',
    { x: M, y: 6.02, w: 12.13, h: 0.35, fontSize: 12.5, color: C.MUTED, align: 'center' });
  txt(s, K.n + ' 人：後腦杓殘留厚度跟 Dice 進步多少 r = ' + sgn(K.r) + '（' + pval(K.p) + '）→ 沒有關係',
    { x: M, y: 6.4, w: 12.13, h: 0.35, fontSize: 14.5, bold: true, color: C.TEAL, align: 'center' });
  txt(s, (HAS_METHOD ? '算法同頭頂（第 ' + PG.res_m1 + '～' + PG.res_m4 + ' 頁），吸管改成前後方向。' : '')
    + '左右腦中間、大小腦之間本來就有腦膜（test ' + B.n + ' 位中位數 ' + B.median.toFixed(2) + ' mm）',
    { x: M, y: 6.75, w: 11.0, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

if (HAS_BASE6) {
  // ─────────────────────────────────────────────────────── ③ 顱底：跟第 13 頁（頭頂）、後腦杓那頁同一個樣子
  // 2026-10-05 使用者：「這個顱底也來一張」。⚠️「這 6 位裡殘留多的進步還略多」是照現在的數字寫的
  const s = base('③ 只算殘留旁邊的 Dice（p23）', '顱底也有殘留，但跟配準好不好沒有關係');
  fitImage(s, CH('1014_six_base.png'), M, 1.42, 12.13, 4.55, '顱底殘留最多 3 位與最少 3 位的旁邊結構 Dice');
  const K = MM.base;
  const bl = [...new Set(R.base.labels_pooled.split('、').map((x) => x.replace(/^[左右]/, '')))].join('、');
  txt(s, 'test 裡顱底殘留最多、最少各 3 位，紅色＝離腦 10 mm 以外還留著的東西；只算殘留旁邊的' + bl + '。這 6 位裡殘留多的進步還略多',
    { x: M, y: 6.02, w: 12.13, h: 0.35, fontSize: 12, color: C.MUTED, align: 'center' });
  txt(s, K.n + ' 人：顱底殘留體積跟 Dice 進步多少 r = ' + sgn(K.r) + '（' + pval(K.p) + '）→ 沒有關係',
    { x: M, y: 6.4, w: 12.13, h: 0.35, fontSize: 14.5, bold: true, color: C.TEAL, align: 'center' });
  txt(s, '殘留大多在腦的前下方；切面選穿過最大一坨殘留中心的那一片，所以每個人切的位置不一樣',
    { x: M, y: 6.75, w: 11.0, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 12 ③ 是不是 FreeSurfer 畫錯
{
  const s = base('③ 只算殘留旁邊的 Dice（p23）', '會不會是 FreeSurfer 把腦畫太小？不是，是真的沒切乾淨');
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
  const WV = done('mix_wide_vel');
  const s = base('⑤ 加寬改速度場（p25）', WV ? '加寬改成速度場：Dice 跟加寬位移場打平，而且幾乎不擠爆' : '加寬＋速度場：AI 上跑中');
  const cell = (e) => (done(e) ? { text: e + '\nDice ' + score(e) + '　擠爆 ' + fold(e) }
                               : { text: e + '\n跑中', options: { color: C.RUST, bold: true } });
  table(s, [
    ['全尺寸、平滑權重 1', '預設寬度', '加寬 2 倍'],
    [{ text: '位移場', options: { bold: true } }, cell('mix_exp3'), cell('mix_wide')],
    [{ text: '速度場', options: { bold: true } }, cell('mix_exp6'), cell('mix_wide_vel')],
  ], { x: M, y: 1.65, w: 8.2, colW: [2.2, 3.0, 3.0], fontSize: 14, rowH: 0.85 });
  const X0 = 9.2, WW = 3.53;
  const win = (q) => q.win + '/' + q.n + ' 位';
  const thou = (x) => Math.round(x).toLocaleString('en-US');
  const sub = (t) => ({ text: t, options: { color: C.MUTED, fontSize: 13 } });
  // 擠爆點數跟第 4 頁的圖用同一個來源（check_folding.py）；沒算過的才退回 test CSV 換算（兩邊算法差 0.1% 左右）
  const FP = (e) => FD[e] || { points_mean: m[e].points, points_max: m[e].points_max, n_any: m[e].n_any };
  if (WV) {
    bullets(s, [
      [{ text: '橫著比：只差寬度', options: { bold: true } },
       sub('\n位移場 ' + sgn(P.width_disp.mean, 4) + '（' + win(P.width_disp) + '變好）'
           + '\n速度場 ' + sgn(P.width_vel.mean, 4) + '（' + win(P.width_vel) + '變好）')],
      [{ text: '直著比：速度場 − 位移場', options: { bold: true } },
       sub('\n預設寬度 ' + sgn(P.version_w1_vel.mean, 4) + '（' + win(P.version_w1_vel) + '較高）'
           + '\n加寬 2 倍 ' + sgn(P.version_wide.mean, 4) + '（' + win(P.version_wide) + '）→ 打平')],
      [{ text: '擠爆的點', options: { bold: true } },
       sub('\n位移場加寬：每人約 ' + thou(FP('mix_wide').points_mean) + ' 點'
           + '\n速度場加寬：' + FP('mix_wide_vel').n_any + ' 位有、最多 ' + thou(FP('mix_wide_vel').points_max) + ' 點')],
    ], { x: X0, y: 1.6, w: WW, h: 3.5, fontSize: 15, paraSpaceAfter: 8 });
    txt(s, '→ 分數跟最高的 mix_wide 一樣，又幾乎不擠爆：目前最好的一顆',
      { x: M, y: 4.45, w: 8.2, h: 0.45, fontSize: 16, bold: true, color: C.TEAL });
    card(s, M, 5.2, 12.13, 1.7, 'FFF3E8');
    const t = TT.mix_wide_vel, t0 = TT.mix_wide;
    txt(s, [
      { text: '順便：加寬版為什麼比預估慢', options: { bold: true } },
      { text: '\nPyTorch 會先多佔一些顯存備用，加寬那顆想佔約 33 GB，超過 AI 的 24 GB，多的部分拿一般記憶體頂，所以變慢。' },
      { text: '\n訓練前多設一行限制最多佔多少（只管記憶體、不改計算）'
          + (t && t0 ? '：這顆每步 ' + t.sec.toFixed(1) + ' 秒、' + Math.round(t.hours) + ' 小時跑完；mix_wide 沒加，每步 '
                       + t0.sec.toFixed(1) + ' 秒、' + Math.round(t0.hours) + ' 小時' : '')
          + '（mix_exp6、7 同時跑：每步 10 幾秒 → 約 3 秒）。',
        options: { color: C.MUTED } },
    ], { x: M + 0.3, y: 5.35, w: 11.5, h: 1.45, fontSize: 14, paraSpaceAfter: 4 });
  } else {
    bullets(s, [
      [{ text: '橫著比：只差寬度', options: { bold: true } },
       sub('\n位移場加寬 ' + sgn(P.width_disp.mean, 4) + '（' + win(P.width_disp) + '變好）')],
      [{ text: '直著比：只差版本', options: { bold: true } },
       sub('\n加寬之後，速度場還是比較好、又不擠爆嗎？')],
    ], { x: X0, y: 1.7, w: WW, h: 2.6, fontSize: 15, paraSpaceAfter: 12 });
    card(s, M, 4.45, 12.13, 2.35, 'FFF3E8');
    txt(s, [
      { text: '順便發現加寬版為什麼比預估慢：', options: { bold: true } },
      { text: '\nPyTorch 會先多佔一些顯存備用，加寬那顆想佔約 33 GB，超過 AI 的 24 GB，多的部分拿一般記憶體頂，所以變慢。' },
      { text: '\n訓練前多設一行限制最多佔多少，只管記憶體、不改計算。mix_exp6、7 同時跑時已在 AI 上試過：每步 10 幾秒 → 約 3 秒。',
        options: { color: C.MUTED } },
      { text: '\n加寬＋速度場這顆也加了，預計約 19 小時跑完（mix_wide 當時 26 小時）。', options: { color: C.MUTED } },
    ], { x: M + 0.3, y: 4.62, w: 11.5, h: 2.05, fontSize: 14.5, paraSpaceAfter: 4 });
  }
}

if (HAS_WCURVE) {
  // ─────────────────────────────────────────────────────── ⑤ 訓練過程：版本 × 寬度四顆
  const s = base('⑤ 加寬改速度場（p25）', '訓練過程：加寬的兩顆分數一樣高，速度場從頭到尾都不擠爆');
  fitImage(s, CH('curve_wide.png'), M, 1.45, 12.13, 4.7, '四顆的驗證集 Dice 與擠爆比例');
  txt(s, '上：驗證集 51 位的 Dice，星號是挑中的那一輪。下：擠爆的比例，虛線是論文同版本的 0.366%。',
    { x: M, y: 6.2, w: 12.13, h: 0.35, fontSize: 13, color: C.MUTED, align: 'center' });
  // 2026-10-06：不用再訓練更久（gather.py 的 plateau：第 100 輪之後的範圍、上下晃的大小、每 100 輪的趨勢）
  const PV = D.plateau.mix_wide_vel, PW = D.plateau.mix_wide;
  txt(s, '不用再訓練更久：加寬兩顆從第 100 輪之後，驗證集都在 ' + f3(Math.min(PV.lo, PW.lo)) + '～' + f3(Math.max(PV.hi, PW.hi))
    + ' 之間上下晃（標準差約 ' + f3(PV.sd) + '），每 100 輪的趨勢只有 ' + sgn(PV.slope100, 4) + '、' + sgn(PW.slope100, 4) + '，比晃的幅度還小',
    { x: M, y: 6.6, w: 12.13, h: 0.35, fontSize: 13.5, bold: true, color: C.TEAL, align: 'center' });
}

if (HAS_WM.wide_struct) {
  // ─────────────────────────────────────────────────────── ⑤ 每個結構（make_charts.py 的 1014_wide_struct.png）
  // ⚠️「蒼白球、殼核」「方向相反」是照現在的數字寫的
  const S = D.struct, nm = Object.keys(S.mix_wide_vel);
  const dv = (n) => S.mix_wide_vel[n] - S.mix_exp6[n], dp = (n) => S.mix_wide[n] - S.mix_exp3[n];
  const worse = nm.filter((n) => dv(n) < 0).sort((a, b) => dv(a) - dv(b));
  const ver = nm.map((n) => Math.abs(S.mix_wide_vel[n] - S.mix_wide[n]));
  const s = base('⑤ 加寬改速度場（p25）', '每個結構：加寬讓大部分結構變好；加寬之後換版本，每個結構都差不多');
  fitImage(s, CH('1014_wide_struct.png'), M, 1.4, 12.13, 4.9, '加寬的效果、加寬後換版本，每個結構的 Dice 變化');
  bullets(s, [
    [{ text: '左：速度場加寬，' + nm.length + ' 種結構裡 ' + (nm.length - worse.length) + ' 種變好；', options: { bold: true } },
     { text: '變差的是' + worse.map((n) => n + ' ' + sgn(dv(n), 3)).join('、') + '（位移場加寬時蒼白球是 ' + sgn(dp('蒼白球'), 3)
            + '，方向相反）', options: { color: C.MUTED } }],
    [{ text: '右：同樣加寬 2 倍，速度場和位移場每個結構都差在 ±' + f3(Math.max(...ver)) + ' 以內', options: { bold: true } },
     { text: '　→「打平」不是平均剛好抵消，每個結構都差不多', options: { color: C.MUTED } }],
  ], { x: M, y: 6.35, w: 12.13, h: 0.75, fontSize: 13, paraSpaceAfter: 3 });
}

if (HAS_WM.wide_diff) {
  // ─────────────────────────────────────────────────────── ⑤ 越難對的人幫越多（1014_wide_difficulty.png）
  const WV = D.wide_diff.vel, WP = D.wide_diff.disp;
  const s = base('⑤ 加寬改速度場（p25）', '越難對的人，加寬幫越多：位移場、速度場都一樣');
  fitImage(s, CH('1014_wide_difficulty.png'), M, 1.38, 12.13, 4.25, '起點 Dice 與加寬後多進步多少');
  table(s, [
    ['加寬後多進步多少（Dice）', '起點最差 10 位', '中間 31 位', '起點最好 10 位', '相關'],
    ['位移場（mix_wide - mix_exp3）', sgn(WP.hard10, 4), sgn(WP.mid, 4), sgn(WP.easy10, 4), sgn(WP.r, 2)],
    ['速度場（mix_wide_vel - mix_exp6）', sgn(WV.hard10, 4), sgn(WV.mid, 4), sgn(WV.easy10, 4), sgn(WV.r, 2)],
  ], { x: M, y: 5.72, w: 12.13, colW: [4.0, 2.1, 2.0, 2.1, 1.93], fontSize: 12.5 });
  txt(s, '每個點是一個人（test 51 位）。分組用「起點 Dice」（只做線性對位，兩顆模型都沒碰過），用其中一顆模型的分數分組會有回歸平均的假象',
    { x: M, y: 6.82, w: 11.0, h: 0.3, fontSize: 11, color: C.MUTED });
}

if (HAS_WM.wide_loss) {
  // ─────────────────────────────────────────────────────── ⑤ 訓練 loss（1014_wide_loss.png）
  const L = D.loss_final;
  const s = base('⑤ 加寬改速度場（p25）', '訓練 loss：加寬的兩顆影像對得比較像，兩個版本的影像項幾乎疊在一起');
  fitImage(s, CH('1014_wide_loss.png'), M, 1.4, 12.13, 4.4, '四顆的訓練 loss：影像項、平滑項');
  table(s, [
    ['最後一輪（平均 100 步）', '位移場・預設', '位移場・加寬', '速度場・預設', '速度場・加寬'],
    ['影像項（越低越像）', L.mix_exp3.image.toFixed(3), L.mix_wide.image.toFixed(3), L.mix_exp6.image.toFixed(3), L.mix_wide_vel.image.toFixed(3)],
    ['平滑項', L.mix_exp3.smooth.toFixed(4), L.mix_wide.smooth.toFixed(4), L.mix_exp6.smooth.toFixed(4), L.mix_wide_vel.smooth.toFixed(4)],
  ], { x: M, y: 5.85, w: 12.13, colW: [3.33, 2.2, 2.2, 2.2, 2.2], fontSize: 12.5 });
  txt(s, '⚠️ 平滑項不能跨版本比：速度場罰的是「速度場」（積分之前）、位移場罰的是位移場本身。同一個版本裡，加寬前後幾乎一樣',
    { x: M, y: 6.95, w: 12.13, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

if (HAS_WM.wide_full) {
  // ─────────────────────────────────────────────────────── ⑤ 視覺化（大圖）：整片腦（check_folding.py --zoom-pair 一起畫的）
  // 2026-10-06 使用者看了放大圖：「可以來大圖的嗎」→ 整片腦、藍框＝下一頁放大的那一塊
  const s = base('⑤ 加寬改速度場（p25）', '視覺化（大圖）：位移場的擠爆點沿著腦溝散在各處，速度場一個都沒有');
  fitImage(s, FC('folding_full_pair_T054.png'), M, 1.38, 8.3, 5.7, 'T054 整片腦，加寬位移場 vs 加寬速度場');
  bullets(s, [
    [{ text: 'T054（test 的一位），三個方向各切一片', options: { bold: true } },
     { text: '\n穿過加寬位移場最大的一團擠爆點（左大腦白質）', options: { color: C.MUTED } }],
    [{ text: '上：加寬位移場（mix_wide）', options: { bold: true, color: C.RUST } },
     { text: '\n紅點＝擠爆的點，沿著腦溝散在皮質和白質，不只藍框那一團', options: { color: C.MUTED } }],
    [{ text: '下：加寬速度場（mix_wide_vel）', options: { bold: true, color: C.TEAL } },
     { text: '\n同樣三片一個都沒有，整顆腦 0 個', options: { color: C.MUTED } }],
    [{ text: '藍框＝下一頁放大的那一塊；格子每 4 mm 一條（放大那頁每 2 mm）', options: { color: C.MUTED, fontSize: 12 } }],
  ], { x: M + 8.5, y: 1.6, w: 3.63, h: 5.2, fontSize: 14, paraSpaceAfter: 12 });
}

if (HAS_WM.wide_vis) {
  // ─────────────────────────────────────────────────────── ⑤ 視覺化：同一個位置放大（check_folding.py --zoom-pair）
  const s = base('⑤ 加寬改速度場（p25）', '視覺化：同一個位置，位移場的格子翻過去，速度場只是扭、沒有翻');
  fitImage(s, FC('folding_zoom_pair_T054.png'), M, 1.38, 7.6, 5.65, 'T054 同一個位置，加寬位移場 vs 加寬速度場');
  bullets(s, [
    [{ text: 'T054，加寬位移場（mix_wide）最大的一團擠爆點，在左大腦白質', options: { bold: true } },
     { text: '\n上：加寬位移場　下：加寬速度場（mix_wide_vel），同一個位置、同一片', options: { color: C.MUTED } }],
    [{ text: '黃線＝格子被形變拉成的樣子，紅點＝擠爆的點', options: { color: C.MUTED } }],
    [{ text: '位移場：格線交叉、翻過去（紅點）', options: { bold: true, color: C.RUST } }],
    [{ text: '速度場：一樣扭得很厲害，但格子沒有翻，整顆腦 0 個擠爆點', options: { bold: true, color: C.TEAL } }],
    [{ text: (HAS_WM.wide_full ? '上一頁大圖的藍框放大來看；' : '') + '格子每 2 mm 一條', options: { color: C.MUTED, fontSize: 12 } }],
  ], { x: M + 7.8, y: 1.6, w: 4.33, h: 5.2, fontSize: 14, paraSpaceAfter: 12 });
}

// ───────────────────────────────────────────────────────── 14 下一步
{
  const s = base('NEXT', '下一步');
  const PEND = ['mix_exp6', 'mix_exp7', 'mix_wide_vel'].filter((e) => !done(e));
  const where = [...new Set(PEND.map((e) => (e === 'mix_wide_vel' ? '第 ' + PG.wide + ' 頁（加寬＋速度場）'
                                                                  : '第 ' + PG.lam + ' 頁（平滑權重）')))];
  const items = PEND.length
    ? [[{ text: PEND.join('、') + ' 結果回來', options: { bold: true } },
        { text: '\n　補進' + where.join('和'), options: { color: C.MUTED } }]]
    : [];
  bullets(s, items.concat([
    [{ text: '訓練時也用 FreeSurfer 標籤（論文的做法）', options: { bold: true } },
     { text: '\n　標籤權重 0.5、5 兩顆，接著在 AI 上跑。測試時一樣只用影像，看 Dice 能不能再往上', options: { color: C.MUTED } }],
    [{ text: '新資料：第五包 MRS（' + D.mrs.n + ' 人）已經前處理好', options: { bold: true } },
     { text: '\n　等其他包到齊，一起併進來重新切分，當成新的一版資料', options: { color: C.MUTED } }],
    [{ text: '順帶看到：模型拿去對從沒看過的 MRS 研究，Dice ' + f3(D.mrs.after) + '（起點 ' + f3(D.mrs.before) + '）', options: { bold: true } },
     { text: '\n　跟原本 test 的 ' + score('mix_exp3') + ' 一樣 → 換一個研究的資料也能用', options: { color: C.MUTED } }],
  ]), { x: M, y: 1.8, w: 12.13, h: 4.2, fontSize: 16, paraSpaceAfter: 16 });
}

if (page !== ORDER.length) throw new Error('頁數對不上：做了 ' + page + ' 頁，ORDER 排了 ' + ORDER.length + ' 頁（內文引用的頁碼會錯）');
pres.writeFile({ fileName: OUT }).then(() => console.log('ok ->', OUT, '｜' + page + ' 頁'));
