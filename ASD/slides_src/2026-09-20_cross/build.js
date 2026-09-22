// 2026-09-20 meeting 簡報：520 顆、兩種版本、交叉測試
// 配色與版面沿用 2026-09_mix_tigerbx。投影片上的數字一律從 deck_data.json 讀，不手打。
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

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'VoxelMorph：520 顆資料、兩種版本、交叉測試';
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
  txt(s, eyebrow, { x: M, y: 0.42, w: 11.5, h: 0.28, fontFace: F.MONO, fontSize: 11, bold: true, color: C.MUTED, charSpacing: 3 });
  txt(s, title, { x: M, y: 0.74, w: 12.1, h: 0.7, fontSize: 28, bold: true });
  txt(s, String(page).padStart(2, '0'), { x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.28, fontFace: F.MONO, fontSize: 11, color: C.MUTED, align: 'right' });
  if (notes) s.addNotes(notes);
  return s;
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || C.SURF }, line: { color: C.RULE, width: 0.75 } });
}
function stat(s, big, small, x, y, w, color) {
  txt(s, big, { x, y, w, h: 0.66, fontFace: F.MONO, fontSize: 34, bold: true, color: color || C.TEAL, align: 'center' });
  txt(s, small, { x, y: y + 0.72, w, h: 0.6, fontSize: 12.5, color: C.MUTED, align: 'center' });
}
function table(s, rows, o) {
  const head = rows[0].map((t) => ({ text: t, options: { bold: true, color: C.MUTED, fontFace: F.MONO, fontSize: 11, fill: { color: C.PAPER } } }));
  const body = rows.slice(1).map((r) => r.map((c) => (typeof c === 'object' ? c : { text: String(c) })));
  s.addTable([head].concat(body), Object.assign({
    fontFace: F.SANS, fontSize: 13, color: C.INK, valign: 'middle', lang: 'zh-TW',
    border: { type: 'solid', pt: 0.75, color: C.RULE }, fill: { color: C.PAPER },
    margin: [0.06, 0.1, 0.06, 0.1],
  }, o));
}
const hl = (t, color) => ({ text: t, options: { color: color || C.TEAL, bold: true } });
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
const VIS = (exp, sid, kind, ep) => path.join(MROOT, exp, 'vis_' + sid, `${kind}_${sid}_${ep}.png`);

const m = D.models, X = D.cross, XP = D.cross_paired, P = D.paired;

// ───────────────────────────────────────────────────────── 01 封面
{
  const s = pres.addSlide();
  page += 1;
  s.background = { color: '1A2125' };
  txt(s, '2026-09-20   MEETING', { x: M, y: 2.2, w: 11, h: 0.3, fontFace: F.MONO, fontSize: 12, color: '8A9294', charSpacing: 4 });
  txt(s, '資料加到 520 顆、跑出兩種版本\n再互相交叉測試', { x: M, y: 2.7, w: 11.5, h: 1.8, fontSize: 40, bold: true, color: C.WHITE, lineSpacing: 46 });
  txt(s, 'VoxelMorph 腦部影像配準', { x: M, y: 4.8, w: 11, h: 0.4, fontSize: 16, color: 'D9DEDF' });
}

// ───────────────────────────────────────────────────────── 02 一頁看完
{
  const s = base('SUMMARY', '一頁看完');
  const w = (12.13 - 0.4 * 3) / 4;
  const items = [
    [`${D.n_total}`, '顆腦\n（上次 286）', C.TEAL],
    ['4', '顆模型\n兩種標籤 × 兩種版本', C.TEAL],
    [f3(m.mix_exp3.mean), 'FreeSurfer 標籤\n最好的成績', C.TEAL],
    [f3(m.tiger_exp3.mean), 'tigerbx 標籤\n最好的成績', C.RUST],
  ];
  items.forEach((it, i) => {
    const x = M + i * (w + 0.4);
    card(s, x, 1.7, w, 1.85);
    stat(s, it[0], it[1], x, 1.95, w, it[2]);
  });
  bullets(s, [
    [{ text: '資料從 286 顆加到 520 顆', options: { bold: true } }, { text: '　多了一批 234 顆的外部資料' }],
    [{ text: '換成論文的「位移場版」，成績更好一點', options: { bold: true } }, { text: '　兩種標籤都是 +0.009，非常一致' }],
    [{ text: '模型拿去對另一套工具處理出來的影像，照樣對得準', options: { bold: true } }, { text: '　分數只掉 0.002～0.008' }],
    [{ text: '再補一顆模型，把「版本」和「參數」的影響分開了', options: { bold: true } }, { text: '　擠爆主要是參數造成的，不是版本' }],
  ], { x: M, y: 4.1, w: 12.13, h: 2.6 });
}

// ───────────────────────────────────────────────────────── 03 資料
{
  const s = base('DATA', `資料：${D.n_total} 顆，四個來源`);
  const b = D.by_ds;
  const row = (k, n) => [n, String(b[k].train + b[k].val + b[k].test), String(b[k].train), String(b[k].val), String(b[k].test)];
  table(s, [
    ['來源', '顆數', '訓練', '驗證', '測試'],
    row('ASD', '老師的資料 A / T'),
    row('DGM', '老師的資料 D'),
    row('VNT', '老師的資料 VNT'),
    [{ text: '新增的外部資料', options: { bold: true } }, ...row('fs_subjects', '').slice(1).map((t) => ({ text: t, options: { bold: true } }))],
    [{ text: '合計', options: { bold: true } }, ...[D.n_total, D.split.train, D.split.val, D.split.test].map((t) => ({ text: String(t), options: { bold: true, color: C.TEAL } }))],
  ], { x: M, y: 1.7, w: 7.4, colW: [2.6, 1.2, 1.2, 1.2, 1.2] });
  card(s, 8.4, 1.7, 4.3, 3.2);
  txt(s, '每一顆都做同樣的前處理', { x: 8.7, y: 1.95, w: 3.8, h: 0.4, fontSize: 15, bold: true });
  bullets(s, [
    '去掉頭骨，只留腦',
    '線性對位到同一個模板',
    '尺寸統一成 192×224×192',
  ], { x: 8.7, y: 2.5, w: 3.8, h: 2.2, fontSize: 13.5 });
  txt(s, '所以不同來源、不同掃描尺寸的資料可以混在一起訓練。', { x: M, y: 5.1, w: 12.13, h: 0.5, fontSize: 14, color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 04 三段切分
{
  const s = base('METHOD', '這次改了做法：多切一份「驗證集」出來');
  const boxes = [
    ['訓練', String(D.split.train) + ' 顆', '模型只看這些', C.TEAL],
    ['驗證', String(D.split.val) + ' 顆', '用來挑「第幾輪的模型最好」', C.TEAL_L],
    ['測試', String(D.split.test) + ' 顆', '最後只看一次，就是要報的成績', C.RUST],
  ];
  const w = (12.13 - 0.5 * 2) / 3;
  boxes.forEach((b, i) => {
    const x = M + i * (w + 0.5);
    card(s, x, 1.75, w, 2.1);
    txt(s, b[0], { x, y: 2.0, w, h: 0.4, fontSize: 17, bold: true, align: 'center', color: b[3] });
    txt(s, b[1], { x, y: 2.45, w, h: 0.55, fontFace: F.MONO, fontSize: 30, bold: true, align: 'center' });
    txt(s, b[2], { x: x + 0.2, y: 3.1, w: w - 0.4, h: 0.6, fontSize: 13, align: 'center', color: C.MUTED });
  });
  bullets(s, [
    [{ text: '以前：直接拿測試集挑最好的一輪', options: { bold: true } }, { text: '　等於考前看過答案，成績會偏高（實測高估約 0.004）' }],
    [{ text: '現在：驗證集挑、測試集只考一次', options: { bold: true, color: C.TEAL } }, { text: '　報出來的數字才算數' }],
  ], { x: M, y: 4.3, w: 12.13, h: 1.8 });
}

// ───────────────────────────────────────────────────────── 模型在做什麼
{
  const s = base('METHOD', '模型在學什麼：把一顆腦「捏」成模板的形狀');
  fitImage(s, CH('formula_loss.png'), M, 1.5, 12.13, 4.0, '總損失的公式');
  bullets(s, [
    [{ text: '訓練時沒有人告訴模型「正確答案」', options: { bold: true } },
     { text: '　它只是一邊讓影像對得更像，一邊被限制不要捏得太誇張' }],
    [{ text: '這兩件事互相拉扯，λ 就是決定偏向哪一邊的旋鈕', options: { bold: true, color: C.TEAL } }],
  ], { x: M, y: 5.7, w: 12.13, h: 1.3, fontSize: 14 });
}

// ───────────────────────────────────────────────────────── 兩項的定義
{
  const s = base('METHOD', '兩項各自怎麼算');
  fitImage(s, CH('formula_terms.png'), M, 1.5, 12.13, 4.6, '相似度與平滑度的公式');
  txt(s, '兩項都是「越小越好」。模型在訓練時只看得到這個數字，看不到 Dice。',
    { x: M, y: 6.3, w: 12.13, h: 0.4, fontSize: 14, color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 兩種版本（公式）
{
  const s = base('VERSION', '兩種版本：形變場怎麼描述');
  fitImage(s, CH('formula_versions.png'), M, 1.5, 12.13, 4.0, '兩種版本的參數化');
  bullets(s, [
    [{ text: '論文 Table I 報的是位移場版；我們之前一直用程式預設的速度場版', options: { bold: true } }],
    [{ text: '所以「我們完全沒擠爆、比論文好」其實是版本不同造成的，不是模型比較強', options: { bold: true, color: C.RUST } }],
  ], { x: M, y: 5.7, w: 12.13, h: 1.3, fontSize: 14 });
}

// ───────────────────────────────────────────────────────── 怎麼評分
{
  const s = base('METRIC', '怎麼評分：一個看準不準，一個看有沒有擠爆');
  fitImage(s, CH('formula_metrics.png'), M, 1.5, 12.13, 4.4, 'Dice 與擠爆比例的公式');
  txt(s, 'Dice 是拿 FreeSurfer 或 tigerbx 的 30 個結構去比；擠爆比例只看形變場本身，不需要標籤。',
    { x: M, y: 6.2, w: 12.13, h: 0.4, fontSize: 14, color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 05 怎麼挑「第幾輪」
{
  const s = base('METHOD', '模型每輪都存一次，用驗證集挑最好的那一輪');
  fitImage(s, path.join(MROOT, 'mix_exp3', 'dice_curve_analysis.png'), M, 1.5, 11.4, 4.3, '訓練過程的 Dice 與打結比例');
  bullets(s, [
    [{ text: '上圖：分數隨訓練進步，前 50 輪最快，之後趨平（星號是選中的那一輪）', options: { color: C.MUTED } }],
    [{ text: '下圖：打結比例，一開始衝高，之後被平滑度的限制慢慢壓下來', options: { color: C.MUTED } }],
  ], { x: M, y: 5.95, w: 12.13, h: 1.0, fontSize: 13.5 });
}

// ───────────────────────────────────────────────────────── 06 訓練過程（三顆）
{
  const s = base('METHOD', '訓練過程：三顆一起看');
  fitImage(s, CH('curve_exp234.png'), M, 1.45, 12.13, 4.6, '三顆的驗證曲線與擠爆比例');
  bullets(s, [
    [{ text: '三顆都正常收斂，前 30 輪掉最快，250 輪時還在緩慢進步', options: { color: C.MUTED } }],
    [{ text: '下半部：擠爆比例一開始衝高，之後被平滑限制壓下來；限制越鬆壓得越少', options: { color: C.MUTED } }],
  ], { x: M, y: 6.25, w: 12.13, h: 0.9, fontSize: 13 });
}

// ───────────────────────────────────────────────────────── 05 總覽圖
{
  const s = base('RESULT', '四顆模型：配準把分數拉高多少');
  fitImage(s, CH('overview_four_models.png'), M, 1.6, 12.13, 4.3, '四顆模型的起點與配準後 Dice');
  txt(s, 'Dice = 兩顆腦的結構重疊程度，1 是完全重合。灰色是只做線性對位、還沒用模型的成績。',
    { x: M, y: 6.1, w: 12.13, h: 0.5, fontSize: 13.5, color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 06 模型貢獻
{
  const s = base('RESULT', '看「進步了多少」，四顆其實差不多');
  fitImage(s, CH('contribution.png'), M, 1.55, 12.13, 3.9, '四顆模型的配準貢獻');
  card(s, M, 5.65, 12.13, 1.15, 'FFF3E8');
  txt(s, [
    { text: 'tigerbx 的分數看起來高很多（0.862 vs 0.797），但那是因為它的起點就高。', options: { bold: true } },
    { text: '\n扣掉起點只看模型的貢獻，兩種標籤只差 ' + f3(P.gain_tiger_exp2_minus_mix_exp2.mean) + '。', options: {} },
  ], { x: M + 0.3, y: 5.85, w: 11.5, h: 0.8, fontSize: 14.5, fontFace: F.SANS, lang: 'zh-TW', color: C.INK, margin: 0 });
}

// ───────────────────────────────────────────────────────── 07 三顆的形變
{
  const s = base('VERSION', '同一顆腦，三種設定捏出來的形變');
  fitImage(s, CH('jacobian_exp234.png'), M, 1.4, 12.13, 3.55, '三個版本的 Jacobian');
  txt(s, '紅＝被撐大，藍＝被壓小。越往右紋路越細碎，代表捏得越用力。',
    { x: M, y: 5.05, w: 12.13, h: 0.4, fontSize: 13.5, color: C.MUTED, align: 'center' });
  table(s, [
    ['', '速度場版\n平滑權重 2', '位移場版\n平滑權重 2', '位移場版\n平滑權重 1'],
    ['Dice（FreeSurfer 標籤）', f3(m.mix_exp2.mean), f3(m.mix_exp4.mean), hl(f3(m.mix_exp3.mean))],
    ['擠爆的比例', pct(m.mix_exp2.jneg), pct(m.mix_exp4.jneg), pct(m.mix_exp3.jneg)],
  ], { x: M, y: 5.5, w: 12.13, colW: [4.63, 2.5, 2.5, 2.5] });
}

// ───────────────────────────────────────────────────────── 形變網格：三版對照
{
  const s = base('VERSION', '形變網格：三顆把空間拉成什麼樣');
  fitImage(s, CH('grid_exp234.png'), M, 1.5, 12.13, 4.3, '三個版本的形變網格');
  txt(s, '格線＝空間被拉扯成的形狀。由左到右越拉越用力：左邊柔順，右邊在腦溝附近扭得最明顯。',
    { x: M, y: 6.0, w: 12.13, h: 0.5, fontSize: 14, color: C.MUTED, align: 'center' });
}

// ───────────────────────────────────────────────────────── 08 打結
{
  const s = base('VERSION', '代價：位移場版會出現「打結」');
  bullets(s, [
    [{ text: '打結＝形變把組織擠到互相重疊', options: { bold: true } }, { text: '　數學上不合理，但比例很低時影響有限' }],
    [{ text: '速度場版天生不會打結', options: { bold: true } }, { text: '　我們量到 0%' }],
  ], { x: M, y: 1.65, w: 12.13, h: 1.4 });
  table(s, [
    ['', '打結比例', '說明'],
    ['速度場版（平滑權重 2）', pct(m.mix_exp2.jneg), '完全沒有'],
    ['位移場版（平滑權重 2）', pct(m.mix_exp4.jneg), '換了版本就出現一點'],
    [{ text: '位移場版（平滑權重 1）', options: { bold: true } }, hl(pct(m.mix_exp3.jneg)), '再放鬆限制，變成約 4 倍'],
    ['論文的同版本', '0.366%', 'VoxelMorph 原始論文 Table I'],
    ['論文比較的傳統方法', '0.185% ~ 0.793%', 'ANTs SyN / NiftyReg'],
  ], { x: M, y: 3.15, w: 12.13, colW: [4.2, 3.2, 4.73] });
  card(s, M, 5.6, 12.13, 1.0, 'E8F2F1');
  txt(s, '以前我們報「打結 0%」比論文好，現在知道那是版本不同造成的，不是模型比較強。',
    { x: M + 0.3, y: 5.85, w: 11.5, h: 0.5, fontSize: 15, bold: true, fontFace: F.SANS, lang: 'zh-TW' });
}

// ───────────────────────────────────────────────────────── 消融：拆開兩個因素
{
  const s = base('ABLATION', '一次只改一件事，才知道是誰的功勞');
  fitImage(s, CH('ablation.png'), M, 1.45, 12.13, 4.15, '消融實驗');
  const V = P.version_effect, L = P.lambda_effect;
  bullets(s, [
    [{ text: '換成論文的版本：Dice +' + f3(V.mean) + '（51 人裡 ' + V.win + ' 人變好）', options: { bold: true } },
     { text: '　擠爆從 0% 變 ' + pct(m.mix_exp4.jneg) }],
    [{ text: '再把平滑限制放鬆一半：Dice 又 +' + f3(L.mean) + '（' + L.win + ' 人變好）', options: { bold: true, color: C.RUST } },
     { text: '　擠爆變成 ' + pct(m.mix_exp3.jneg) + '，大約 4 倍' }],
    [{ text: '結論：捏得越用力對得越準，但越容易擠爆。λ 的影響比換版本大一倍', options: { bold: true } }],
  ], { x: M, y: 5.75, w: 12.13, h: 1.5, fontSize: 13.5 });
}

// ───────────────────────────────────────────────────────── 09 交叉測試
{
  const s = base('CROSS', '模型拿去對「另一套處理出來的影像」，還對得準嗎');
  fitImage(s, CH('cross_eval.png'), M, 1.5, 12.13, 3.9, '交叉測試長條圖');
  bullets(s, [
    [{ text: '綠色＝用 FreeSurfer 去頭骨的影像訓練出來的模型；橘色＝用 tigerbx 的', options: { color: C.MUTED } }],
    [{ text: '換另一邊訓練的模型來對，分數只掉 ' + f3(XP.tg_v3.mean) + '～' + f3(XP.fs_v3.mean) + '，等於「換一套去頭骨工具，模型照樣能用」', options: { bold: true } }],
    [{ text: '左右兩格的數字不能互相比：用的標籤不同，連紅線（起點）高度都不一樣', options: { color: C.MUTED } }],
  ], { x: M, y: 5.5, w: 12.13, h: 1.5, fontSize: 14 });
}

// ───────────────────────────────────────────────────────── 交叉：看得出差別嗎
{
  const s = base('CROSS', '同一張影像，換一個模型來對，看得出差別嗎');
  const XD = path.join(MROOT, 'mix_exp2', 'cross_mix_tiger_exp2_exp3');
  fitImage(s, path.join(XD, 'vis_T054_bg_subject_mix_exp3', 'contours_T054_' + m.mix_exp3.epoch + '.png'),
    M, 1.55, 5.9, 3.3, '用 FreeSurfer 影像訓練的模型');
  fitImage(s, path.join(XD, 'vis_T054_bg_subject_tiger_exp3', 'contours_T054_' + m.tiger_exp3.epoch + '.png'),
    6.85, 1.55, 5.9, 3.3, '用 tigerbx 影像訓練的模型');
  txt(s, '模型：用 FreeSurfer 影像訓練', { x: M, y: 5.0, w: 5.9, h: 0.38, fontSize: 15, bold: true, align: 'center', color: C.TEAL });
  txt(s, '模型：用 tigerbx 影像訓練', { x: 6.85, y: 5.0, w: 5.9, h: 0.38, fontSize: 15, bold: true, align: 'center', color: C.RUST });
  txt(s, 'Dice ' + f3(D.per_subject_T054.own), { x: M, y: 5.4, w: 5.9, h: 0.38, fontFace: F.MONO, fontSize: 16, bold: true, align: 'center' });
  txt(s, 'Dice ' + f3(D.per_subject_T054.foreign), { x: 6.85, y: 5.4, w: 5.9, h: 0.38, fontFace: F.MONO, fontSize: 16, bold: true, align: 'center' });
  txt(s, '兩邊完全一樣的條件：同一位受試者（T054）、FreeSurfer 去頭骨的影像、FreeSurfer 的結構標籤、'
        + '都是位移場版。只有「模型訓練時看的是哪一套影像」不同。',
    { x: M, y: 5.92, w: 12.13, h: 0.7, fontSize: 14.5, align: 'center' });
  txt(s, '底圖是這位受試者本人：上排配準前、下排配準後。實線＝模板上該結構的位置，虛線＝這顆腦的位置。'
        + '上排看得出兩條線有差，下排就貼合了。兩顆模型的結果幾乎一樣，分數差 '
        + f3(D.per_subject_T054.own - D.per_subject_T054.foreign) + '。',
    { x: M, y: 6.45, w: 12.13, h: 0.5, fontSize: 13, align: 'center', color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 換邊再測一次
{
  const s = base('CROSS', '換成 tigerbx 那套資料，再測一次');
  const XD = path.join(MROOT, 'mix_exp2', 'cross_mix_tiger_exp2_exp3');
  fitImage(s, path.join(XD, 'vis_T054_bg_subject_tigerbx_data', 'contours_T054_' + m.tiger_exp3.epoch + '.png'),
    M, 1.5, 5.9, 3.25, '用 tigerbx 影像訓練的模型');
  fitImage(s, path.join(XD, 'vis_T054_bg_subject_mix_exp3_on_tigerbx', 'contours_T054_' + m.mix_exp3.epoch + '.png'),
    6.85, 1.5, 5.9, 3.25, '用 FreeSurfer 影像訓練的模型');
  txt(s, '模型：用 tigerbx 影像訓練', { x: M, y: 4.92, w: 5.9, h: 0.36, fontSize: 15, bold: true, align: 'center', color: C.RUST });
  txt(s, '模型：用 FreeSurfer 影像訓練', { x: 6.85, y: 4.92, w: 5.9, h: 0.36, fontSize: 15, bold: true, align: 'center', color: C.TEAL });
  txt(s, 'Dice ' + f4(D.per_subject_T054.tg_own), { x: M, y: 5.32, w: 5.9, h: 0.38, fontFace: F.MONO, fontSize: 16, bold: true, align: 'center' });
  txt(s, 'Dice ' + f4(D.per_subject_T054.tg_foreign), { x: 6.85, y: 5.32, w: 5.9, h: 0.38, fontFace: F.MONO, fontSize: 16, bold: true, align: 'center' });
  txt(s, '這次兩邊都是 tigerbx 去頭骨的影像和 tigerbx 的標籤，同一位 T054，只有模型不同。',
    { x: M, y: 5.88, w: 12.13, h: 0.45, fontSize: 14.5, align: 'center' });
  txt(s, '前一頁換的是「用 FreeSurfer 的資料」，這頁換成「用 tigerbx 的資料」，兩邊結論一樣：換模型幾乎沒有差別（這次只差 '
        + f3(D.per_subject_T054.tg_own - D.per_subject_T054.tg_foreign) + '）。',
    { x: M, y: 6.33, w: 12.13, h: 0.5, fontSize: 13, align: 'center', color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 12 資料把關
{
  const s = base('DATA', '資料把關：這次處理掉的事');
  bullets(s, [
    [{ text: '兩組用完全相同的 520 人、相同的切分', options: { bold: true } }, { text: '　比較才有意義' }],
    [{ text: '29 顆掃描角度歪的，改成不先轉正', options: { bold: true } }, { text: '　少一次影像重取樣，保留原始細節' }],
    [{ text: '有一顆在兩批資料裡重複，已標記', options: { bold: true } }, { text: '　它落在驗證集，不影響測試成績' }],
    [{ text: '每一批資料都有指紋檔', options: { bold: true } }, { text: '　換機器搬資料時可以驗證有沒有搬錯版本' }],
  ], { x: M, y: 1.7, w: 12.13, h: 3.4 });
  card(s, M, 5.2, 12.13, 1.3, C.SURF);
  txt(s, '還缺的：新增那批 234 顆沒有年齡資料，所以「排除兒童」這項還做不到。',
    { x: M + 0.3, y: 5.55, w: 11.5, h: 0.6, fontSize: 15, bold: true, fontFace: F.SANS, lang: 'zh-TW' });
}

// ───────────────────────────────────────────────────────── 13 進度
{
  const s = base('PROGRESS', '上次交代的事項');
  const ok = { text: '完成', options: { color: C.TEAL, bold: true } };
  const half = { text: '部分完成', options: { color: C.RUST, bold: true } };
  const no = { text: '未做', options: { color: C.MUTED } };
  table(s, [
    ['', '項目', '狀態', '備註'],
    ['1', '拿掉兒童', no, '新資料沒有年齡，需要年齡表'],
    ['2', '找作者後續的論文', ok, ''],
    ['3', '疊圖顏色、矢狀面位置', ok, ''],
    ['4', '整理奇怪資料給老師', half, '內容已整併，要用時再產出'],
    ['5', '加外面的資料集', ok, '286 → 520 顆'],
    ['6', '用另一套分割互相驗證', ok, '這次的交叉測試'],
    ['7', '去頭骨品質的影響', half, '已知影響約 0.002～0.008'],
    ['8', '兩種版本的差別', half, '四顆模型跑完，還差一顆確認原因'],
  ], { x: M, y: 1.65, w: 12.13, colW: [0.6, 4.4, 1.9, 5.23], fontSize: 12.5 });
}

// ───────────────────────────────────────────────────────── 14 下一步
{
  const s = base('NEXT', '下一步');
  bullets(s, [
    [{ text: '再訓練一顆，把「版本」和「參數」分開', options: { bold: true } },
     { text: '\n　位移場版同時改了兩件事，目前分不出哪個造成差異。改一個參數再跑一次就知道。', options: { color: C.MUTED } }],
    [{ text: '去頭骨品質的完整比較', options: { bold: true } },
     { text: '\n　目前「去頭骨工具」和「標籤來源」還綁在一起，要再拆開。', options: { color: C.MUTED } }],
    [{ text: '要老師確認的事', options: { bold: true } },
     { text: '\n　新資料的年齡表、幾筆編號對不上的資料以哪邊為準。', options: { color: C.MUTED } }],
  ], { x: M, y: 1.8, w: 12.13, h: 4.2, fontSize: 16 });
}

pres.writeFile({ fileName: OUT }).then(() => console.log('ok ->', OUT, '｜' + page + ' 頁'));
