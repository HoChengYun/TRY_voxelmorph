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
const f4 = (x) => x.toFixed(4);
const sg = (x) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(2);
const pval = (p) => (p < 0.001 ? '< 0.001' : p.toFixed(3));

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'VoxelMorph：10/14 會議意見之回覆';
pres.author = 'HoChengYun';

const W = 13.333, M = 0.6;
let page = 0;
// 2026-10-21 使用者「開做」：加 ⑥ 架構修改之結果（2 × 2 完成、延長訓練）、⑧ 與 ANTs SyN 之比較、附錄（影像邊界之 folding）。
// 數字從 deck_data_arch.json 讀（make_arch_res.py）；ANTs 與 VoxelMorph 並排之受試者圖（make_ants_vis.py）有了才加那一頁
const DA_P = path.join(HERE, 'deck_data_arch.json');
const DA = fs.existsSync(DA_P) ? JSON.parse(fs.readFileSync(DA_P, 'utf8')) : null;
// 形變結果之比較（make_ants_vis.py）：冠狀、矢狀、水平各一頁＋各結構 Dice 一頁；圖或 json 有了才加那一頁
const VIS3 = [['coronal', 'ants_vis', '冠狀面（經過視丘）'], ['sagittal', 'ants_vis_sag', '矢狀面（中線往左 2 voxels）'],
  ['axial', 'ants_vis_ax', '水平面（經過基底核與視丘）']].filter(([v]) =>
  fs.existsSync(path.join(MROOT, 'deck_charts', '1028_ants_vis' + (v === 'coronal' ? '' : '_' + v) + '.png')));
const VDJ = path.join(MROOT, 'deck_charts', '1028_ants_vis_dice.json');
const VDICE = fs.existsSync(VDJ) ? JSON.parse(fs.readFileSync(VDJ, 'utf8')) : null;
// ⑥⑧ 與試作頁的三個架構修改模型（deck_models.json；換模型只改那個檔，見 換模型.md）。延長訓練、腦外 folding 兩頁（已關）講的是舊模型，照舊寫死
const DM = JSON.parse(fs.readFileSync(path.join(__dirname, 'deck_models.json'), 'utf8'));
const XC = DM.cascade.exp, XP = DM.c2f.exp, XM = DM.merged.exp;
// 2026-10-21 使用者同意拿掉附錄「影像邊界之 folding」：原作（VoxelMorph、DWP）也只算腦內，跟結論無關，留著反而要多解釋。
// 內容留在手冊 §28.4；要放回來就改成 true
const SHOW_BGFOLD = false;
// 2026-10-23 使用者：「P9 好像也不用了，給老師講太多頁老師會眼花」→ 延長訓練（mix_pyramid_500）那頁不放；內容留在手冊 §28
const SHOW_P500 = false;
// 10-23 論文常見之視覺化（試作；make_paper_vis.py）：放在最後給使用者看，確定後再搬到該放的地方
const SHOW_TRY = true;
const TRYJ = path.join(MROOT, 'deck_charts', '1028_try.json');
const TRY = SHOW_TRY && fs.existsSync(TRYJ) ? JSON.parse(fs.readFileSync(TRYJ, 'utf8')) : null;
const CN = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'];
// 10-23 使用者：「可以再多不同座標的」→ tigerbx 那頁之後，同一位受試者每個方向切 5 個位置各一頁（make_resvol.py 的 1028_resvol_slices_*.png）
const SHOW_TG_SLICES = true;
const TGS = [['rv_tg_cor', 'cor', '冠狀面'], ['rv_tg_sag', 'sag', '矢狀面'], ['rv_tg_ax', 'ax', '橫切面']]
  .filter(([, k]) => SHOW_TG_SLICES && D.tigerbx && D.tigerbx.slices && D.tigerbx.slices[k]);
// 10-27 使用者：「用其他工具 synthseg 來做去顱骨和分標籤」「給老師看看切出來的樣子和視覺長怎麼樣」「比 dice 就不用放了」
// → ⑨ 三頁：同一位受試者 FreeSurfer／tigerbx／SynthSeg 並排（make_synthseg.py → deck_data_ss.json）；只看影像與標籤，不算 Dice
const SS_P = path.join(HERE, 'deck_data_ss.json');
const SS = fs.existsSync(SS_P) ? JSON.parse(fs.readFileSync(SS_P, 'utf8')) : null;
// 10-28 使用者：「可以更新成新跑回來的」「然後加上 transmorph diff」→ 三個架構修改換成照原作的三顆（deck_models.json），
// 多兩頁：依原作程式補齊細節之影響（新舊對照＋拆開兩級，make_stage_diag.py）、TransMorph-diff（make_tmdiff.py）
const ST_P = path.join(HERE, 'deck_data_stage.json');
const STG = fs.existsSync(ST_P) ? JSON.parse(fs.readFileSync(ST_P, 'utf8')) : null;
const TD_P = path.join(HERE, 'deck_data_tmdiff.json');
const TMD = fs.existsSync(TD_P) ? JSON.parse(fs.readFileSync(TD_P, 'utf8')) : null;
// 10-28 使用者：「可以在 p11 前加上 cascade 和 pyramid 那兩篇的論文核心原理，讓老師和我熟悉一下」「結構圖也可以考慮用原文的」
// → ⑥ 前兩頁：RCN、DWP 之原文架構圖＋編號公式（make_principle.py）
const PR_OK = ['1028_pr_rcn_fig.png', '1028_pr_rcn_eq.png', '1028_pr_dwp_fig.png', '1028_pr_dwp_eq.png']
  .every((n) => fs.existsSync(path.join(MROOT, 'deck_charts', n)));
const ORDER = ['rv_def', 'rv_res', ...(D.models ? ['rv_models'] : []),
  ...(D.tigerbx ? ['rv_tiger_demo', ...TGS.map((v) => v[0]), ...(SS ? ['ss_label', 'ss_detail'] : []), ...(D.tigerbx.exp2 ? ['rv_tiger_exp2'] : [])] : []),
  ...(DA ? [...(PR_OK ? ['pr_rcn', 'pr_dwp'] : []), 'arch_2x2', ...(DA.train ? ['arch_train'] : []), 'arch_metrics',
    ...(TMD && DA.m.tm_diff_mixv2 ? ['arch_tmdiff'] : []), ...(SHOW_P500 ? ['arch_p500'] : []), 'ants_cmp', ...VIS3.map((v) => v[1]), ...(VDICE ? ['ants_vis_dice'] : []), ...(TRY ? ['try_box', 'try_lambda', 'try_errgrid', 'try_c2f'] : []), ...(SHOW_BGFOLD ? ['app_bgfold'] : []),
    ...(DA.p.new_vs_old && STG ? ['arch_rcndwp'] : [])] : [])];
const sgn = (x, d) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);
const thou = (x) => Math.round(x).toLocaleString('en-US');

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

const FOOT = { x: M, y: 6.78, w: 11.6, h: 0.3, fontSize: 10.5, color: C.MUTED };
// 10-28 使用者：「字好多，要不要減肥一下」→ 每頁條列 ≤ 3 條、每條約一行、註腳一行；拿掉的細節都在講稿.md 的備忘
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
  // 10-22 使用者：「簡報不用再放 Dice 變化了」→ 圖只剩 Dice 數值一格，放左邊、條列放右邊
  fitImage(s, CH('1028_resvol_result.png'), M, 1.35, 7.3, 4.75, '殘留體積與 Dice（affine、配準後）');
  bullets(s, [
    [{ text: '配準後 Dice 與殘留體積無關（r = ' + sg(E.r_after) + '，p = ' + pval(E.p_after) + '）', options: { bold: true } }],
    [{ text: 'Affine Dice 隨殘留略高（r = ' + sg(E.r_before) + '）：殘留多者起點略高，配準後與他人相同' }],
  ], { x: 8.15, y: 1.8, w: 4.58, h: 3.2, fontSize: 15, paraSpaceAfter: 14 });
  txt(s, 'Dice：' + [...new Set(R5.labs_name.split('、').map((n) => n.replace(/^[左右]/, '')))].join('、') + '（左右）之平均；' + D.model
    + '；n = ' + E.n + '（test ' + E.test.n + '、val ' + E.val.n + '、MRS ' + E.MRS.n + '，均未參與訓練）', FOOT);
}

if (D.models) {
  // ─────────────────────────────────────────────── 不同模型（10/14 紅字 p18：「想看不同 model 做出來的效果」）
  const Ms = Object.values(D.models);
  const rng = (k) => sg(Math.min(...Ms.map((m) => m[k]))) + '～' + sg(Math.max(...Ms.map((m) => m[k])));
  // 10-23：mix_cascade_pyramid 配準後 r = +0.26（p < 0.001，三批同方向：殘留多者略高）→ 不能寫「無關」，改寫「未因殘留而變差」
  // 10-28 換成照原作之三顆：cascade_rcn +0.16（p 0.04）、pyramid_dwp +0.23、合併 +0.32，皆為正 → 標題照舊
  // 10-28 加 TransMorph-diff：若有模型配準後 Dice 顯著隨殘留下降，標題與條列改寫
  const DOWN = Ms.filter((m) => m.r_after < 0 && m.p_after < 0.05);
  const short = (m) => m.name.replace(/（.*）/, '');
  const s = base(EB, DOWN.length ? '不同配準模型之比較：' + DOWN.map(short).join('、') + ' 之 Dice 隨殘留下降'
    : '不同配準模型之比較：配準後之 Dice 皆未因殘留而變差');
  fitImage(s, CH('1028_resvol_models.png'), M, 1.35, 12.13, 3.7, CN[Ms.length] + '個模型：殘留體積與 Dice（每位受試者一點；灰 affine、彩色配準後）');
  const UP = Ms.filter((m) => !DOWN.includes(m));
  bullets(s, [
    [{ text: DOWN.length
      ? DOWN.map((m) => short(m) + ' 配準後 r = ' + sg(m.r_after) + '（' + pval(m.p_after) + '）').join('；') + '；其餘' + CN[UP.length] + '個模型未隨殘留下降'
      : CN[Ms.length] + '個模型之配準後 Dice 皆未隨殘留下降（r = ' + rng('r_after') + '）', options: { bold: true } }],
    [{ text: '模型越好，整體越高；affine Dice 亦隨殘留略高（r = ' + sg(E.r_before) + '）' }],
  ], { x: M, y: 5.25, w: 12.13, h: 1.1, fontSize: 14, paraSpaceAfter: 6 });
  txt(s, 'Dice：大腦皮質、小腦皮質（左右）之平均；n = ' + E.n + '，殘留體積同前頁；灰：affine，彩色：配準後', FOOT);
}

if (D.tigerbx) {
  // ─────────────────────────────────────────────── tigerbx 之殘留（10/14 紅字 p12「換乾淨的 tbet（ex tigerbx）看 Dice 差多少」）
  // 2026-10-22 使用者：「看 tigerbx 那一套跑出來的效果和 freesurfer 差別」；數字：make_resvol.py 的 D.tigerbx
  const TG = D.tigerbx, PR = TG.pair;
  const s = base(EB, 'tigerbx 之去顱骨殘留：約為 FreeSurfer 之 ' + Math.round(100 * PR.ratio_median) + '%（同一批受試者）');
  fitImage(s, CH('1028_resvol_demo_compare.png'), M, 1.3, 6.9, 4.3, '同一位受試者：FreeSurfer 與 tigerbx 之殘留');
  fitImage(s, CH('1028_resvol_fs_vs_tigerbx.png'), 7.65, 1.35, 5.08, 2.75, '同一位受試者兩套之殘留體積');
  bullets(s, [
    [{ text: '殘留體積中位數 ' + PR.fs_median.toFixed(0) + ' → ' + PR.tg_median.toFixed(0) + ' cm³；' + PR.tg_less + '/' + PR.n
      + ' 位 tigerbx 較少', options: { bold: true } }],
    [{ text: 'tigerbx 僅剩腦表面外一薄層；FreeSurfer 較厚，並含顱底之組織' }],
  ], { x: 7.65, y: 4.35, w: 5.08, h: 2.2, fontSize: 14, paraSpaceAfter: 8 });
  txt(s, D.subject + '（左：上 FreeSurfer、下 tigerbx）；n = ' + PR.n + '（test、val）；tigerbx 以自身之標籤補成球', FOOT);
}

// ─────────────────────────────────────────────── 同一位受試者，每個方向 5 個位置（上 FreeSurfer、中 tigerbx、下 SynthStrip）
// 下面的觀察是 10-23 看圖寫的（只跟殘留有關，換模型不影響）；SynthStrip 那一列 10-27 加（使用者：「p17 可以併到 p5 6 7」）
const TG_OBS = {
  cor: 'FreeSurfer：殘留包覆整個腦表面，顱底（顳葉、小腦下方）最厚；tigerbx：僅腦表面外一薄層',
  sag: 'FreeSurfer：顱頂與顱底（額葉、顳葉、小腦下方）較厚；tigerbx：各位置皆較薄，以中線附近之顱頂與枕部較明顯',
  ax: '顱底（z = −40 mm）：FreeSurfer 之殘留填滿顳葉與小腦之間，tigerbx 幾乎沒有；顱頂（z = +60 mm）：FreeSurfer 較厚',
};
// 三套並排時（10-28 減字）：一條講完三套
const TG3_OBS = {
  cor: 'FreeSurfer 包覆整個腦表面，顱底最厚；SynthStrip 次之；tigerbx 僅腦表面外一薄層',
  sag: 'FreeSurfer 之顱頂與顱底較厚；SynthStrip 次之；tigerbx 各位置皆較薄',
  ax: '顱底（z = −40 mm）差異最大：FreeSurfer 填滿顳葉與小腦之間，SynthStrip 次之，tigerbx 幾乎沒有',
};
const R3 = !!(D.tigerbx && D.tigerbx.slice_rows && D.tigerbx.slice_rows.length === 3);
for (const [, k, nm] of TGS) {
  const SL = D.tigerbx.slices[k];
  const s = base(EB, R3 ? '三種去顱骨工具之殘留：' + nm + '五個位置（同一位受試者）'
    : '同一位受試者之殘留：' + nm + '五個位置（上 FreeSurfer、下 tigerbx）');
  const im = fitImage(s, CH('1028_resvol_slices_' + k + '.png'), M, 1.35, 12.13, R3 ? 4.7 : 4.55, D.subject + '：' + nm + '五個位置之殘留');
  bullets(s, [[{ text: (R3 ? TG3_OBS : TG_OBS)[k], options: { bold: true } }]],
    { x: M, y: 1.35 + im.h + 0.12, w: 12.13, h: 0.4, fontSize: 14, paraSpaceAfter: 2 });
  txt(s, D.subject + '；' + SL.axis + '：affine 後之 MNI 座標；黃線＝標籤、藍線＝補成之球、紅色＝殘留；右圖橘線＝切面位置；'
    + (k === 'sag' ? '影像右側＝前方' : '影像左側＝受試者左側'), FOOT);
}

if (D.tigerbx && SS) {
  // ─────────────────────────────────────────────── 第三套工具 SynthSeg 之標籤（make_synthseg.py；目前只有一位）
  // 10-27 使用者：「比 dice 就不用放了」；原本放在 ⑨（最後），同天使用者：「p18 p19 放到上面去講」→ 接在殘留五個位置之後
  const EB3 = '③ 第三套工具：SynthStrip＋SynthSeg（初步）';
  const V = SS.volume, cm = (x) => Math.round(x).toLocaleString('en-US');
  const pc = (n, t) => Math.round(100 * V[n][t] / V[n].FreeSurfer) + '%';
  const SSFOOT = SS.subject + '（test）；SynthStrip＋SynthSeg 2.0（FreeSurfer 7.4.1）；受試者原始空間';
  {
    const s = base(EB3, '分割（SynthSeg）：編號同 FreeSurfer，腦表面外之腦脊髓液亦標為 CSF');
    const g = fitImage(s, CH('1028_ss_label.png'), M, 1.3, 6.4, 5.4, SS.subject + '：三種工具之標籤');
    const NL = SS.n_labels;
    bullets(s, [
      [{ text: '編號同 FreeSurfer aseg；SynthSeg ' + NL.SynthSeg + ' 個標籤（' + (NL.FreeSurfer === NL.tigerbx ? 'FreeSurfer、tigerbx ' + NL.FreeSurfer
        : 'FreeSurfer ' + NL.FreeSurfer + '、tigerbx ' + NL.tigerbx) + ' 個）', options: { bold: true } }],
      [{ text: 'SynthSeg 無脈絡叢、胼胝體、血管等，併入鄰近結構' }],
      [{ text: 'CSF：SynthSeg ' + cm(SS.csf.SynthSeg) + ' cm³（含腦溝與腦表面外），FreeSurfer、tigerbx 約 ' + SS.csf.FreeSurfer.toFixed(0) + ' cm³' }],
    ], { x: M + g.w + 0.35, y: 1.6, w: 12.13 - g.w - 0.35, h: 4.8, fontSize: 14, paraSpaceAfter: 12 });
    txt(s, SSFOOT + '；顏色依 FreeSurfer LUT（CSF 改淺藍）', FOOT);
  }
  {
    const s = base(EB3, '深部結構之細部與各結構體積（同一位受試者）');
    fitImage(s, CH('1028_ss_zoom.png'), M, 1.3, 12.13, 2.85, SS.subject + '：冠狀面經過海馬迴，深部結構放大');
    fitImage(s, CH('1028_ss_volume.png'), M, 4.2, 6.6, 2.55, '各結構體積（cm³）');
    bullets(s, [
      [{ text: 'SynthSeg 邊界較平滑；蒼白球較小（' + pc('Pallidum', 'SynthSeg') + '），殼核、尾核、杏仁核、依核較大', options: { bold: true } }],
      [{ text: 'tigerbx 與 FreeSurfer 較接近（依核以外皆在 ±15% 內）' }],
    ], { x: 7.45, y: 4.4, w: 5.28, h: 2.2, fontSize: 14, paraSpaceAfter: 10 });
    txt(s, SSFOOT + '（單一受試者，僅供目視參考）；體積為左右合計，百分比以 FreeSurfer 為 100%', FOOT);
  }
}


if (D.tigerbx && D.tigerbx.exp2) {
  // ─────────────────────────────────────────────── FreeSurfer vs tigerbx（exp2，SVF）
  // 2026-10-22 使用者：「FreeSurfer 後面跑的很多實驗 tigerbx 沒跟到，老師比較喜歡速度場，就拿 exp2 來比較」
  const F = D.tigerbx.exp2.fs, G = D.tigerbx.exp2.tg;
  const pt = (p) => (p < 0.001 ? 'p < 0.001' : 'p = ' + p.toFixed(2));
  const s = base(EB, 'FreeSurfer 與 tigerbx（SVF）：配準結果皆未因殘留而變差');
  fitImage(s, CH('1028_resvol_exp2_compare.png'), M, 1.35, 7.4, 4.9, 'mix_exp2 與 tiger_exp2：殘留體積與 Dice（灰 affine）');
  // 「兩者之差」的 p：配準後與 affine 之差對殘留體積之 Spearman（講稿備忘有寫）
  bullets(s, [
    [{ text: 'FreeSurfer：配準後 Dice 與殘留無關（r = ' + sg(F.r_after) + '）', options: { bold: true } }],
    [{ text: 'tigerbx：配準後 r = ' + sg(G.r_after) + '，但 affine 已有同方向之斜率，兩者之差不顯著（' + pt(G.p) + '）' }],
    [{ text: '→ 屬受試者本身之差異，非配準受殘留影響', options: { color: C.TEAL, bold: true } }],
  ], { x: 8.2, y: 1.6, w: 4.53, h: 4.8, fontSize: 14, paraSpaceAfter: 12 });
  txt(s, 'n = ' + F.n + '（test、val）；mix_exp2、tiger_exp2 設定相同（SVF，λ = 1，250 epochs）；兩套標籤不同，Dice 絕對值不可直接比較', FOOT);
}
const wp = (q) => '（' + q.win + '/' + q.n + '）';
const pv = (q) => (q.p < 0.001 ? 'p < 0.001' : 'p = ' + q.p.toFixed(3));
const EB6 = '⑥ 架構修改之結果（紅字以外）';
const EB8 = '⑧ 與傳統方法之比較（紅字以外）';

if (DA && PR_OK) {
  // ─────────────────────────────────────────────── ⑥ 兩篇原作之核心原理（10-28；圖取自原文，公式含原作程式之細節；make_principle.py）
  const EBP = '⑥ 架構修改：兩篇原作之核心原理';
  {
    const s = base(EBP, 'Cascade（RCN）：以多個子網路遞迴修正剩餘之形變');
    const g = fitImage(s, CH('1028_pr_rcn_fig.png'), M, 1.3, 12.13, 2.75, 'RCN 之架構（Zhao et al., ICCV 2019, Fig. 3）');
    fitImage(s, CH('1028_pr_rcn_eq.png'), M, 1.3 + g.h + 0.12, 12.13, 2.45, '式 (1)(2)');
    txt(s, '圖：Zhao et al., Recursive Cascaded Networks for Unsupervised Medical Image Registration, ICCV 2019, Fig. 3（○：合成）', FOOT);
  }
  {
    const s = base(EBP, 'Coarse-to-fine（DWP）：雙流特徵金字塔，逐層 warp 並合成形變');
    fitImage(s, CH('1028_pr_dwp_fig.png'), M, 1.3, 6.2, 3.3, 'DWP 之架構（Jian et al., WBIR 2024, Fig. 1）');
    bullets(s, [
      [{ text: 'Baseline（VoxelMorph）：兩張影像於輸入端串接，於原解析度輸出一次形變' }],
      [{ text: 'Dual：m、f 以權重共享之 encoder 分別抽取特徵金字塔', options: { bold: true } }],
      [{ text: 'Warping＋Pyramid：每層以上一層之形變 warp m 之特徵，估計剩餘形變後合成；大位移於粗層僅數個 voxel',
        options: { bold: true } }],
    ], { x: 7.05, y: 1.5, w: 5.68, h: 3.1, fontSize: 14, paraSpaceAfter: 10 });
    fitImage(s, CH('1028_pr_dwp_eq.png'), M, 4.68, 12.13, 2.05, '式 (3)(4)');
    txt(s, '圖：Jian et al., Mamba? Catch the Hype or Rethink What Really Helps for Image Registration, WBIR 2024, Fig. 1（I_s = m、I_t = f）', FOOT);
  }
}

if (DA) {
  // ─────────────────────────────────────────────── ⑥ 2 × 2 結果（10/14 時 coarse-to-fine 與合併模型尚在訓練）
  const m = DA.m, P = DA.p, PR = DA.params;
  // 10-28 換成照原作之三顆：合併模型低於單獨之 cascade、coarse-to-fine（舊版合併最高）→ 標題與條列依數字判斷
  const MBEST = m[XM].dice > Math.max(m[XC].dice, m[XP].dice);
  const s = base(EB6, MBEST ? '2 × 2 實驗結果：cascade 與 coarse-to-fine 合併之 Dice 最高'
    : '2 × 2 實驗結果：兩項設計皆提升 Dice，合併後未再提升');
  const cell = (e, name) => {
    const q = P[e + '_vs_exp6'];
    const runs = [{ text: name, options: { breakLine: true, fontSize: name.length > 20 ? 9.5 : 11.5, color: C.MUTED } },
                  { text: 'Dice ' + f3(m[e].dice), options: { bold: true, breakLine: true } }];
    // 10-22 使用者：「簡報不用再放 Dice 變化了，希望用 Dice 來比較」→ 只寫幾位高於 mix_exp6
    if (q) runs.push({ text: '高於 mix_exp6：' + q.dice.win + '/' + q.dice.n + ' 位', options: { fontSize: 10.5, color: C.TEAL, breakLine: true } });
    runs.push({ text: (PR[e] / 1e6).toFixed(2) + ' M params', options: { fontSize: 11.5, color: C.MUTED } });
    return { text: runs };
  };
  table(s, [
    ['', 'Single stage', 'Cascade（2 stages）'],
    [{ text: 'Single-resolution\n（VoxelMorph）', options: { bold: true, fontSize: 11.5 } }, cell('mix_exp6', 'mix_exp6'), cell(XC, XC)],
    [{ text: 'Coarse-to-fine', options: { bold: true } }, cell(XP, XP),
      cell(XM, XM)],
  ], { x: M, y: 1.5, w: 5.75, colW: [1.7, 1.9, 2.15], fontSize: 12.5, rowH: [0.4, 1.45, 1.45] });
  fitImage(s, CH('1028_arch_2x2.png'), 6.5, 1.35, 6.23, 3.75, '每位受試者之 Dice（含 TransMorph-diff、半監督）');
  const A = DA.additive, TM = DA.m.tm_mixv2;
  const pmax = Math.max(P.cp_vs_cascade.p, P.cp_vs_pyramid.p);
  // TransMorph 與三種架構修改逐一比較：p < 0.05 者為「低於」，其餘為「無顯著差異」
  const TMV = [[XC, 'cascade'], [XP, 'coarse-to-fine'], [XM, '合併模型']].map(([e, nm]) => [nm, P['tm_vs_' + e]]);
  const TLO = TMV.filter(([, q]) => q.p < 0.05 && q.mean < 0), TNS = TMV.filter(([, q]) => q.p >= 0.05);
  bullets(s, [
    MBEST
      ? [{ text: '合併模型 ' + f3(m[XM].dice) + ' 最高：高於 cascade ' + f3(m[XC].dice) + '、coarse-to-fine ' + f3(m[XP].dice) + '（'
        + (pmax < 0.001 ? 'p < 0.001' : 'p ≤ ' + pmax.toFixed(3)) + '）', options: { bold: true } }]
      : [{ text: 'Cascade ' + f3(m[XC].dice) + '、coarse-to-fine ' + f3(m[XP].dice) + (P.pyramid_vs_cascade.p >= 0.05 ? '（無顯著差異）' : '')
        + '；合併 ' + f3(m[XM].dice) + '，低於兩者（' + (pmax < 0.001 ? 'p < 0.001' : 'p ≤ ' + pmax.toFixed(3)) + '）', options: { bold: true } }],
    [{ text: (MBEST ? '兩項效果部分可加成' : '兩項效果未能加成') + '：完全相加應為 ' + f3(m.mix_exp6.dice + A.cascade + A.pyramid) + '（虛線）'
      + (MBEST ? '' : '，原因見附錄（最後一頁）') }],
    // 10-23：TransMorph（AI 帶回）併入此頁，不另開一頁
    [{ text: 'TransMorph（' + (DA.params.tm_mixv2 / 1e6).toFixed(1) + ' M params）' + f3(TM.dice) + '：'
      + (TLO.length ? '低於 ' + TLO.map((v) => v[0]).join('、') : '') + (TNS.length ? (TLO.length ? '，' : '') + '與' + TNS.map((v) => v[0]).join('、') + '相當' : ''),
      options: { color: '4B5D73' } }],
  ], { x: M, y: 5.2, w: 12.13, h: 1.5, fontSize: 14, paraSpaceAfter: 6 });
  txt(s, 'test ' + DA.n + ' 位，30 結構平均；四格之訓練設定相同（SVF，λ = 1，250 epochs）；TransMorph 為 displacement field；'
    + '* 半監督（淡紫底色）之標籤項即為評估之 30 結構', FOOT);
}

if (DA && DA.train) {
  // ─────────────────────────────────────────────── ⑥ 訓練過程（10-22 使用者：「loss 和 train 的圖可以放簡報」「分開放，train 和 val 放一起」；
  // val loss 下次才有（test_dice.py 10-22 加 ncc_loss 欄），這次用訓練之相似度項＋val Dice）
  const TR = DA.train, ks = ['mix_exp6', 'tm_mixv2', XC, XP, XM];
  // 10-28 照原作之合併模型：val Dice 最高在 epoch 155，200～250 之平均略低於 150～200 → 標題與條列依數字判斷
  const DN = ks.filter((k) => TR[k].val_win['200-250'] < TR[k].val_win['150-200']);
  const s = base(EB6, DN.length ? '訓練過程：訓練損失持續下降，val Dice 約 150 epochs 後趨於持平'
    : '訓練過程：訓練損失與 val Dice 至 250 epochs 仍緩慢改善');
  fitImage(s, CH('1028_train_curves.png'), M, 1.3, 12.13, 4.0, '八個模型之訓練損失（相似度項）與 val Dice（含 TransMorph-diff、半監督）');
  const NM = { mix_exp6: 'mix_exp6', tm_mixv2: 'TransMorph', [XC]: 'cascade', [XP]: 'coarse-to-fine', [XM]: '合併模型' };
  const LOW = [XC, XP, XM].every((k) => TR[k].img['250'] < Math.min(TR.mix_exp6.img['250'], TR.tm_mixv2.img['250']));
  bullets(s, [
    [{ text: LOW ? '相似度項（epoch 250）：三種架構修改皆低於 mix_exp6、TransMorph'
      : '相似度項（epoch 250）：' + ks.map((k) => NM[k] + ' ' + sgn(TR[k].img['250'], 3)).join('、'), options: { bold: true } }],
    [{ text: DN.length ? 'val Dice 約 150 epochs 後持平；' + DN.map((k) => NM[k]).join('、') + ' 200～250 略降（'
      + DN.map((k) => (TR[k].val_win['150-200'] - TR[k].val_win['200-250']).toFixed(4)).join('、') + '），未見明顯 overfitting'
      : 'val Dice 至 250 epochs 仍緩慢上升，未見 overfitting' }],
    // 10-28 使用者：「p14 也可以補上半監督」→ 半監督之相似度項高於 mix_exp6（標籤項使影像對齊略差）、val Dice 至 250 epochs 仍上升
    ...(TR.mix_exp8 && TR.mix_exp9 ? [[{ text: '* 半監督：相似度項 ' + sgn(TR.mix_exp8.img['250'], 3) + '／' + sgn(TR.mix_exp9.img['250'], 3)
      + '（γ = 0.5／5）' + ((TR.mix_exp8.img['250'] > TR.mix_exp6.img['250'] && TR.mix_exp9.img['250'] > TR.mix_exp6.img['250'])
        ? '，高於 mix_exp6（' + sgn(TR.mix_exp6.img['250'], 3) + '）：影像之對齊讓給標籤項' : '')
      + (['mix_exp8', 'mix_exp9'].every((k) => TR[k].val_win['200-250'] > TR[k].val_win['150-200']) ? '；val Dice 至 250 epochs 仍上升' : ''),
      options: { color: '8E5B8A' } }]] : []),
  ], { x: M, y: 5.4, w: 12.13, h: 1.3, fontSize: 13.5, paraSpaceAfter: 4 });
  txt(s, 'λ = 1，250 epochs；相似度項 = −NCC（窗格 9）；★ = val Dice 最高之 epoch；TransMorph-diff 之訓練 log 未帶回，僅畫 val Dice；'
    + '* 半監督之標籤項即為評估之 30 結構', FOOT);
}

if (DA) {
  // ─────────────────────────────────────────────── ⑥ 多項指標
  // 10-28 使用者：「就拿論文裡 30 個結構就好」「把半監督的數字加進來看」→ 拿掉「未參與訓練之結構」，圖上加兩列半監督（*）
  const m = DA.m, P = DA.p, AR = [XC, XP, XM];
  const NMA = { [XC]: 'cascade', [XP]: 'coarse-to-fine', [XM]: '合併模型' };
  const V6 = AR.map((e) => P[e + '_vs_exp6']);
  const HM = V6[2].hd95;                                        // 合併模型 vs mix_exp6 之 HD95
  const s = base(EB6, (HM.p >= 0.05 || HM.mean > 0) ? '多項指標：架構修改皆優於 mix_exp6；合併模型之 HD95 未改善'
    : '多項指標：架構修改之 Dice、HD95 皆優於 mix_exp6');
  fitImage(s, CH('1028_arch_res.png'), M, 1.3, 12.13, 4.3, '九個模型之 Dice、HD95、SDlogJ、folding');
  const lowH = AR.filter((e, i) => V6[i].hd95.p < 0.05 && V6[i].hd95.mean < 0), nsH = AR.filter((e, i) => V6[i].hd95.p >= 0.05);
  const span = (k, f) => { const v = AR.map((e) => m[e][k]); return f(Math.min(...v)) + '～' + f(Math.max(...v)); };
  bullets(s, [
    [{ text: 'HD95：' + lowH.map((e) => NMA[e] + ' ' + m[e].hd95.toFixed(2)).join('、') + ' mm 低於 mix_exp6（' + m.mix_exp6.hd95.toFixed(2) + '）'
      + (nsH.length ? '；' + nsH.map((e) => NMA[e] + ' ' + m[e].hd95.toFixed(2)).join('、') + '，無顯著差異' : ''), options: { bold: true } }],
    [{ text: '代價：SDlogJ ' + m.mix_exp6.sdlogj.toFixed(2) + ' → ' + span('sdlogj', (x) => x.toFixed(2)) + '；腦內 folding ' + m.mix_exp6.fold.toFixed(1) + ' → '
      + span('fold', (x) => String(Math.round(x))) + ' voxels／位（論文 VoxelMorph (CC) ' + thou(DA.paper.vxm_fold) + '）', options: { color: C.RUST } }],
    [{ text: '* 半監督：Dice ' + f3(m.mix_exp8.dice) + '／' + f3(m.mix_exp9.dice) + '、HD95 ' + m.mix_exp8.hd95.toFixed(2) + '／' + m.mix_exp9.hd95.toFixed(2)
      + ' mm（γ = 0.5／5）；標籤項即為評估之 30 結構，不可直接比較', options: { color: '8E5B8A' } }],
  ], { x: M, y: 5.62, w: 12.13, h: 1.1, fontSize: 13, paraSpaceAfter: 3 });
  txt(s, 'test ' + DA.n + ' 位之平均；Dice、HD95 為 30 結構之平均；SDlogJ、folding 僅計 atlas 腦內；超出範圍者以箭頭標示', FOOT);
}

if (TMD && DA && DA.m.tm_diff_mixv2) {
  // ─────────────────────────────────────────────── ⑥ TransMorph-diff（10-28 AI 帶回；TransMorph 專案 mixv2/train_mixv2_diff.py，照官方訓練方式）
  const TD = DA.m.tm_diff_mixv2, TS = TMD.struct, TI = TMD.intensity, Pd = DA.p;
  const s = base(EB6, 'TransMorph-diff（官方訓練方式）：Dice ' + f3(TD.dice) + '，小腦低於 affine');
  fitImage(s, CH('1028_tmdiff.png'), M, 1.3, 7.3, 5.4, 'TransMorph-diff：小腦之配準與各結構 Dice');
  const tr = (nm, k) => [nm, TI[k].atlas.toFixed(2), TI[k].sub.toFixed(2) + ' ± ' + TI[k].sub_sd.toFixed(2)];
  table(s, [
    ['結構（影像值 0～1）', 'Atlas', '受試者（' + DA.n + ' 位）'],
    tr('大腦白質', 'Cerebral-WM'), tr('大腦皮質', 'Cerebral-Cortex'),
    [{ text: '小腦白質', options: { color: C.RUST, bold: true } }, TI['Cerebellum-WM'].atlas.toFixed(2),
      { text: TI['Cerebellum-WM'].sub.toFixed(2) + ' ± ' + TI['Cerebellum-WM'].sub_sd.toFixed(2), options: { color: C.RUST, bold: true } }],
    [{ text: '小腦皮質', options: { color: C.RUST, bold: true } }, TI['Cerebellum-Cortex'].atlas.toFixed(2),
      { text: TI['Cerebellum-Cortex'].sub.toFixed(2) + ' ± ' + TI['Cerebellum-Cortex'].sub_sd.toFixed(2), options: { color: C.RUST, bold: true } }],
    tr('側腦室', 'Lateral-Vent'),
  ], { x: 8.15, y: 1.4, w: 4.58, colW: [2.0, 0.8, 1.78], fontSize: 11.5, rowH: 0.34 });
  bullets(s, [
    [{ text: '小腦 Dice 低於 affine：皮質 ' + f3(TS['Cerebellum-Cortex'].affine) + ' → ' + f3(TS['Cerebellum-Cortex'].tm_diff) + '、白質 '
      + f3(TS['Cerebellum-WM'].affine) + ' → ' + f3(TS['Cerebellum-WM'].tm_diff), options: { bold: true } }],
    [{ text: '原因：官方相似度為 MSE；受試者之小腦較 atlas 暗（上表），MSE 迫使亮度一致而錯位' }],
    [{ text: '腦內 folding ' + thou(TD.fold) + ' voxels／位：官方積分前未除以 2⁷', options: { color: C.MUTED } }],
  ], { x: 8.15, y: 3.75, w: 4.58, h: 2.9, fontSize: 13, paraSpaceAfter: 8 });
  txt(s, '官方 TransMorphDiff（' + (DA.params.tm_diff_mixv2 / 1e6).toFixed(1) + ' M params；MSE＋KL、Adamax、lr 遞減、三向翻轉），訓練量改為 2.5 萬步；val 選定 epoch '
    + DA.train_tmd_best, FOOT);
}

if (DA && SHOW_P500) {
  // ─────────────────────────────────────────────── ⑥ 延長訓練（mix_pyramid 250 → 500 epochs）
  const W5 = DA.p500.win, P = DA.p;
  const s = base(EB6, '延長訓練（250 → 500 epochs）：val Dice 約 300 epochs 後持平');
  fitImage(s, CH('1028_p500.png'), M, 1.35, 12.13, 3.9, 'coarse-to-fine 延長訓練之 val Dice 與訓練損失');
  bullets(s, [
    [{ text: 'val Dice 區間平均：200–250 epochs ' + W5['200-250'].toFixed(4) + ' → 300–400 ' + W5['300-400'].toFixed(4) + ' → 400–500 '
      + W5['400-500'].toFixed(4) + '；訓練損失持續下降（' + sgn(DA.p500.img_loss['250'], 4) + ' → '
      + sgn(DA.p500.img_loss['500'], 4) + '）', options: { bold: true } }],
    [{ text: 'test（val 選定 epoch ' + DA.p500.best500[0] + '）：Dice ' + DA.m.mix_pyramid_500.dice.toFixed(4) + '，較 250 epochs 之模型 '
      + sgn(P.p500_vs_pyramid.mean, 4) + wp(P.p500_vs_pyramid) + '；與合併模型（' + DA.m.mix_cascade_pyramid.dice.toFixed(4) + '）相近（' + sgn(-P.cp_vs_p500.mean, 4) + '，'
      + pv(P.cp_vs_p500) + '）' }],
    [{ text: '合併模型僅訓練 250 epochs，200–250 epochs 之 val Dice 仍上升（' + DA.p500.cp_win['150-200'].toFixed(4) + ' → '
      + DA.p500.cp_win['200-250'].toFixed(4) + '）；兩者訓練長度不同，比較時需註明', options: { color: C.RUST } }],
  ], { x: M, y: 5.32, w: 12.13, h: 1.4, fontSize: 13.5, paraSpaceAfter: 5 });
  txt(s, '自 mix_pyramid 之 epoch 250 權重接續訓練，其餘設定不變；optimizer（Adam）狀態未保存，接續時重新累積；訓練時間 '
    + DA.hours.mix_pyramid_500.toFixed(1) + ' h（TITAN RTX）', FOOT);
}

if (DA) {
  // ─────────────────────────────────────────────── ⑧ ANTs SyN 與論文 Table I
  const m = DA.m, T = DA.paper, P = DA.p;
  const s = base(EB8, 'ANTs SyN 與 VoxelMorph 相當：與 VoxelMorph 論文之關係一致');
  fitImage(s, CH('1028_ants_paper.png'), M, 1.35, 6.2, 4.1, '論文 Table I 與本研究之 Dice');
  table(s, [
    ['', '論文 Table I', '本研究（test ' + DA.n + ' 位）'],
    ['Affine', f3(T.affine), f3(m.affine.dice)],
    // 10-22 使用者：「簡報不用再放 Dice 變化了」→ 拿掉括號內之增量與「VoxelMorph − ANTs」列
    ['ANTs SyN', f3(T.ants), f3(m.ants.dice)],
    ['VoxelMorph', f3(T.vxm), f3(m.mix_exp3.dice)],
    // 10-28 減字：TransMorph、TransMorph-diff 只放左圖，表格不重複
    ['VxM 較高者', '—', P.mix_exp3_vs_ants.win + '/' + P.mix_exp3_vs_ants.n + ' 位（' + pv(P.mix_exp3_vs_ants) + '）'],
    // 2026-10-21 使用者：「應該要說為啥他們 ants 有東西」→ 論文 §V-B：ANTs、NiftyReg 能強制／強烈鼓勵微分同胚的設定會損及時間或結果，
    // 故以 Dice 最佳之設定執行（未強制微分同胚）；本研究照論文列出之參數、以 ANTsPy 預設之 SyN 實作執行
    ['Folding voxels',
      { text: [{ text: 'ANTs ' + thou(T.ants_fold) + '／VxM ' + thou(T.vxm_fold), options: { breakLine: true } },
               { text: '採 Dice 最佳設定，未強制微分同胚（§V-B）', options: { fontSize: 9.5, color: C.MUTED } }] },
      { text: [{ text: 'ANTs ' + thou(m.ants.fold) + '／VxM ' + thou(m.mix_exp3.fold), options: { breakLine: true } },
               { text: 'ANTsPy 預設 SyN（微分同胚）', options: { fontSize: 9.5, color: C.MUTED } }] }],
    ['時間／位', 'ANTs ' + thou(T.ants_sec) + ' s（CPU）', 'ANTs ' + thou(DA.ants.sec_mean) + ' s（CPU，' + DA.ants.threads + ' threads）'],
  ], { x: 7.0, y: 1.45, w: 5.73, colW: [1.6, 2.13, 2.0], fontSize: 11.5, rowH: 0.4 });
  const WA = [XC, XP, XM, 'tm_mixv2'].map((e) => P[e + '_vs_ants'].win);
  const TA = P.tm_diff_mixv2_vs_ants;
  bullets(s, [
    [{ text: '本研究 Dice 較高，主因為 affine 即較高（' + f3(m.affine.dice) + ' vs ' + f3(T.affine) + '）；ANTs 亦得相近之 Dice → 本資料集較易配準',
       options: { bold: true } }],
    [{ text: '架構修改與 TransMorph 皆高於 ANTs SyN（' + Math.min(...WA) + '～' + Math.max(...WA) + '/' + DA.n + ' 位）'
      + (TA ? '；TransMorph-diff 則 ' + (TA.n - TA.win) + '/' + TA.n + ' 位低於 ANTs' : ''), options: { color: C.TEAL, bold: true } }],
  ], { x: M, y: 5.55, w: 12.13, h: 1.1, fontSize: 14, paraSpaceAfter: 6 });
  txt(s, 'ANTs SyN 參數依論文 §V-A-3；affine 已於前處理完成，SyN 僅做非線性部分；VoxelMorph = mix_exp3（displacement field，λ = 1）', FOOT);
}

// 10-23 使用者：「放一下好了（單獨 cascade、coarse-to-fine）」「再放 VoxelMorph 速度場」「還有半監督」「把 table 放到其他 page」
// 「同方向 log|J| 和形變後影像放一頁」→ 冠狀、矢狀、水平各一頁（make_ants_vis.py：4 列 × 6 欄，兩段各「影像＋log|J|」），Dice 表另一頁
for (const [view, , title] of (DA ? VIS3 : [])) {
  // ─────────────────────────────────────────────── ⑧ 形變結果之比較（一個方向一頁）
  const s = base(EB8, '形變結果之比較：' + title);
  fitImage(s, CH('1028_ants_vis' + (view === 'coronal' ? '' : '_' + view) + '.png'), M, 1.3, 12.13, 5.4, '各方法之形變結果（' + title + '）');
  txt(s, 'T054；' + { coronal: '受試者左側在影像左側', sagittal: '中線往左 2 voxels，影像右側＝前方', axial: '受試者左側在影像左側' }[view]
    + '；VoxelMorph（disp.）= mix_exp3、（SVF）= mix_exp6；左下數字＝腦內 |J| ≤ 0 voxel 數', FOOT);
}

if (DA && VDICE) {
  // ─────────────────────────────────────────────── ⑧ 各結構之 Dice（T054；三個方向畫到的結構）
  const VD = VDICE, SD = VD.struct_dice, NS = Object.keys(SD), ks = VD.methods, SEMI = new Set(VD.semi);
  const s = base(EB8, '形變結果之比較：各結構之 Dice（T054）');
  // 粗體＝各列最高（不含 affine 與半監督：半監督之標籤項即為這些結構）
  const cand = ks.filter((k) => k !== 'moving' && !SEMI.has(k));
  // 10-28 加 TransMorph-diff 後 11 欄，表頭字小一點，「VoxelMorph」才不會被拆成兩行
  const HF = ks.length > 10 ? 10.5 : 12;
  const runs = (t) => t.split('\n').map((x, i, a) => ({ text: x, options: Object.assign({ fontSize: HF }, i < a.length - 1 ? { breakLine: true } : {}) }));
  const head = ['結構'].concat(ks.map((k) => runs(k === 'moving' ? 'Affine' : VD.names[k] + (SEMI.has(k) ? ' *' : ''))));
  const rows = [head].concat(NS.map((n) => [n].concat(ks.map((k) => {
    const v = SD[n][k], best = Math.max(...cand.map((c) => SD[n][c]));
    return { text: f3(v), options: SEMI.has(k) ? { color: C.MUTED } : (cand.includes(k) && v === best ? { bold: true } : {}) };
  }))));
  const cw = (12.13 - 1.3) / ks.length;
  table(s, rows, { x: M, y: 1.4, w: 12.13, colW: [1.3].concat(ks.map(() => cw)), fontSize: 12, rowH: 0.34,
    margin: [0.02, 0.04, 0.02, 0.04], align: 'center' });
  txt(s, 'T054 之 test 結果（3D Dice，左右平均）；粗體＝各列最高（不含 affine、半監督）；* 半監督之標籤項即為評估之 30 結構', FOOT);
}

if (TRY) {
  // ─────────────────────────────────────────────── 附：論文常見之視覺化（試作；make_paper_vis.py）
  // 10-23 使用者：「每一個都蠻值得嘗試，先記起來…這幾個先試看看」→ 先放在最後給使用者看，再決定放哪裡
  const EBT = '附：論文常見之視覺化（試作）';
  {
    const s = base(EBT, '各結構之 Dice 分布（test ' + 51 + ' 位）');
    fitImage(s, CH('1028_try_box.png'), M, 1.3, 12.13, 5.3, '各結構之 Dice 箱形圖');
    txt(s, '畫法同 VoxelMorph Fig. 5（左右合併、依 ANTs SyN 排序）；上排縱軸自 0.68 起，↓ 為範圍外之中位數；* 半監督之標籤項即為評估之 30 結構', FOOT);
  }
  {
    const L = TRY.lambda, sv = L.SVF, dp = L.displacement;
    const s = base(EBT, 'λ 與 Dice、folding 之關係');
    fitImage(s, CH('1028_try_lambda.png'), M, 1.3, 12.13, 4.1, 'λ 與 Dice、folding');
    bullets(s, [
      [{ text: 'SVF（λ = ' + sv.map((r) => r.lambda).join('、') + '）：Dice ' + sv.map((r) => f4(r.dice)).join('、') + '；folding 皆 ≤ '
        + Math.ceil(Math.max(...sv.map((r) => r.fold))) + ' voxels／位', options: { bold: true } }],
      [{ text: 'Displacement field（λ = ' + dp.map((r) => r.lambda).join(' → ') + '）：Dice ' + dp.map((r) => f4(r.dice)).join(' → ')
        + '，folding ' + dp.map((r) => thou(r.fold)).join(' → ') + ' voxels／位' }],
    ], { x: M, y: 5.55, w: 12.13, h: 1.1, fontSize: 14, paraSpaceAfter: 6 });
    txt(s, '畫法同 VoxelMorph Fig. 7（此處為 test Dice）；SVF = mix_exp7／6／5，displacement field = mix_exp3／4', FOOT);
  }
  {
    const E = TRY.errgrid, LN = ['ants', 'mix_exp3', 'mix_exp6', 'tm_mixv2'], AR = [XC, XP, XM];
    const rng = (ks, k, d) => Math.min(...ks.map((x) => E[x][k])).toFixed(d) + '～' + Math.max(...ks.map((x) => E[x][k])).toFixed(d);
    // 10-27 使用者：「P21 是不是還可以加那種形變場表示方向的 RGB 圖」「直接加進第 21 頁」→ 圖改成 10 欄 × 3 列（誤差、網格、位移場 RGB），全寬；條列移到圖下面
    const s = base(EBT, '誤差圖、變形網格與位移場（T054，冠狀面）');
    const g = fitImage(s, CH('1028_try_errgrid.png'), M, 1.25, 12.13, 3.65, '誤差圖、變形網格與位移場 RGB');
    bullets(s, [
      [{ text: 'MAE：架構修改 ' + rng(AR, 'mae', 3) + '，低於 ANTs、VoxelMorph、TransMorph（' + rng(LN, 'mae', 3) + '）', options: { bold: true } }],
      // 10-28 加 TransMorph-diff：MSE 直接以亮度差為目標 → 誤差最小，但 Dice 最低（第 15 頁）
      ...(E.tm_diff_mixv2 ? [[{ text: 'TransMorph-diff：MAE ' + f3(E.tm_diff_mixv2.mae) + ' 最低但 Dice 最低 → 亮度一致不等於結構對齊', options: { color: C.RUST } }]] : []),
      [{ text: '位移場：ANTs 大範圍、平滑（' + E.ants.umag.toFixed(2) + ' mm）；學習式較小、局部變化多；半監督 γ = 5 呈區塊狀' }],
    ], { x: M, y: 1.25 + g.h + 0.15, w: 12.13, h: 6.7 - (1.25 + g.h + 0.15), fontSize: 14, paraSpaceAfter: 6 });
    txt(s, '誤差圖、網格同 DWP（Jian et al.）Fig. 2；位移場 RGB 同 VoxelMorph Fig. 6（各方法同一刻度 ±' + TRY.rgb_mm + ' mm）；* 半監督之標籤項即為評估之 30 結構', FOOT);
  }
  {
    const P = TRY.c2f;
    const s = base(EBT, 'Coarse-to-fine 各層之累積形變（T054）');
    fitImage(s, CH('1028_try_c2f.png'), M, 1.3, 12.13, 4.2, 'Coarse-to-fine 各層');
    bullets(s, [
      [{ text: 'Dice 逐層上升：affine ' + f3(P.affine_dice) + ' → ' + P.level.map((l, i) => l + ' ' + f3(P.dice[i])).join(' → '),
        options: { bold: true } }],
      [{ text: '粗層為大範圍、平滑之位移（1/16 層平均 ' + P.umag[0].toFixed(2) + ' mm），細層補局部細節（原尺寸累積 '
        + P.umag[P.umag.length - 1].toFixed(2) + ' mm）' }],
    ], { x: M, y: 5.6, w: 12.13, h: 1.05, fontSize: 13, paraSpaceAfter: 4 });
    txt(s, '畫法同 Dual-PRNet++ Fig. 6；模型 ' + XP + '；各層之總形變放大到原尺寸後套用，最後一層即模型輸出', FOOT);
  }
}

if (DA && SHOW_BGFOLD) {
  // ─────────────────────────────────────────────── 附錄：影像邊界之 folding（ASD/check_bg_folding.py）
  const B = DA.bgfold;
  const s = base('附錄', 'Coarse-to-fine 之影像邊界 folding：形變合成時 zero padding 所致');
  fitImage(s, CH('1028_bgfold.png'), M, 1.35, 12.13, 3.6, '影像邊界 folding 之成因與驗證');
  bullets(s, [
    [{ text: '位置：腦外 folding 99.99% 以上位於影像最外側 0–4 voxels（主要為下緣）；test ' + DA.n + ' 位平均 coarse-to-fine '
      + thou(B.mix_pyramid.bg_test51) + '、合併模型 ' + thou(B.mix_cascade_pyramid.bg_test51) + ' voxels／位', options: { bold: true } }],
    [{ text: '成因：合成 U(x) = u(x) + U↑(x + u(x)) 時 x + u(x) 超出影像，grid_sample 以 0 補值；coarse-to-fine 之位移延伸至影像邊界（平均 '
      + B.mix_pyramid.disp_border.toFixed(1) + ' mm，VoxelMorph ' + B.mix_exp6.disp_border.toFixed(1) + ' mm）' }],
    [{ text: '僅於推論改為 padding_mode = \'border\'：腦外 folding 消失，腦內 folding 與 Dice 不變 → 不影響配準結果；報告以腦內 folding 為準',
       options: { color: C.TEAL, bold: true } }],
  ], { x: M, y: 5.05, w: 12.13, h: 1.65, fontSize: 13.5, paraSpaceAfter: 5 });
  txt(s, 'ASD/check_bg_folding.py；(b) 為 ' + B.mix_cascade_pyramid.n + ' 位之平均（T054、D031、T079、VNT045、sub-0043）；'
    + '模型未重新訓練', FOOT);
}

if (DA && DA.p.new_vs_old && STG) {
  // ─────────────────────────────────────────────── ⑥ 依原作程式補齊細節之影響（10-21 補、10-28 帶回；手冊 §29、文獻/cascade與pyramid原理筆記.md 第 7 節）
  const NV = DA.p.new_vs_old, O = DA.old, m = DA.m, o = STG.old, n = STG.new;
  // 10-28 使用者：「p12 可以到最後」→ 放在附錄最後一頁
  const s = base('附：架構修改之補充', '依原作程式補齊細節：單一設計略為提升，合併模型則下降');
  const cmp = (r) => (NV[r].win + '/' + NV[r].n + ' 位');
  table(s, [
    ['', '補齊之細節', '先前實作', '依原作程式', '依原作較高者'],
    ['Cascade', 'RCN：每級輸出 ×1/n、串接時 border', f4(m[O.cascade].dice), f4(m[XC].dice), cmp('cascade')],
    ['Coarse-to-fine', 'DWP：合成時 δ detach、子層亦計 loss', f4(m[O.c2f].dice), f4(m[XP].dice), cmp('c2f')],
    [{ text: '兩者合併', options: { bold: true } }, '上述兩者皆加', f4(m[O.merged].dice), { text: f4(m[XM].dice), options: { bold: true, color: C.RUST } },
      { text: cmp('merged'), options: { color: C.RUST } }],
  ], { x: M, y: 1.45, w: 12.13, colW: [1.7, 4.1, 1.9, 1.9, 2.53], fontSize: 13, rowH: 0.42 });
  txt(s, '合併模型拆成兩級分析（test ' + STG.n + ' 位）', { x: M, y: 3.45, w: 7, h: 0.3, fontSize: 13, bold: true, color: C.MUTED });
  table(s, [
    ['', '第 1 級後 Dice', '第 1 級平均位移', '第 2 級後 Dice', '第 2 級平均位移'],
    ['先前實作', f4(o.d1), o.u1.toFixed(2) + ' mm', f4(o.d2), o.u2.toFixed(2) + ' mm'],
    ['依原作程式', { text: f4(n.d1), options: { color: C.RUST, bold: true } }, n.u1.toFixed(2) + ' mm', f4(n.d2),
      { text: n.u2.toFixed(2) + ' mm', options: { color: C.RUST, bold: true } }],
  ], { x: M, y: 3.8, w: 7.4, colW: [1.6, 1.45, 1.45, 1.45, 1.45], fontSize: 12.5, rowH: 0.4 });
  bullets(s, [
    [{ text: '差距產生於第 1 級（' + f4(o.d1) + ' → ' + f4(n.d1) + '，' + STG.stage1_new_lower + '/' + STG.n + ' 位較低）', options: { bold: true } }],
    [{ text: '推測：DWP 之子層 loss 僅加於第 2 級，約束負責細部修正之第 2 級；尚未以消融實驗確認', options: { color: C.MUTED } }],
  ], { x: 8.25, y: 3.75, w: 4.48, h: 2.8, fontSize: 14, paraSpaceAfter: 12 });
  txt(s, 'test ' + DA.n + ' 位；先前實作 = 10-06 依論文文字；依原作程式 = 10-21 對照 RCN、DWP 程式補齊，其餘設定相同；位移為 atlas 腦內平均', FOOT);
}

if (page !== ORDER.length) throw new Error('頁數對不上：做了 ' + page + ' 頁，ORDER 排了 ' + ORDER.length + ' 頁');
ORDER.forEach((k, i) => {
  if (!NOTES[k]) console.warn('[!] 講稿.md 沒有 ' + k + '（第 ' + (i + 1) + ' 頁）');
  else if (NOTES[k].page !== i + 1) console.warn('[!] 講稿.md 的 ' + k + ' 寫第 ' + NOTES[k].page + ' 頁，實際是第 ' + (i + 1) + ' 頁');
});
pres.writeFile({ fileName: OUT }).then(() => console.log('ok -> ' + path.basename(OUT) + ' ｜' + page + ' 頁'));
