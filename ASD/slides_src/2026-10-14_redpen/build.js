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
pres.title = 'VoxelMorph：09-30 會議意見之回覆';
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
const MM = D.residue_mm;            // ③④ 原始數值版（mm／mm³、Dice 進步不扣平均、用 mm 分組），2026-10-05 起第 12、14、15 頁用這個
const done = (e) => m[e].status === 'done';
const score = (e) => (done(e) ? f3(m[e].mean) : '訓練中');
// 速度場權重 1、0.5 只有零星幾個點（平均 0.000002%、0.0001%），印 0.000% 會被看成完全沒有
const jfmt = (x) => (x === 0 ? '0%' : x < 0.001 ? '< 0.001%' : pct(x));
const fold = (e) => (done(e) ? jfmt(m[e].jneg) : '訓練中');
// 2026-10-07 使用者：「正式一點、不要太口語」→ ①～⑤ 也改成學術用語（術語對照見 README）。各段的小標統一寫在這裡
const EB = { fold: '① Folding 之位置（p18）', lam: '② SVF 之 λ（p18）', res: '③ 殘留鄰近結構之 Dice（p23）',
  both: '③④ 三個部位之比較（p22、p23）', back: '④ 枕部殘留（p22）', wide: '⑤ 2× width SVF（p25）' };
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

// ⑥ 架構修改（紅字以外；2026-10-06 使用者：「把改架構的進度也加進 10/14 簡報」）：五頁。
// 2026-10-07 使用者：「正式一點、公式 block 都很重要、不要太口語」→ 學術用語；cascade、coarse-to-fine 各一頁架構圖＋編號公式。
// 長條圖是 make_charts.py 的，架構圖與公式是 make_arch.py 的
const ARCH_FIGS = ['1014_arch_lit.png', '1014_arch_step0.png', '1014_arch_step0_eq.png', '1014_arch_cascade.png',
  '1014_arch_cascade_eq.png', '1014_arch_pyramid.png', '1014_arch_pyramid_eq.png'];
const HAS_ARCH = ARCH_FIGS.every((f) => fs.existsSync(CH(f))) && D.multipass && D.multipass.mix_exp6 && D.multipass_vs_wide;
// 補充評估指標（2026-10-07 使用者：「把 HD95 和 SDlogJ 加進去，然後可以更新這次 meeting 簡報」）：定義一頁＋結果一頁
// 數值：ASD/test_dice.py --surface → models/<exp>/surface_<epoch>.csv → gather.py 的 surface；圖與公式：make_metrics.py
const SF = D.surface || {};
const HAS_METRIC = SF.affine && ['mix_exp2', 'mix_exp5', 'mix_exp6', 'mix_exp7', 'mix_exp4', 'mix_exp3', 'mix_wide', 'mix_wide_vel']
  .every((e) => SF[e]) && ['1014_metric_demo.png', '1014_metric_eq.png'].every((f) => fs.existsSync(CH(f)));
const FOLD_FG = Object.values(m).every((x) => x.status !== 'done' || x.fold_def === 'fg');   // folding 百分比是否已改用「分母：atlas 非背景」（跟論文比較用 voxel 數）

// 頁碼：第 2 頁的表、最後一頁的「下一步」會引用後面的頁，所以先排好順序再算（最後會檢查有沒有對上）
const ORDER = ['cover', 'summary', 'fold_where', ...(HAS_PARAMS ? ['fold_params'] : []), 'fold_regions', 'fold_zoom', 'lam_prev', 'lam',
  ...(HAS_LAM ? ['lam_struct', 'lam_grid'] : []),
  'res_method', ...(HAS_METHOD ? ['res_m1', 'res_m2', 'res_m3', 'res_m4'] : []), 'res_result', ...(HAS_SIX ? ['res_six'] : []), 'res_regions', 'back',
  ...(HAS_BASE6 ? ['base6'] : []), 'res_check',
  'wide', ...(HAS_WCURVE ? ['wide_curve'] : []), ...WIDE_MORE.filter(([, f]) => f.every((x) => fs.existsSync(x))).map(([k]) => k),
  ...(HAS_METRIC ? ['met_def', 'met_res'] : []),
  ...(HAS_ARCH ? ['arch_why', 'arch_step0', 'arch_cascade', 'arch_pyramid', 'arch_plan'] : []),
  'next'];
const PG = Object.fromEntries(ORDER.map((k, i) => [k, i + 1]));

// ───────────────────────────────────────────────────────── 01 封面
{
  const s = pres.addSlide();
  page += 1;
  s.background = { color: '1A2125' };
  txt(s, '2026-10-14   MEETING', { x: M, y: 2.2, w: 11, h: 0.3, fontFace: F.MONO, fontSize: 12, color: '8A9294', charSpacing: 4 });
  txt(s, '09-30 會議意見之回覆', { x: M, y: 2.7, w: 11.5, h: 1.0, fontSize: 40, bold: true, color: C.WHITE });
  txt(s, 'Folding 之位置、SVF 之 λ、殘留鄰近結構之 Dice、枕部殘留、2× width SVF',
    { x: M, y: 3.75, w: 11.5, h: 0.5, fontSize: 18, color: 'D9DEDF' });
  if (HAS_ARCH) {
    txt(s, '另：VoxelMorph 架構修改之進度', { x: M, y: 4.25, w: 11.5, h: 0.4, fontSize: 16, color: 'D9DEDF' });
  }
  txt(s, 'VoxelMorph 腦部 MRI 配準（scan-to-atlas，MNI152）', { x: M, y: 4.8, w: 11, h: 0.4, fontSize: 16, color: '8A9294' });
}

// ───────────────────────────────────────────────────────── 02 一頁看完
{
  const s = base('SUMMARY', '摘要：五項會議意見之處理結果');
  const lam = HAS_LAM
    ? 'λ 2 → 1：' + sgn(P.lam_vel_1.mean, 4) + '；1 → 0.5：' + sgn(P.lam_vel_05.mean, 4) + '（' + pval(P.lam_vel_05.p) + '）\nfolding 均近於 0'
    : 'λ = 2：' + score('mix_exp5') + '，無 folding；λ = 1、0.5 訓練中';
  const wide = done('mix_wide_vel')
    ? 'Dice ' + score('mix_wide_vel') + '，與 2× width displacement field 相當\nfolding 近於 0'
    : '訓練中';
  table(s, [
    ['會議意見（09-30）', '方法', '結果'],
    ['① 確認 folding 之位置（p18）', 'test 51 位之 folding voxel 定位', '呈小型團簇，沿腦溝分布於皮質與白質'],
    ['② 調整 SVF 之 λ（p18）', 'λ = 2、1、0.5 各訓練一個模型', lam],
    ['③ Dice 僅計算殘留鄰近區域（p23）', '僅平均殘留鄰近結構之 Dice', hl('顱頂殘留越厚，皮質 ΔDice 越小（n = ' + DL.n + '）', C.RUST)],
    ['④ 枕部殘留（p22）', '量測枕部殘留厚度', '有殘留，與 ΔDice 無顯著相關'],
    ['⑤ 2× width 改用 SVF（p25）', '2× width U-Net + SVF', wide],
  ], { x: M, y: 1.65, w: 12.13, colW: [3.7, 3.75, 4.68], fontSize: 13.5, rowH: 0.62 });
  // 2026-10-07 使用者：「P2 就和老師說 SVF」→ SVF 第一次出現的地方寫出全名與定義
  txt(s, [
    { text: '註：SVF', options: { bold: true } },
    { text: ' = stationary velocity field（穩態速度場），φ = exp(v)；' },
    { text: 'displacement field', options: { bold: true } },
    { text: '：直接輸出位移 u，φ = Id + u（VoxelMorph 論文 Table I 之版本）' },
  ], { x: M, y: 5.52, w: 12.13, h: 0.35, fontSize: 12.5, color: C.MUTED });
  txt(s, '③ 另確認：顱頂殘留並非 FreeSurfer 分割低估所致，而係去顱骨不完全（第 ' + PG.res_check + ' 頁）',
    { x: M, y: 6.0, w: 12.13, h: 0.45, fontSize: 14.5, color: C.MUTED });
  if (HAS_ARCH) {
    const G = D.multipass.mix_exp6.gain2;
    txt(s, [
      { text: '⑥（紅字以外）架構修改：', options: { bold: true, color: C.TEAL } },
      { text: 'test-time recursion ΔDSC = ' + sgn(G.mean, 4) + '（' + G.win + '/' + G.n + '）；cascade、coarse-to-fine 已實作，待訓練（第 '
          + PG.arch_why + '～' + PG.arch_plan + ' 頁）' },
    ], { x: M, y: 6.42, w: 12.13, h: 0.45, fontSize: 13.5, color: C.MUTED });
  }
  if (HAS_METRIC) {
    // 2026-10-07 加：補充評估指標兩頁（HD95、SDlogJ、folding；跟論文比較用 voxel 數）
    txt(s, [
      { text: '補充評估指標：', options: { bold: true, color: C.TEAL } },
      { text: 'HD95、SDlogJ；folding 百分比之分母為 atlas 非背景 voxel，與論文比較改用 folding voxel 數（第 '
          + PG.met_def + '～' + PG.met_res + ' 頁）' },
    ], { x: M, y: 6.8, w: 12.13, h: 0.4, fontSize: 13.5, color: C.MUTED });
  }
}

// ───────────────────────────────────────────────────────── 03 ① 擠爆在哪
const F3 = FD.mix_exp3;
{
  const s = base(EB.fold, 'Folding 之分布：散布於皮質與白質，個體間位置不一致');
  fitImage(s, FC('folding_views.png'), M, 1.42, 7.75, 5.5, '軸狀面、冠狀面、矢狀面各 4 個切面');
  const X0 = 8.6, WW = 4.13;
  txt(s, 'Folding：Jacobian 行列式 |J| ≤ 0 之 voxel（形變局部翻轉）。圖為 displacement field、λ = 1（mix_exp3），'
        + 'test 51 位疊合於 atlas；僅顯示 ≥ 3 位重疊之 voxel，顏色越紅表示人數越多。',
    { x: X0, y: 1.5, w: WW, h: 1.05, fontSize: 12.5, color: C.MUTED });
  [[F3.any1.toFixed(0) + '%', '腦內 voxel 中，至少 1 位出現 folding', C.TEAL],
   [F3.any5.toFixed(1) + '%', '≥ 5 位於同一 voxel 出現 folding\n→ 個體間位置不一致', C.RUST],
   [(F3.points_med / FD.mix_exp4.points_med).toFixed(1) + ' 倍', 'folding voxel 數：λ = 1 相對 λ = 2（中位數）', C.TEAL]].forEach((it, i) => {
    const y = 2.65 + i * 1.42;
    card(s, X0, y, WW, 1.3);
    stat(s, it[0], it[1], X0, y + 0.06, WW, it[2]);
  });
}

if (HAS_PARAMS) {
  // ─────────────────────────────────────────────────────── ① 不同設定的擠爆位置
  const s = base(EB.fold, 'Folding 與設定：displacement field 隨 λ 降低而增加，SVF 近於零');
  fitImage(s, FC('folding_params.png'), M, 1.42, 12.13, 5.0, '不同設定 × 三個切面方向之 folding 位置');
  txt(s, '每欄為一個模型、每列為一個切面方向；同樣僅顯示 ≥ 3 位重疊之 voxel。'
        + 'SVF 三欄無標示：folding voxel 極少，且個體間位置不重疊。',
    { x: M, y: 6.5, w: 12.13, h: 0.45, fontSize: 13, color: C.MUTED, align: 'center' });
}

// ───────────────────────────────────────────────────────── 04 ① 哪些區域、多深
{
  const s = base(EB.fold, 'Folding 之解剖分布：主要位於大腦皮質與白質，深部結構極少');
  fitImage(s, CH('1014_folding_regions.png'), M, 1.45, 12.13, 3.6, 'Folding voxel 之區域分布');
  const w = (12.13 - 0.4 * 2) / 3;
  [[F3.ctx_wm.toFixed(0) + '%', 'folding voxel 位於大腦皮質與白質', C.RUST],
   [F3.depth_med.toFixed(0) + ' mm', '距腦表面深度（中位數）\n位於腦溝深度範圍內', C.TEAL],
   [F3.share['深部灰質・海馬・杏仁核'].toFixed(1) + '%', '深部灰質、海馬、杏仁核', C.TEAL]].forEach((it, i) => {
    const x = M + i * (w + 0.4);
    card(s, x, 5.2, w, 1.6);
    stat(s, it[0], it[1], x, 5.32, w, it[2]);
  });
}

// ───────────────────────────────────────────────────────── 05 ① 放大一團
{
  const s = base(EB.fold, 'Folding 局部放大：沿腦溝呈線狀分布，形變網格局部翻轉');
  fitImage(s, FC('folding_zoom_T054.png'), M, 1.45, 12.13, 4.55, 'T054 最大之 folding 團簇');
  bullets(s, [
    [{ text: '黃線：規則網格經形變後之樣貌；紅點：folding voxel', options: { color: C.MUTED } }],
    [{ text: '紅點沿腦溝呈線狀分布，網格線於該處交叉', options: { bold: true } },
     { text: '　→ 為對齊 atlas 之腦溝，局部形變過大而翻轉' }],
  ], { x: M, y: 6.1, w: 12.13, h: 0.9, fontSize: 14, paraSpaceAfter: 6 });
}

// ───────────────────────────────────────────────────────── 06 ② 上次的結論
{
  // 圖是 09-20 那份 ablation.png 的正式用語版（make_charts.py 的 1014_ablation.png；09-20 的產生器不動）
  const s = base(EB.lam, '前次結論：同條件下 SVF 之 Dice 較高，且無 folding');
  fitImage(s, CH('1014_ablation.png'), M, 1.45, 12.13, 4.4, '一次僅改變一項因素之消融比較');
  bullets(s, [
    [{ text: '全解析度、λ = 2：SVF ' + score('mix_exp5') + ' ＞ displacement field ' + score('mix_exp4'), options: { bold: true, color: C.TEAL } },
     { text: '　folding ' + fold('mix_exp5') + ' vs ' + fold('mix_exp4') }],
    [{ text: 'Displacement field 之最佳結果 ' + score('mix_exp3') + ' 為 λ = 1。', options: { bold: true } },
     { text: '會議提問：SVF 降低 λ 之效果為何？' }],
  ], { x: M, y: 6.0, w: 12.13, h: 1.0, fontSize: 14.5, paraSpaceAfter: 6 });
}

// ───────────────────────────────────────────────────────── 07 ② λ 掃描
{
  const s = base(EB.lam, HAS_LAM ? 'SVF 之 λ：2 → 1 Dice 上升，降至 0.5 無進一步改善' : 'SVF 之 λ：2 → 1 → 0.5');
  fitImage(s, CH('1014_lambda.png'), M, 1.45, 12.13, 4.4, 'λ 與 Dice、folding ratio');
  let items;
  if (HAS_LAM) {
    const L1 = P.lam_vel_1, L05 = P.lam_vel_05, VW = P.version_w1;
    items = [
      [{ text: 'λ 2 → 1：' + sgn(L1.mean, 4) + '（' + L1.win + '/' + L1.n + ' 位上升，' + pval(L1.p) + '）；1 → 0.5：'
           + sgn(L05.mean, 4) + '（' + pval(L05.p) + '）', options: { bold: true } }],
      [{ text: 'SVF 最佳為 λ = 1（' + score('mix_exp6') + '），較 displacement field λ = 1（' + score('mix_exp3') + '）低 ' + f4(VW.mean),
         options: { bold: true } },
       { text: '　folding 則由 ' + fold('mix_exp3') + ' 降至 ' + fold('mix_exp6'), options: { bold: true, color: C.TEAL } }],
    ];
  } else {
    items = [[{ text: 'λ = 2（mix_exp5）：' + score('mix_exp5') + '，folding ' + fold('mix_exp5'), options: { bold: true } }]];
    ['mix_exp6', 'mix_exp7'].forEach((e) => {
      items.push(done(e)
        ? [{ text: 'λ = ' + m[e].weight + '（' + e + '）：' + score(e) + '，folding ' + fold(e), options: { bold: true } }]
        : [{ text: 'λ = ' + m[e].weight + '（' + e + '）：訓練中', options: { bold: true, color: C.RUST } }]);
    });
  }
  bullets(s, items, { x: M, y: 6.0, w: 12.13, h: 0.95, fontSize: 14.5, paraSpaceAfter: 6 });
}

if (HAS_LAM) {
  // ─────────────────────────────────────────────────────── ② 權重 0.5 為什麼沒再變好
  const S = D.struct, df = (n) => S.mix_exp7[n] - S.mix_exp5[n];
  const s = base(EB.lam, 'λ = 0.5 未再提升之原因：大型結構上升、小型結構下降');
  fitImage(s, CH('1014_lambda_struct.png'), M, 1.45, 7.7, 5.45, '各結構相對 λ = 2 之 Dice 變化');
  const X0 = 8.55, WW = 4.18;
  bullets(s, [
    [{ text: '大型結構持續上升', options: { bold: true, color: C.TEAL } },
     { text: '\n大腦皮質 ' + sgn(df('大腦皮質'), 3) + '、白質 ' + sgn(df('大腦白質'), 3), options: { color: C.MUTED, fontSize: 13.5 } }],
    [{ text: '小型結構下降', options: { bold: true, color: C.RUST } },
     { text: '\n脈絡叢 ' + sgn(df('脈絡叢'), 3) + '、腦脊髓液 ' + sgn(df('腦脊髓液'), 3), options: { color: C.MUTED, fontSize: 13.5 } }],
    [{ text: 'Dice 為 30 個結構之等權平均', options: { bold: true } },
     { text: '\n大、小型結構之變化相互抵銷，平均持平', options: { color: C.MUTED, fontSize: 13.5 } }],
    [{ text: 'λ = 0.5 之皮質 Dice ' + f3(S.mix_exp7['大腦皮質']), options: { bold: true } },
     { text: '\n高於 displacement field λ = 1 之 ' + f3(S.mix_exp3['大腦皮質']), options: { color: C.MUTED, fontSize: 13.5 } }],
  ], { x: X0, y: 1.65, w: WW, h: 5.2, fontSize: 15, paraSpaceAfter: 14 });
}

if (HAS_LAM) {
  // ─────────────────────────────────────────────────────── ② 形變網格：四顆對照
  const s = base(EB.lam, 'λ 越小，形變場之局部變化越細緻');
  fitImage(s, CH('grid_lambda.png'), M, 1.5, 12.13, 4.0, '四個模型之形變網格');
  txt(s, '同一受試者（T054）、同一軸狀切面；黃線為規則網格經形變後之樣貌。',
    { x: M, y: 5.65, w: 12.13, h: 0.4, fontSize: 14.5, align: 'center' });
  txt(s, 'SVF 三個模型幾近無 folding；最右之 displacement field 同樣呈細緻形變，但 folding ' + fold('mix_exp3') + '。',
    { x: M, y: 6.1, w: 12.13, h: 0.4, fontSize: 13.5, align: 'center', color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 08 ③ 老師的做法
{
  const s = base(EB.res, '評估方式：僅平均殘留鄰近結構之 Dice，不平均全部 30 個結構');
  txt(s, '對每個殘留 voxel 找出距離最近之結構；占殘留 voxel ≥ 5% 且屬於 30 個評估結構者，納入平均。',
    { x: M, y: 1.55, w: 12.13, h: 0.45, fontSize: 15 });
  // 2026-10-07 使用者：結構要附英文名稱 → 一個結構一列：中文、FreeSurferColorLUT 名稱（標籤編號）、左／右占比、是否納入
  // （gather.py 的 residue_near；視交叉「不屬於 30 個評估結構」、枕部腦幹「占比 < 5%」也在表裡交代）
  const RN = D.residue_near;
  const body = [];
  [['top', '顱頂'], ['base', '顱底'], ['back', '枕部']].forEach(([k, nm]) => {
    RN[k].forEach((g, i) => {
      const share = g.shares.map((x) => x.toFixed(1) + '%').join('／');
      const inc = g.included ? { text: '納入', options: { bold: true, color: C.TEAL } }
                             : { text: '不納入（' + g.note + '）', options: { color: C.MUTED } };
      body.push([...(i === 0 ? [{ text: nm, options: { rowspan: RN[k].length, bold: true } }] : []),
                 g.zh, { text: g.fs, options: { color: C.MUTED } }, share, inc]);
    });
  });
  table(s, [['殘留部位', '鄰近結構', 'FreeSurfer 名稱（標籤編號）', '占殘留 voxel（左／右）', 'Dice 平均']].concat(body),
    { x: M, y: 2.15, w: 12.13, colW: [1.1, 1.35, 4.2, 2.4, 3.08], fontSize: 13, rowH: 0.34 });
  card(s, M, 5.75, 12.13, 0.8, 'FFF3E8');
  txt(s, [
    { text: '理由：', options: { bold: true } },
    { text: '殘留位於腦組織外側，鄰近結構幾乎皆為大腦皮質；若將視丘、海馬迴等遠離殘留之結構一併平均，其影響將被稀釋。' },
  ], { x: M + 0.3, y: 5.94, w: 11.5, h: 0.45, fontSize: 15, fontFace: F.SANS, lang: 'zh-TW', color: C.INK, margin: 0 });
}

if (HAS_METHOD) {
  // ─────────────────────────────────────────────────────── ③ 頭頂殘留怎麼量（四頁）
  // 2026-10-06 使用者：「放進簡報取代第 12 頁、一定要放公式、希望可以和 paper 一樣好閱讀」。
  // 每一步一頁：上面是圖、下面是編號公式＋「其中」符號說明（公式用 LaTeX 字型畫成圖，都是 make_method.py 產生的）。
  // 圖和公式區塊就是投影片上的大小（寬 12.13 吋），第 4 頁公式多一行（25 mm 的說明），所以圖矮一點
  // 第 2、4 頁的公式說明各多一行（換門檻、換 25 mm 結論都一樣），圖相對矮一點
  // 標題不放符號（投影片標題打不出下標，z_top 會變成底線）；符號在公式區塊裡定義
  const STEP = [['腦組織上緣', 3.6, 1.85], ['強度門檻', 3.55, 2.02], ['各 column 之殘留厚度', 3.6, 1.85],
    ['顱頂區域之平均', 2.95, 2.66]];
  STEP.forEach(([name, hf, he], i) => {
    const s = base(EB.res, '顱頂殘留厚度之量測（' + (i + 1) + '/4）：' + name);
    fitImage(s, CH('1014_method_' + (i + 1) + '.png'), M, 1.38, 12.13, hf, '第 ' + (i + 1) + ' 步之圖');
    fitImage(s, CH('1014_method_eq' + (i + 1) + '.png'), M, 1.38 + hf + 0.04, 12.13, he, '第 ' + (i + 1) + ' 步之公式');
  });
}

// ───────────────────────────────────────────────────────── 09 ③ 結果
{
  const s = base(EB.res, '顱頂殘留越厚，皮質 ΔDice 越小；30 結構平均時無此關係');
  fitImage(s, CH('1014_dilution.png'), M, 1.4, 12.13, 3.55, '顱頂殘留厚度 vs ΔDice：30 結構平均 vs 殘留鄰近結構');
  // 2026-10-05 使用者要標 Dice 數值：殘留最多／最少 1/4 的「起點 → 配準後」。起點一定要一起寫（只寫配準後會被起點騙）。
  // 分組直接用 mm 切（170 人的 1/4、3/4 分位數，gather.py 的 residue_mm）
  // ⚠️「起點就高」「進步差不多」「起點差不多」這幾個字是照現在的數字寫的
  const T = MM.top;
  const ba = (b, a) => f3(b) + ' → ' + f3(a) + '（' + sgn(a - b, 3) + '）';
  table(s, [
    ['顱頂殘留厚度', '30 結構平均：Dice（affine → 配準後）', '殘留鄰近結構（大腦皮質）：Dice（affine → 配準後）'],
    ['≥ ' + T.hi.toFixed(2) + ' mm（' + T.dirty_n + ' 位）', ba(T.all_dirty_b, T.all_dirty_a), ba(T.dirty_b, T.dirty_a)],
    ['≤ ' + T.lo.toFixed(2) + ' mm（' + T.clean_n + ' 位）', ba(T.all_clean_b, T.all_clean_a), ba(T.clean_b, T.clean_a)],
    [{ text: '解讀', options: { bold: true } },
     'affine 已高 ' + f3(T.all_dirty_b - T.all_clean_b) + '，ΔDice 相近 → 無差異',
     hl('affine 相近；配準後低 ' + f3(T.clean_a - T.dirty_a) + '，ΔDice 少 ' + f3((T.clean_a - T.clean_b) - (T.dirty_a - T.dirty_b)), C.RUST)],
  ], { x: M, y: 5.12, w: 12.13, colW: [2.75, 4.65, 4.73], fontSize: 13 });
  txt(s, '每點代表一位受試者，共 ' + T.n + ' 位（test 50、val 50、MRS 70，均未參與訓練；三組分別計算，方向一致）',
    { x: M, y: 6.72, w: 11.0, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

if (HAS_SIX) {
  // ─────────────────────────────────────────────────────── ③ 同樣 6 位：兩種算法
  const s = base(EB.res, '代表性個案（6 位）：30 結構平均無法區分，皮質 Dice 可以區分');
  // 公式 2026-10-06 一度放這頁右邊，使用者說分開 → 獨立一頁（res_formula），這頁恢復整頁寬的圖
  fitImage(s, CH('1014_six.png'), M, 1.45, 12.13, 4.95, '顱頂殘留最多 3 位與最少 3 位之皮質 Dice');
  txt(s, 'test 中顱頂殘留最多與最少各 3 位（與先前殘留對照圖相同），紅色為殘留。大字為皮質 Dice（affine → 配準後），其下為 ΔDice。',
    { x: M, y: 6.5, w: 12.13, h: 0.45, fontSize: 13, color: C.MUTED, align: 'center' });
}

// ───────────────────────────────────────────────────────── 10 ③④ 三個位置
{
  const s = base(EB.both, '三個部位之比較：僅顱頂殘留與 ΔDice 相關，顱底與枕部無相關');
  fitImage(s, CH('1014_regions.png'), M, 1.38, 12.13, 3.15, '三個部位：殘留量 vs 殘留鄰近結構之 ΔDice');
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
                         two('≥ ' + amt(k, MM[k].hi), ba2(MM[k].dirty_b, MM[k].dirty_a)),
                         two('≤ ' + amt(k, MM[k].lo), ba2(MM[k].clean_b, MM[k].clean_a))];
  table(s, [
    ['部位', '納入 Dice 之 FreeSurfer 結構（標籤編號，左右合併）', '殘留最多 1/4：affine → 配準後', '殘留最少 1/4：affine → 配準後'],
    row('top', '顱頂'), row('base', '顱底'), row('back', '枕部'),
  ], { x: M, y: 4.68, w: 12.13, colW: [0.95, 5.3, 2.94, 2.94], fontSize: 12 });
}

// ───────────────────────────────────────────────────────── 11 ④ 後腦杓
{
  const s = base(EB.back, '枕部殘留：與 ΔDice 無顯著相關');
  // 2026-10-05 使用者：第 13、15 頁在講同一件事，圖要統一 → 改成跟第 13 頁一樣的 3 對 3（原本是 1 對、五個切面、不標 Dice）
  fitImage(s, CH('1014_six_back.png'), M, 1.42, 12.13, 4.55, '枕部殘留最多 3 位與最少 3 位之皮質 Dice');
  const B = D.back, K = MM.back;
  const bl = [...new Set(R.back.labels_pooled.split('、').map((x) => x.replace(/^[左右]/, '')))].join('、');
  txt(s, 'test 中枕部殘留最多與最少各 3 位，紅色為殘留；僅計算鄰近之' + bl + '。兩組之 ΔDice 互有高低，無法區分',
    { x: M, y: 6.02, w: 12.13, h: 0.35, fontSize: 12.5, color: C.MUTED, align: 'center' });
  txt(s, 'n = ' + K.n + '：枕部殘留厚度與 ΔDice 之相關 r = ' + sgn(K.r) + '（' + pval(K.p) + '），無顯著相關',
    { x: M, y: 6.4, w: 12.13, h: 0.35, fontSize: 14.5, bold: true, color: C.TEAL, align: 'center' });
  txt(s, (HAS_METHOD ? '量測方式同顱頂（第 ' + PG.res_m1 + '～' + PG.res_m4 + ' 頁），column 改為前後方向。' : '')
    + '大腦縱裂與小腦天幕處原有硬腦膜（test ' + B.n + ' 位中位數 ' + B.median.toFixed(2) + ' mm）',
    { x: M, y: 6.75, w: 11.0, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

if (HAS_BASE6) {
  // ─────────────────────────────────────────────────────── ③ 顱底：跟第 13 頁（頭頂）、後腦杓那頁同一個樣子
  // 2026-10-05 使用者：「這個顱底也來一張」。⚠️「這 6 位裡殘留多的進步還略多」是照現在的數字寫的
  const s = base(EB.res, '顱底殘留：與 ΔDice 無顯著相關');
  fitImage(s, CH('1014_six_base.png'), M, 1.42, 12.13, 4.55, '顱底殘留最多 3 位與最少 3 位之鄰近結構 Dice');
  const K = MM.base;
  const bl = [...new Set(R.base.labels_pooled.split('、').map((x) => x.replace(/^[左右]/, '')))].join('、');
  txt(s, 'test 中顱底殘留最多與最少各 3 位，紅色為距腦組織 10 mm 以外之殘留；僅計算鄰近之' + bl + '。此 6 位中，殘留多者之 ΔDice 略高',
    { x: M, y: 6.02, w: 12.13, h: 0.35, fontSize: 12, color: C.MUTED, align: 'center' });
  txt(s, 'n = ' + K.n + '：顱底殘留體積與 ΔDice 之相關 r = ' + sgn(K.r) + '（' + pval(K.p) + '），無顯著相關',
    { x: M, y: 6.4, w: 12.13, h: 0.35, fontSize: 14.5, bold: true, color: C.TEAL, align: 'center' });
  txt(s, '殘留多位於腦之前下方；切面取通過最大殘留團塊中心之矢狀面，故各受試者之切面位置不同',
    { x: M, y: 6.75, w: 11.0, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

// ───────────────────────────────────────────────────────── 12 ③ 是不是 FreeSurfer 畫錯
{
  const s = base(EB.res, '排除 FreeSurfer 分割低估：殘留位於皮質標籤外側，屬去顱骨不完全');
  txt(s, '殘留之定義為「FreeSurfer 腦標籤外、強度 ≥ τ 之 voxel」。若實為 FreeSurfer 低估皮質（漏標），'
        + 'Dice 下降即非殘留所致。結果：皮質標籤（綠）完整，殘留（紅）皆位於其外側。',
    { x: M, y: 1.45, w: 12.13, h: 0.7, fontSize: 14, color: C.MUTED });
  fitImage(s, CH('1014_top_example.png'), M, 2.2, 12.13, 3.05, '殘留多與殘留少之受試者，顱頂放大');
  const w = (12.13 - 0.4 * 2) / 3;
  [[TC.ctx_thick.dirty.toFixed(1) + ' vs ' + TC.ctx_thick.clean.toFixed(1), '白質至腦組織上緣之 voxel 數：殘留多 vs 少\n→ 皮質厚度相同，無缺漏', C.TEAL],
   [(100 * TC.top_ctx_min).toFixed(1) + '%', '腦組織最上層 voxel 屬於皮質之比例\n（' + TC.n + ' 位之最小值）', C.TEAL],
   [TC.cont_int.dirty.toFixed(1) + ' 倍', '殘留之強度為皮質之 ' + TC.cont_int.dirty.toFixed(1) + ' 倍\n→ 低於皮質，符合腦膜等組織', C.RUST]].forEach((it, i) => {
    const x = M + i * (w + 0.4);
    card(s, x, 5.4, w, 1.5);
    stat(s, it[0], it[1], x, 5.5, w, it[2]);
  });
}

// ───────────────────────────────────────────────────────── 13 ⑤ 加寬＋速度場
{
  const WV = done('mix_wide_vel');
  const s = base(EB.wide, WV ? '2× width SVF：Dice 與 displacement field 相當，且幾近無 folding' : '2× width SVF：訓練中');
  const cell = (e) => (done(e) ? { text: e + '\nDice ' + score(e) + '　folding ' + fold(e) }
                               : { text: e + '\n訓練中', options: { color: C.RUST, bold: true } });
  table(s, [
    ['全解析度、λ = 1', 'Default width', '2× width'],
    [{ text: 'Displacement field', options: { bold: true } }, cell('mix_exp3'), cell('mix_wide')],
    [{ text: 'SVF', options: { bold: true } }, cell('mix_exp6'), cell('mix_wide_vel')],
  ], { x: M, y: 1.65, w: 8.2, colW: [2.2, 3.0, 3.0], fontSize: 14, rowH: 0.85 });
  const X0 = 9.2, WW = 3.53;
  const win = (q) => q.win + '/' + q.n + ' 位';
  const thou = (x) => Math.round(x).toLocaleString('en-US');
  const sub = (t) => ({ text: t, options: { color: C.MUTED, fontSize: 13 } });
  // 擠爆點數跟第 4 頁的圖用同一個來源（check_folding.py）；沒算過的才退回 test CSV 換算（兩邊算法差 0.1% 左右）
  const FP = (e) => FD[e] || { points_mean: m[e].points, points_max: m[e].points_max, n_any: m[e].n_any };
  if (WV) {
    // 右欄只有 3.5 吋寬：英文術語較長，標題用 14.5 pt、說明寫短（2026-10-07 改正式用語時重排）
    bullets(s, [
      [{ text: '寬度比較（同一參數化）', options: { bold: true } },
       sub('\nDisplacement ' + sgn(P.width_disp.mean, 4) + '（' + win(P.width_disp) + '上升）'
           + '\nSVF ' + sgn(P.width_vel.mean, 4) + '（' + win(P.width_vel) + '上升）')],
      [{ text: '參數化比較（SVF − displacement）', options: { bold: true } },
       sub('\nDefault width ' + sgn(P.version_w1_vel.mean, 4) + '（' + pval(P.version_w1_vel.p) + '）'
           + '\n2× width ' + sgn(P.version_wide.mean, 4) + '（' + pval(P.version_wide.p) + '）→ 相當')],
      [{ text: 'Folding voxels（2× width）', options: { bold: true } },
       sub('\nDisplacement：平均 ' + thou(FP('mix_wide').points_mean) + '／位'
           + '\nSVF：' + FP('mix_wide_vel').n_any + ' 位出現，最多 ' + thou(FP('mix_wide_vel').points_max) + ' voxels')],
    ], { x: X0, y: 1.6, w: WW, h: 3.5, fontSize: 14.5, paraSpaceAfter: 8 });
    txt(s, '→ Dice 與目前最高之 mix_wide 相當，且幾近無 folding：目前最佳模型',
      { x: M, y: 4.45, w: 8.2, h: 0.45, fontSize: 16, bold: true, color: C.TEAL });
    // 原本下方有「附記：2× width 模型訓練時間超出預估之原因」（顯存與每步時間）；
    // 2026-10-07 使用者在 PowerPoint 裡刪掉了，這裡同步拿掉，重建才不會又出現（資料仍在 gather.py 的 train_time）
  } else {
    bullets(s, [
      [{ text: '寬度比較（同一參數化）', options: { bold: true } },
       sub('\nDisplacement field ' + sgn(P.width_disp.mean, 4) + '（' + win(P.width_disp) + '上升）')],
      [{ text: '參數化比較', options: { bold: true } },
       sub('\n2× width 下，SVF 是否仍優於 displacement field 且無 folding')],
    ], { x: X0, y: 1.7, w: WW, h: 2.6, fontSize: 15, paraSpaceAfter: 12 });
  }
}

if (HAS_WCURVE) {
  // ─────────────────────────────────────────────────────── ⑤ 訓練過程：版本 × 寬度四顆
  const s = base(EB.wide, '訓練曲線：2× width 兩模型之 Dice 相當；SVF 全程無 folding');
  fitImage(s, CH('curve_wide.png'), M, 1.45, 12.13, 4.7, '四個模型之 validation Dice 與 folding ratio');
  txt(s, '上：validation set（51 位）之 Dice，星號為選定之 epoch。下：每位平均 folding voxel 數，虛線為論文 VoxelMorph (CC) 之 19,077。',
    { x: M, y: 6.2, w: 12.13, h: 0.35, fontSize: 13, color: C.MUTED, align: 'center' });
  // 2026-10-06：不用再訓練更久（gather.py 的 plateau：第 100 輪之後的範圍、上下晃的大小、每 100 輪的趨勢）
  const PV = D.plateau.mix_wide_vel, PW = D.plateau.mix_wide;
  txt(s, '無需延長訓練：第 100 epoch 後 Dice 介於 ' + f3(Math.min(PV.lo, PW.lo)) + '～' + f3(Math.max(PV.hi, PW.hi))
    + '（SD ≈ ' + f3(PV.sd) + '）；每 100 epoch 之趨勢 ' + sgn(PV.slope100, 4) + '、' + sgn(PW.slope100, 4) + '，小於波動',
    { x: M, y: 6.6, w: 12.13, h: 0.35, fontSize: 13.5, bold: true, color: C.TEAL, align: 'center' });
}

if (HAS_WM.wide_struct) {
  // ─────────────────────────────────────────────────────── ⑤ 每個結構（make_charts.py 的 1014_wide_struct.png）
  // ⚠️「蒼白球、殼核」「方向相反」是照現在的數字寫的
  const S = D.struct, nm = Object.keys(S.mix_wide_vel);
  const dv = (n) => S.mix_wide_vel[n] - S.mix_exp6[n], dp = (n) => S.mix_wide[n] - S.mix_exp3[n];
  const worse = nm.filter((n) => dv(n) < 0).sort((a, b) => dv(a) - dv(b));
  const ver = nm.map((n) => Math.abs(S.mix_wide_vel[n] - S.mix_wide[n]));
  const s = base(EB.wide, '各結構之 Dice：加寬改善多數結構；2× width 下兩種參數化相近');
  fitImage(s, CH('1014_wide_struct.png'), M, 1.4, 12.13, 4.9, '加寬之效果、2× width 下兩種參數化之差異（各結構 Dice）');
  bullets(s, [
    [{ text: '左：SVF 加寬，' + nm.length + ' 個結構中 ' + (nm.length - worse.length) + ' 個上升；', options: { bold: true } },
     { text: '下降：' + worse.map((n) => n + ' ' + sgn(dv(n), 3)).join('、') + '（displacement field 之蒼白球為 ' + sgn(dp('蒼白球'), 3)
            + '，方向相反）', options: { color: C.MUTED } }],
    [{ text: '右：同為 2× width，SVF 與 displacement field 各結構之差異均在 ±' + f3(Math.max(...ver)) + ' 以內', options: { bold: true } },
     { text: '　→ 整體相當並非平均抵銷所致，各結構皆相近', options: { color: C.MUTED } }],
  ], { x: M, y: 6.35, w: 12.13, h: 0.75, fontSize: 13, paraSpaceAfter: 3 });
}

if (HAS_WM.wide_diff) {
  // ─────────────────────────────────────────────────────── ⑤ 越難對的人幫越多（1014_wide_difficulty.png）
  const WV = D.wide_diff.vel, WP = D.wide_diff.disp;
  const s = base(EB.wide, 'Affine Dice 越低之受試者，加寬之改善越大（兩種參數化皆同）');
  fitImage(s, CH('1014_wide_difficulty.png'), M, 1.38, 12.13, 4.25, 'Affine Dice 與加寬之 ΔDice');
  table(s, [
    ['加寬之 ΔDice', 'Affine 最低 10 位', '中間 31 位', 'Affine 最高 10 位', 'r'],
    ['Displacement field（mix_wide − mix_exp3）', sgn(WP.hard10, 4), sgn(WP.mid, 4), sgn(WP.easy10, 4), sgn(WP.r, 2)],
    ['SVF（mix_wide_vel − mix_exp6）', sgn(WV.hard10, 4), sgn(WV.mid, 4), sgn(WV.easy10, 4), sgn(WV.r, 2)],
  ], { x: M, y: 5.72, w: 12.13, colW: [4.0, 2.1, 2.0, 2.1, 1.93], fontSize: 12.5 });
  txt(s, '每點代表一位受試者（test 51 位）。依 affine Dice 分組；若以模型之 Dice 分組，會產生均值迴歸（regression to the mean）之假象',
    { x: M, y: 6.82, w: 11.0, h: 0.3, fontSize: 11, color: C.MUTED });
}

if (HAS_WM.wide_loss) {
  // ─────────────────────────────────────────────────────── ⑤ 訓練 loss（1014_wide_loss.png）
  const L = D.loss_final;
  const s = base(EB.wide, '訓練損失：加寬降低相似度項，兩種參數化之曲線幾近重疊');
  fitImage(s, CH('1014_wide_loss.png'), M, 1.4, 12.13, 4.4, '四個模型之 training loss：相似度項、平滑項');
  table(s, [
    ['最後一個 epoch（100 iterations 平均）', 'Displacement・default', 'Displacement・2×', 'SVF・default', 'SVF・2×'],
    ['相似度項（−NCC，越低越相似）', L.mix_exp3.image.toFixed(3), L.mix_wide.image.toFixed(3), L.mix_exp6.image.toFixed(3), L.mix_wide_vel.image.toFixed(3)],
    ['平滑項', L.mix_exp3.smooth.toFixed(4), L.mix_wide.smooth.toFixed(4), L.mix_exp6.smooth.toFixed(4), L.mix_wide_vel.smooth.toFixed(4)],
  ], { x: M, y: 5.85, w: 12.13, colW: [3.33, 2.2, 2.2, 2.2, 2.2], fontSize: 12.5 });
  txt(s, '⚠️ 平滑項不可跨參數化比較：SVF 懲罰速度場（積分前）之梯度，displacement field 懲罰位移場本身之梯度。同一參數化下，加寬前後幾近相同',
    { x: M, y: 6.95, w: 12.13, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

if (HAS_WM.wide_full) {
  // ─────────────────────────────────────────────────────── ⑤ 視覺化（大圖）：整片腦（check_folding.py --zoom-pair 一起畫的）
  // 2026-10-06 使用者看了放大圖：「可以來大圖的嗎」→ 整片腦、藍框＝下一頁放大的那一塊
  const s = base(EB.wide, '形變網格（T054）：displacement field 有 folding，SVF 無');
  fitImage(s, FC('folding_full_pair_T054.png'), M, 1.38, 8.3, 5.7, 'T054 全腦切面：2× width displacement field vs 2× width SVF');
  bullets(s, [
    [{ text: 'T054（test），三個方向各一切面', options: { bold: true } },
     { text: '\n通過 2× width displacement field 最大之 folding 團簇（左側大腦白質）', options: { color: C.MUTED } }],
    [{ text: '上：2× width displacement field（mix_wide）', options: { bold: true, color: C.RUST } },
     { text: '\n紅點為 folding voxel，沿腦溝分布於皮質與白質，不限於藍框處', options: { color: C.MUTED } }],
    [{ text: '下：2× width SVF（mix_wide_vel）', options: { bold: true, color: C.TEAL } },
     { text: '\n相同切面皆無 folding，全腦 0 個', options: { color: C.MUTED } }],
    [{ text: '藍框：下一頁之放大範圍；網格間距 4 mm（下一頁為 2 mm）', options: { color: C.MUTED, fontSize: 12 } }],
  ], { x: M + 8.5, y: 1.6, w: 3.63, h: 5.2, fontSize: 14, paraSpaceAfter: 12 });
}

if (HAS_WM.wide_vis) {
  // ─────────────────────────────────────────────────────── ⑤ 視覺化：同一個位置放大（check_folding.py --zoom-pair）
  const s = base(EB.wide, '局部放大（T054）：displacement field 網格翻轉，SVF 無翻轉');
  fitImage(s, FC('folding_zoom_pair_T054.png'), M, 1.38, 7.6, 5.65, 'T054 同一位置：2× width displacement field vs 2× width SVF');
  bullets(s, [
    [{ text: 'T054：左側大腦白質之最大 folding 團簇', options: { bold: true } },
     { text: '\n上：2× width displacement field（mix_wide）\n下：2× width SVF（mix_wide_vel）\n同一位置、同一切面', options: { color: C.MUTED } }],
    [{ text: '黃線：形變網格；紅點：folding voxel', options: { color: C.MUTED } }],
    [{ text: 'Displacement field：網格交叉、翻轉', options: { bold: true, color: C.RUST } }],
    [{ text: 'SVF：形變幅度相近，網格未翻轉（全腦 0 個 folding voxel）', options: { bold: true, color: C.TEAL } }],
    [{ text: (HAS_WM.wide_full ? '上一頁藍框之放大；' : '') + '網格間距 2 mm', options: { color: C.MUTED, fontSize: 12 } }],
  ], { x: M + 7.8, y: 1.6, w: 4.33, h: 5.2, fontSize: 14, paraSpaceAfter: 12 });
}

if (HAS_METRIC) {
  // ─────────────────────────────────────────────────────── 補充評估指標：定義（make_metrics.py 的示意圖＋式 (1)～(3)）
  const s = base('補充　評估指標', 'HD95、SDlogJ 與 folding 之定義');
  fitImage(s, CH('1014_metric_demo.png'), M, 1.38, 12.13, 2.75, 'HD95 與 SDlogJ 之示意（合成之 2D 例子）');
  fitImage(s, CH('1014_metric_eq.png'), M, 4.2, 12.13, 2.72, 'HD95、SDlogJ、folding 之公式');
}

if (HAS_METRIC) {
  // ─────────────────────────────────────────────────────── 補充評估指標：結果（gather.py 的 surface、surface_paired）
  const NAMEP = { mix_exp2: ['SVF', 'half-res.', '2', 'default'], mix_exp5: ['SVF', 'full-res.', '2', 'default'],
    mix_exp6: ['SVF', 'full-res.', '1', 'default'], mix_exp7: ['SVF', 'full-res.', '0.5', 'default'],
    mix_exp4: ['Displacement', 'full-res.', '2', 'default'], mix_exp3: ['Displacement', 'full-res.', '1', 'default'],
    mix_wide: ['Displacement', 'full-res.', '1', '2×'], mix_wide_vel: ['SVF', 'full-res.', '1', '2×'] };
  // 標題與重點的數字都從 gather.py 的 surface／surface_paired 讀；⚠️ 文字是照 2026-10-07 的結果寫的
  // （HD95 各模型差距小、SDlogJ 隨 λ 變小而上升、displacement field 之 SDlogJ 主要來自 folding voxel）
  const fmtF = (x) => (x === 0 ? '0' : x < 0.001 ? '< 0.001' : x.toFixed(3));
  const best = (k, lo) => Object.keys(NAMEP).reduce((a, b) => ((lo ? SF[b][k] < SF[a][k] : SF[b][k] > SF[a][k]) ? b : a));
  const bD = best('dice', false), bH = best('hd95', true);
  const ids = Object.keys(NAMEP);
  const hdLo = Math.min(...ids.map((e) => SF[e].hd95)), hdHi = Math.max(...ids.map((e) => SF[e].hd95));
  // 2026-10-07 結果：同條件下 SVF 之 HD95 較 displacement field 低（邊界對位較好）；λ 1 → 0.5 之 HD95 反而變差；
  //   displacement field 之 SDlogJ 主要來自 folding voxel。數字與 p 值從 gather.py 的 surface_paired 讀
  const SPD = D.surface_paired;
  const hdv = (k) => SPD[k].hd95_mean;
  const thou = (x) => Math.round(x).toLocaleString('en-US');
  const pvs = (q) => (q.p < 0.001 ? 'p < 0.001' : 'p = ' + q.p.toFixed(2));
  const METRIC_TITLE = '補充指標：同條件下 SVF 之 HD95 較低，形變較平滑';
  const sub = (t) => ({ text: '\n' + t, options: { color: C.MUTED, fontSize: 12 } });
  const METRIC_BULLETS = [
    [{ text: 'HD95：affine ' + SF.affine.hd95.toFixed(2) + ' mm → ' + hdLo.toFixed(2) + '～' + hdHi.toFixed(2)
       + ' mm；同條件下 SVF 低於 displacement field；最低為 ' + bH, options: { bold: true } },
     sub('default width ' + (-hdv('version_w1').mean).toFixed(3) + ' mm（' + pvs(hdv('version_w1')) + '）、2× width '
       + hdv('version_wide').mean.toFixed(3) + ' mm（' + pvs(hdv('version_wide')) + '）；Dice 相近時，SVF 之邊界誤差較小')],
    [{ text: 'SVF 之 λ 2 → 1 → 0.5：SDlogJ ' + ['mix_exp5', 'mix_exp6', 'mix_exp7'].map((e) => SF[e].sdlogj.toFixed(2)).join(' → ')
       + '；HD95 於 λ = 1 最低', options: { bold: true } },
     sub('λ 1 → 0.5 之 HD95 ' + sgn(hdv('lam_vel_05').mean, 3) + ' mm（' + pvs(hdv('lam_vel_05')) + '）；displacement field 之 SDlogJ（'
       + SF.mix_exp4.sdlogj.toFixed(2) + '～' + SF.mix_exp3.sdlogj.toFixed(2) + '）主要來自 folding voxel（log 10⁻⁹ = −20.7）')],
    [{ text: 'Folding voxels：displacement field λ = 1 平均 ' + thou(SF.mix_exp3.fold_n) + '（2× width ' + thou(SF.mix_wide.fold_n)
       + '），與論文 VoxelMorph (CC) 之 19,077 相當', options: { bold: true } },
     sub('λ = 2 為 ' + thou(SF.mix_exp4.fold_n) + '；SVF 各模型 < 15。論文百分比之分母為 520 萬 voxel，與本研究不同，故以 voxel 數比較')],
  ];
  const s = base('補充　評估指標', METRIC_TITLE);
  const cellv = (e, k, txt) => ({ text: txt, options: (e === bD && k === 'dice') || (e === bH && k === 'hd95')
    ? { bold: true, color: C.TEAL } : {} });
  const rows = [['Model', '參數化', '解析度', 'λ', 'Width', 'Dice ↑', 'HD95（mm）↓', 'SDlogJ ↓', 'Folding（%）↓', 'Folding voxels ↓'],
    [{ text: 'Affine（形變配準前）', options: { color: C.MUTED } }, '—', '—', '—', '—', f3(SF.affine.dice),
     SF.affine.hd95.toFixed(2), '0', '0', '0']];
  Object.keys(NAMEP).forEach((e) => {
    const [pz, rs, lam, wd] = NAMEP[e];
    rows.push([e + (SF[e].amp ? ' *' : ''), pz, rs, lam, wd, cellv(e, 'dice', f3(SF[e].dice)),
               cellv(e, 'hd95', SF[e].hd95.toFixed(2)), SF[e].sdlogj.toFixed(3), fmtF(SF[e].fold_fg),
               SF[e].fold_n === 0 ? '0' : SF[e].fold_n < 10 ? SF[e].fold_n.toFixed(1) : thou(SF[e].fold_n)]);
  });
  table(s, rows, { x: M, y: 1.42, w: 12.13, colW: [2.05, 1.25, 0.95, 0.5, 0.8, 0.85, 1.5, 1.1, 1.5, 1.63], fontSize: 11.5, rowH: 0.28 });
  bullets(s, METRIC_BULLETS, { x: M, y: 4.88, w: 12.13, h: 1.8, fontSize: 13, paraSpaceAfter: 4 });
  const anyAmp = ids.some((e) => SF[e].amp);
  txt(s, 'test 51 位之平均；HD95 為 30 個結構之平均。Folding（%）之分母為 atlas 非背景 voxel（187 萬）；Folding voxels 為每位平均。'
    + (anyAmp ? '* 以半精度推論（與單精度之差異可忽略）' : '全部以單精度（float32）推論'),
    { x: M, y: 6.7, w: 12.13, h: 0.3, fontSize: 11, color: C.MUTED });
}

if (HAS_ARCH) {
  // ─────────────────────────────────────────────────────── ⑥ 架構修改：文獻依據（數字照論文抄，寫死在 make_charts.py）
  // 2026-10-07 使用者：「正式一點、公式 block 都很重要、不要太口語」→ ⑥ 全部改為學術用語＋架構圖＋編號公式（make_arch.py）
  const s = base('⑥ 架構修改（紅字以外）', '架構修改方向：加入配準專用設計，而非更換 backbone');
  fitImage(s, CH('1014_arch_lit.png'), M, 1.4, 12.13, 3.9, '更換 backbone 與加入 coarse-to-fine 之比較；LUMIR 2024 排名');
  txt(s, '資料來源：Jian et al., WBIR 2024, Table 2（LPBA 200 pairs；training: OASIS、ADNI、IXI）；LUMIR 2024 test leaderboard（Learn2Reg 2024）',
    { x: M, y: 5.33, w: 12.13, h: 0.3, fontSize: 10.5, color: C.MUTED });
  bullets(s, [
    [{ text: '更換 backbone（Mamba、Transformer）之 DSC 差異 < 1%；coarse-to-fine 提升 3.4%', options: { bold: true } }],
    [{ text: 'LUMIR 2024 前段方法（SITReg、VFA）皆採 coarse-to-fine；VoxelMorph 排名後段', options: { bold: true } }],
    [{ text: 'VoxelMorph 僅於最後一層輸出形變場（Balakrishnan et al., TMI 2019, Fig. 3）', options: { color: C.MUTED } }],
    [{ text: '實驗設計：以 mix_exp6 為 baseline，每次僅改變一項架構因素', options: { bold: true, color: C.TEAL } }],
  ], { x: M, y: 5.7, w: 12.13, h: 1.3, fontSize: 13.5, paraSpaceAfter: 3 });
}

if (HAS_ARCH) {
  // ─────────────────────────────────────────────────────── ⑥ Step 0：test-time recursion（ASD/test_multipass.py；公式 make_arch.py）
  const MP = D.multipass, E6 = MP.mix_exp6, VW = D.multipass_vs_wide;
  const three = ['mix_exp6', 'mix_exp3', 'mix_wide_vel'].filter((e) => MP[e]);
  const allWin = three.every((e) => MP[e].gain2.win === MP[e].gain2.n);
  const s = base('⑥ 架構修改（紅字以外）', 'Step 0：Test-time recursion（不重新訓練）');
  fitImage(s, CH('1014_arch_step0.png'), M, 1.4, 7.3, 4.1, '同一模型遞迴 1、2、3 次之 test DSC');
  bullets(s, [
    [{ text: '2 passes：' + (allWin ? '三個模型皆 ' + E6.gain2.n + '/' + E6.gain2.n : E6.gain2.win + '/' + E6.gain2.n) + ' 位 DSC 上升',
       options: { bold: true, color: C.TEAL } },
     { text: '\n　mix_exp6：ΔDSC = ' + sgn(E6.gain2.mean, 4) + '（Wilcoxon ' + pval(E6.gain2.p) + '）\n　3 passes 無進一步改善',
       options: { color: C.MUTED, fontSize: 12.5 } }],
    [{ text: 'mix_exp6 × 2（' + f3(VW.two) + '）> mix_wide_vel（' + f3(VW.wide) + '）', options: { bold: true } },
     { text: '\n　' + VW.win + '/' + VW.n + ' 位；參數量為其 1/4，且不需重新訓練', options: { color: C.MUTED, fontSize: 12.5 } }],
    [{ text: '改善集中於 cortex（ΔDSC ' + sgn(E6.struct['大腦皮質'], 3) + '）', options: { bold: true } },
     { text: '\n　affine baseline 最低 10 位 ' + sgn(E6.diff.hard10, 3) + '、最高 10 位 ' + sgn(E6.diff.easy10, 3),
       options: { color: C.MUTED, fontSize: 12.5 } }],
    [{ text: 'Folding：~0 → ' + Math.round(E6.passes[1].points) + ' voxels／subject（SVF）', options: { bold: true, color: C.RUST } }],
  ], { x: 8.15, y: 1.5, w: 4.58, h: 4.0, fontSize: 14, paraSpaceAfter: 8 });
  fitImage(s, CH('1014_arch_step0_eq.png'), M, 5.62, 12.13, 1.4, 'Test-time recursion 之公式');
}

if (HAS_ARCH) {
  // ─────────────────────────────────────────────────────── ⑥ Step 1：Cascade（架構圖＋式 (2)(3)，make_arch.py）
  const s = base('⑥ 架構修改（紅字以外）', 'Step 1：Cascaded registration（Zhao et al., ICCV 2019）');
  fitImage(s, CH('1014_arch_cascade.png'), M, 1.45, 12.13, 2.75, 'Cascade 架構圖');
  fitImage(s, CH('1014_arch_cascade_eq.png'), M, 4.35, 12.13, 2.4, 'Cascade 公式 (2)(3)');
}

if (HAS_ARCH) {
  // ─────────────────────────────────────────────────────── ⑥ Step 2：Coarse-to-fine（架構圖＋式 (4)(5)，make_arch.py）
  const s = base('⑥ 架構修改（紅字以外）', 'Step 2：Coarse-to-fine（multi-resolution pyramid + warping）');
  fitImage(s, CH('1014_arch_pyramid.png'), M, 1.4, 12.13, 3.45, 'Coarse-to-fine 架構圖');
  fitImage(s, CH('1014_arch_pyramid_eq.png'), M, 4.85, 12.13, 2.25, 'Coarse-to-fine 公式 (4)(5)');
}

if (HAS_ARCH) {
  // ─────────────────────────────────────────────────────── ⑥ 實驗設計（2 × 2）與進度（gather.py 的 arch）
  // 參數從網路計算；每步倍數、顯存、訓練時間為筆電實測＋外插（寫死於 gather.py，手冊 §25.3、§25.4）
  const byE = Object.fromEntries(D.arch.map((a) => [a.exp, a]));
  const s = base('⑥ 架構修改（紅字以外）', '實驗設計（2 × 2）與進度');
  const st = (e) => (done(e) ? 'DSC ' + score(e) : (e === 'mix_cascade_pyramid' ? '待前兩者結果' : '待訓練'));
  const cell = (e) => ({ text: [{ text: e, options: { breakLine: true, fontSize: 12, color: C.MUTED } },
                                { text: st(e), options: { bold: true, color: done(e) ? C.INK : C.RUST } }] });
  table(s, [
    ['', 'Single stage', 'Cascade（2 stages）'],
    [{ text: 'Single-resolution\n（VoxelMorph）', options: { bold: true } }, cell('mix_exp6'), cell('mix_cascade')],
    [{ text: 'Coarse-to-fine', options: { bold: true } }, cell('mix_pyramid'), cell('mix_cascade_pyramid')],
  ], { x: M, y: 1.6, w: 6.5, colW: [2.1, 1.9, 2.5], fontSize: 14, rowH: 0.85 });
  const NAME = { mix_exp6: 'VoxelMorph（mix_exp6）', mix_pyramid: 'Coarse-to-fine', mix_cascade: 'Cascade',
                 mix_cascade_pyramid: 'Cascade + coarse-to-fine' };
  table(s, [
    ['Model', '#Params', '訓練時間（TITAN RTX）'],
    ...['mix_exp6', 'mix_pyramid', 'mix_cascade', 'mix_cascade_pyramid'].map((e) => [
      NAME[e], (byE[e].params / 1e6).toFixed(2) + ' M', done(e) ? '完成' : '約 ' + byE[e].hours + ' h']),
  ], { x: 7.3, y: 1.6, w: 5.43, colW: [2.4, 0.95, 2.08], fontSize: 13 });
  bullets(s, [
    [{ text: '實作驗證：', options: { bold: true } },
     { text: '1-stage cascade 與 VoxelMorph 輸出一致；translation test 確認各層 upsampling 與 composition 正確' }],
    [{ text: '訓練順序：', options: { bold: true } },
     { text: 'mix_exp8／9（semi-supervised）→ coarse-to-fine → cascade → cascade + coarse-to-fine' }],
  ], { x: M, y: 4.75, w: 12.13, h: 1.4, fontSize: 15, paraSpaceAfter: 12 });
}

// ───────────────────────────────────────────────────────── 14 下一步
{
  const s = base('NEXT', '下一步');
  const PEND = ['mix_exp6', 'mix_exp7', 'mix_wide_vel'].filter((e) => !done(e));
  const where = [...new Set(PEND.map((e) => (e === 'mix_wide_vel' ? '第 ' + PG.wide + ' 頁（2× width SVF）'
                                                                  : '第 ' + PG.lam + ' 頁（λ）')))];
  const items = PEND.length
    ? [[{ text: PEND.join('、') + ' 訓練完成後', options: { bold: true } },
        { text: '\n　更新' + where.join('與'), options: { color: C.MUTED } }]]
    : [];
  bullets(s, items.concat([
    [{ text: '半監督訓練：訓練時加入 FreeSurfer 標籤（Balakrishnan et al., TMI 2019, Eq. 10）', options: { bold: true } },
     { text: '\n　γ = 0.5、5 兩個模型（mix_exp8／9）接續訓練；測試時僅使用影像，評估 Dice 是否進一步提升', options: { color: C.MUTED } }],
    ...(HAS_ARCH ? [[{ text: '架構修改：coarse-to-fine → cascade → cascade + coarse-to-fine（第 ' + PG.arch_plan + ' 頁）', options: { bold: true } },
                     { text: '\n　接續上述兩個模型依序訓練；皆以 mix_exp6 為 baseline，每次僅改變一項因素', options: { color: C.MUTED } }]] : []),
    [{ text: '新資料：第五批 MRS（' + D.mrs.n + ' 位）已完成前處理', options: { bold: true } },
     { text: '\n　待其餘資料到齊後一併納入，重新切分 train／val／test，作為新版資料集', options: { color: C.MUTED } }],
    [{ text: '外部資料（附帶結果）：模型應用於未參與訓練之 MRS 研究，Dice ' + f3(D.mrs.after) + '（affine ' + f3(D.mrs.before) + '）', options: { bold: true } },
     { text: '\n　與原 test set 之 ' + score('mix_exp3') + ' 相當，顯示模型可泛化至不同研究之資料', options: { color: C.MUTED } }],
  ]), { x: M, y: 1.8, w: 12.13, h: 4.2, fontSize: 16, paraSpaceAfter: 16 });
}

if (page !== ORDER.length) throw new Error('頁數對不上：做了 ' + page + ' 頁，ORDER 排了 ' + ORDER.length + ' 頁（內文引用的頁碼會錯）');
pres.writeFile({ fileName: OUT }).then(() => console.log('ok ->', OUT, '｜' + page + ' 頁'));
