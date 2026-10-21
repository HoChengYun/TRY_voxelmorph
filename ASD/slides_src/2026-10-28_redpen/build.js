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
// 2026-10-21 使用者「開做」：加 ⑥ 架構修改之結果（2 × 2 完成、延長訓練）、⑧ 與 ANTs SyN 之比較、附錄（影像邊界之 folding）。
// 數字從 deck_data_arch.json 讀（make_arch_res.py）；ANTs 與 VoxelMorph 並排之受試者圖（make_ants_vis.py）有了才加那一頁
const DA_P = path.join(HERE, 'deck_data_arch.json');
const DA = fs.existsSync(DA_P) ? JSON.parse(fs.readFileSync(DA_P, 'utf8')) : null;
const HAS_ANTS_VIS = fs.existsSync(path.join(MROOT, 'deck_charts', '1028_ants_vis.png'));
const ORDER = ['rv_def', 'rv_res', ...(D.models ? ['rv_models'] : []),
  ...(DA ? ['arch_2x2', 'arch_metrics', 'arch_p500', 'ants_cmp', ...(HAS_ANTS_VIS ? ['ants_vis'] : []), 'app_bgfold'] : [])];
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

const FOOT = { x: M, y: 6.78, w: 11.6, h: 0.3, fontSize: 10.5, color: C.MUTED };
const wp = (q) => '（' + q.win + '/' + q.n + '）';
const pv = (q) => (q.p < 0.001 ? 'p < 0.001' : 'p = ' + q.p.toFixed(3));
const EB6 = '⑥ 架構修改之結果（紅字以外）';
const EB8 = '⑧ 與傳統方法之比較（紅字以外）';

if (DA) {
  // ─────────────────────────────────────────────── ⑥ 2 × 2 結果（10/14 時 coarse-to-fine 與合併模型尚在訓練）
  const m = DA.m, P = DA.p, PR = DA.params;
  const s = base(EB6, '2 × 2 實驗結果：cascade 與 coarse-to-fine 合併之 Dice 最高');
  const cell = (e, name) => {
    const q = P[e + '_vs_exp6'];
    const runs = [{ text: name, options: { breakLine: true, fontSize: 11.5, color: C.MUTED } },
                  { text: 'Dice ' + f3(m[e].dice), options: { bold: true, breakLine: true } }];
    if (q) runs.push({ text: 'ΔDice ' + sgn(q.dice.mean, 4) + wp(q.dice), options: { fontSize: 12, color: C.TEAL, breakLine: true } });
    runs.push({ text: (PR[e] / 1e6).toFixed(2) + ' M params', options: { fontSize: 11.5, color: C.MUTED } });
    return { text: runs };
  };
  table(s, [
    ['', 'Single stage', 'Cascade（2 stages）'],
    [{ text: 'Single-resolution\n（VoxelMorph）', options: { bold: true } }, cell('mix_exp6', 'mix_exp6'), cell('mix_cascade', 'mix_cascade')],
    [{ text: 'Coarse-to-fine', options: { bold: true } }, cell('mix_pyramid', 'mix_pyramid'),
      cell('mix_cascade_pyramid', 'mix_cascade_pyramid')],
  ], { x: M, y: 1.5, w: 6.2, colW: [1.75, 2.2, 2.25], fontSize: 13, rowH: [0.4, 1.45, 1.45] });
  fitImage(s, CH('1028_arch_2x2.png'), 6.95, 1.35, 5.78, 4.15, '每位受試者相對 mix_exp6 之 ΔDice');
  const A = DA.additive, H = DA.hard.mix_cascade_pyramid;
  bullets(s, [
    [{ text: '合併模型高於 cascade ' + sgn(P.cp_vs_cascade.mean, 4) + wp(P.cp_vs_cascade) + '、coarse-to-fine '
      + sgn(P.cp_vs_pyramid.mean, 4) + wp(P.cp_vs_pyramid) + '（Wilcoxon，皆 ' + pv(P.cp_vs_cascade) + '）', options: { bold: true } }],
    [{ text: '兩項設計之增益部分可加成：合併之 ΔDice（' + sgn(A.both, 4) + '）為兩者之和（' + sgn(A.cascade + A.pyramid, 4) + '）之 '
      + Math.round(100 * A.ratio) + '%' }],
    [{ text: 'Affine Dice 最低 10 位改善 ' + sgn(H.hard10, 3) + '，最高 10 位 ' + sgn(H.easy10, 3) + '（相對 mix_exp6）' }],
  ], { x: M, y: 5.6, w: 12.13, h: 1.15, fontSize: 13.5, paraSpaceAfter: 4 });
  txt(s, 'test ' + DA.n + ' 位；ΔDice 相對 mix_exp6（SVF，λ = 1，250 epochs）；四個模型之訓練設定相同，僅架構不同；括號內為較高之人數',
    FOOT);
}

if (DA) {
  // ─────────────────────────────────────────────── ⑥ 多項指標
  const m = DA.m, P = DA.p, C6 = P.mix_cascade_pyramid_vs_exp6;
  const s = base(EB6, '多項指標：合併模型之 Dice、未參與訓練之結構、HD95 皆最佳');
  fitImage(s, CH('1028_arch_res.png'), M, 1.35, 12.13, 3.35, '五個模型之 Dice、未參與訓練之結構、HD95、SDlogJ、folding');
  bullets(s, [
    [{ text: '未參與訓練之結構 Dice ' + f3(m.mix_exp6.held) + ' → ' + f3(m.mix_cascade_pyramid.held) + wp(C6.held)
      + '；形變後影像與 atlas 之 NCC ' + sgn(C6.ncc.mean, 3) + wp(C6.ncc) + '：改善不限於評估之 30 結構', options: { bold: true } }],
    [{ text: 'Coarse-to-fine 與 cascade 之 Dice 相近（' + sgn(P.pyramid_vs_cascade.mean, 4) + '，' + pv(P.pyramid_vs_cascade)
      + '），參數量為其 ' + (DA.params.mix_pyramid / DA.params.mix_cascade).toFixed(2) + ' 倍' }],
    [{ text: '代價：SDlogJ ' + m.mix_exp6.sdlogj.toFixed(2) + ' → ' + m.mix_cascade_pyramid.sdlogj.toFixed(2) + '；腦內 folding '
      + m.mix_exp6.fold.toFixed(1) + ' → ' + Math.round(m.mix_cascade_pyramid.fold) + ' voxels／位（論文 VoxelMorph (CC) '
      + thou(DA.paper.vxm_fold) + '）', options: { color: C.RUST } }],
  ], { x: M, y: 4.85, w: 12.13, h: 1.8, fontSize: 13.5, paraSpaceAfter: 6 });
  txt(s, 'test ' + DA.n + ' 位之平均；未參與訓練之結構：Accumbens、Inf-Lat-Vent、胼胝體；HD95 為 30 結構之平均；'
    + 'SDlogJ 與 folding 僅計 atlas 腦內（影像邊界之 folding 見附錄）', FOOT);
}

if (DA) {
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
  const d = (a, b) => '（' + sgn(a - b, 3) + '）';
  table(s, [
    ['', '論文 Table I', '本研究（test ' + DA.n + ' 位）'],
    ['Affine', f3(T.affine), f3(m.affine.dice)],
    ['ANTs SyN', f3(T.ants) + d(T.ants, T.affine), f3(m.ants.dice) + d(m.ants.dice, m.affine.dice)],
    ['VoxelMorph', f3(T.vxm) + d(T.vxm, T.affine), f3(m.mix_exp3.dice) + d(m.mix_exp3.dice, m.affine.dice)],
    ['VoxelMorph − ANTs', sgn(T.vxm - T.ants, 3), sgn(P.mix_exp3_vs_ants.mean, 4) + wp(P.mix_exp3_vs_ants)],
    ['Folding voxels', 'ANTs ' + thou(T.ants_fold) + '／VxM ' + thou(T.vxm_fold), 'ANTs ' + thou(m.ants.fold) + '／VxM ' + thou(m.mix_exp3.fold)],
    ['時間／位', 'ANTs ' + thou(T.ants_sec) + ' s（CPU）', 'ANTs ' + thou(DA.ants.sec_mean) + ' s（CPU，' + DA.ants.threads + ' threads）'],
  ], { x: 7.0, y: 1.45, w: 5.73, colW: [1.65, 1.85, 2.23], fontSize: 11.5, rowH: 0.42 });
  bullets(s, [
    [{ text: '本研究 Dice 之絕對值較高，主因為 affine 較高（' + f3(m.affine.dice) + ' vs ' + f3(T.affine) + '）；ΔDice 反而較小',
       options: { bold: true } }],
    [{ text: '傳統方法亦得到相近之 Dice → 本資料集較易配準，非模型較強；與論文比較應以相對量為準' }],
    [{ text: '架構修改後之模型高於 ANTs SyN：合併模型 ' + sgn(P.mix_cascade_pyramid_vs_ants.mean, 4) + wp(P.mix_cascade_pyramid_vs_ants),
       options: { color: C.TEAL, bold: true } }],
  ], { x: M, y: 5.55, w: 12.13, h: 1.2, fontSize: 13.5, paraSpaceAfter: 4 });
  txt(s, 'ANTs SyN 參數依論文 §V-A-3：step 0.25、Gaussian (9, 0.2)、3 scales × 201 iterations、CC（半徑 4）；affine 已於前處理完成，SyN 僅做非線性部分；'
    + 'VoxelMorph = mix_exp3（displacement field，λ = 1，同論文 Table I）', FOOT);
}

if (DA && HAS_ANTS_VIS) {
  // ─────────────────────────────────────────────── ⑧ ANTs 與 VoxelMorph 之形變結果並排（make_ants_vis.py）
  const s = base(EB8, '形變結果之比較：ANTs SyN、VoxelMorph 與合併模型');
  fitImage(s, CH('1028_ants_vis.png'), M, 1.35, 12.13, 5.35, 'ANTs SyN 與 VoxelMorph 之形變結果');
}

if (DA) {
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

if (page !== ORDER.length) throw new Error('頁數對不上：做了 ' + page + ' 頁，ORDER 排了 ' + ORDER.length + ' 頁');
ORDER.forEach((k, i) => {
  if (!NOTES[k]) console.warn('[!] 講稿.md 沒有 ' + k + '（第 ' + (i + 1) + ' 頁）');
  else if (NOTES[k].page !== i + 1) console.warn('[!] 講稿.md 的 ' + k + ' 寫第 ' + NOTES[k].page + ' 頁，實際是第 ' + (i + 1) + ' 頁');
});
pres.writeFile({ fileName: OUT }).then(() => console.log('ok -> ' + path.basename(OUT) + ' ｜' + page + ' 頁'));
