// ASD 延伸實驗簡報：三包混合訓練 × tigerbx 對照
// 配色沿用上一份 meeting 簡報（墨黑 / 青綠 / 鏽紅 / 米白底）。
// 投影片上的數字一律從 deck_data.json 讀（由 gather.py 從原始 CSV 算出），不手打。
const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const OUT = process.argv[2] || path.join(HERE, 'deck.pptx');
const D = JSON.parse(fs.readFileSync(path.join(HERE, 'deck_data.json'), 'utf8'));

const C = {
  INK: '141A1D', PAPER: 'FAFAF8', SURF: 'EFEFEB', RULE: 'D9D9D2', MUTED: '5F6A6B',
  TEAL: '0E7C7B', TEAL_L: '5BB8B6', RUST: 'A34F1B', RUST_L: 'D9895A',
  DARK: '1A2125', DARK2: '2A3237', DMUTED: '8A9294', WHITE: 'FFFFFF', SOFT: 'D9DEDF',
};
const F = { SANS: 'Microsoft JhengHei', MONO: 'Consolas' };

const f3 = (x) => x.toFixed(3);
const f4 = (x) => x.toFixed(4);
const sg = (x, d = 4) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';            // 13.333 × 7.5 in
pres.title = 'ASD 延伸實驗：三包混合訓練 × tigerbx 對照';
pres.author = 'HoChengYun';

const W = 13.333, M = 0.6;
let page = 0;
// 頁碼不手打：base() 依 eyebrow 記下每頁的頁碼，需要引用後面頁碼的內容放進 later，最後才畫
const PG = {};
const later = [];
function pg(eyebrow, last) {
  const a = PG[eyebrow];
  if (!a) throw new Error('沒有這個 eyebrow：' + eyebrow);
  return String(last ? a[a.length - 1] : a[0]);
}
const rng = (a, b) => (a === b ? a : `${a}–${b}`);

function pngSize(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function txt(s, text, o) {
  // lang 設 zh-TW：pptxgenjs 預設 en-US，PowerPoint 就不套中文的標點避頭規則，「。」會跑到行首
  s.addText(text, Object.assign({ fontFace: F.SANS, color: C.INK, margin: 0, isTextBox: true, valign: 'top', lang: 'zh-TW' }, o));
}

function base(eyebrow, title, notes) {
  const s = pres.addSlide();
  page += 1;
  (PG[eyebrow] = PG[eyebrow] || []).push(page);
  s.background = { color: C.PAPER };
  txt(s, eyebrow, { x: M, y: 0.42, w: 11.5, h: 0.28, fontFace: F.MONO, fontSize: 11, bold: true, color: C.MUTED, charSpacing: 3 });
  txt(s, title, { x: M, y: 0.74, w: 12.1, h: 0.66, fontSize: 28, bold: true });
  txt(s, String(page).padStart(2, '0'), { x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.28, fontFace: F.MONO, fontSize: 11, color: C.MUTED, align: 'right' });
  if (notes) s.addNotes(notes);
  return s;
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || C.SURF }, line: { color: C.RULE, width: 0.75 } });
}

function label(s, t, x, y, w, color) {
  txt(s, t, { x, y, w, h: 0.26, fontFace: F.MONO, fontSize: 10.5, bold: true, color: color || C.MUTED, charSpacing: 2 });
}

function stat(s, big, small, x, y, w, color) {
  txt(s, big, { x, y, w, h: 0.62, fontFace: F.MONO, fontSize: 32, bold: true, color: color || C.TEAL });
  txt(s, small, { x, y: y + 0.66, w, h: 0.5, fontSize: 12, color: C.MUTED });
}

// 表格：第一列是表頭
function table(s, rows, o) {
  const head = rows[0].map((t) => ({ text: t, options: { bold: true, color: C.MUTED, fontFace: F.MONO, fontSize: 10.5, fill: { color: C.PAPER } } }));
  const body = rows.slice(1).map((r) => r.map((c) => (typeof c === 'object' ? c : { text: String(c) })));
  s.addTable([head].concat(body), Object.assign({
    fontFace: F.SANS, fontSize: 12.5, color: C.INK, valign: 'middle', lang: 'zh-TW',
    border: { type: 'solid', pt: 0.75, color: C.RULE }, fill: { color: C.PAPER },
    margin: [0.05, 0.1, 0.05, 0.1],
  }, o));
}
const hl = (t, color, bold) => ({ text: t, options: { color: color || C.TEAL, bold: bold !== false } });

// 條列：每項一段，第一段粗體小標可選
function bullets(s, items, o) {
  const arr = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    if (Array.isArray(it)) {
      it.forEach((run, j) => arr.push({ text: run.text, options: Object.assign({ bullet: j === 0 ? { indent: 14 } : undefined, breakLine: j === it.length - 1 && !last }, run.options) }));
    } else {
      arr.push({ text: it, options: { bullet: { indent: 14 }, breakLine: !last } });
    }
  });
  s.addText(arr, Object.assign({ fontFace: F.SANS, fontSize: 14, color: C.INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 8, lang: 'zh-TW' }, o));
}

function lineChart(s, series, o) {
  const opts = Object.assign({
    chartColors: series.map((x) => x.color),
    lineSize: 2, lineDataSymbol: 'circle', lineDataSymbolSize: 5,
    valAxisMinVal: 0.64, valAxisMaxVal: 0.88, valAxisMajorUnit: 0.04, valAxisLabelFormatCode: '0.00',
    catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, catAxisLabelFontFace: F.MONO, valAxisLabelFontFace: F.MONO,
    catAxisLabelColor: C.MUTED, valAxisLabelColor: C.MUTED,
    valGridLine: { color: 'E4E4DE', size: 0.5 }, catGridLine: { style: 'none' },
    catAxisLineColor: C.RULE, valAxisLineShow: false,
    showLegend: series.length > 1, legendPos: 'b', legendFontFace: F.SANS, legendFontSize: 11, legendColor: C.INK,
    showCatAxisTitle: true, catAxisTitle: 'epoch', catAxisTitleFontSize: 10, catAxisTitleColor: C.MUTED,
    showValAxisTitle: true, valAxisTitle: 'Dice（28 顆 test 平均）', valAxisTitleFontSize: 10, valAxisTitleColor: C.MUTED,
    catAxisLabelFrequency: 5,
  }, o);
  s.addChart(pres.charts.LINE, series.map((x) => ({ name: x.name, labels: x.labels, values: x.values })), opts);
}

const curve = (k) => D['curve_' + k];
const S = D.summary;
const P_MIX = D.plateau_mix_exp1, P_TG = D.plateau_tiger_exp1;
const DS = D.by_ds;

// ── 視覺化圖片：直接引用 models/ 底下 visualize_dice.py / visualize_reg_ixi.py 的輸出 ──
const MROOT = path.join(HERE, '..', '..', '..', 'models');   // 專案根目錄的 models/
const MX = (sid, kind) => `${MROOT}/mix_exp1/vis_${sid}/${kind}_${sid}_0230.png`;
const TG = (sid, kind) => `${MROOT}/tiger_exp1/vis_${sid}/${kind}_${sid}_0240.png`;
const MX_TRI = (sid) => `${MROOT}/mix_exp1/vis_${sid}/reg_${sid}_0230_triplanar.png`;
const AU = (sid, kind) => `${MROOT}/author_exp1/vis_OAS1_${sid}/${kind}_OASIS_OAS1_${sid}_MR1_vxm_dense_brain_T1_3D_mse.png`;
const SUB = (id) => {
  const x = D.subjects.find((q) => q.id === id);
  if (!x) throw new Error('deck_data.json 裡沒有 ' + id);
  return x;
};

// 等比例縮放塞進框內；align='left' 靠左，否則水平置中
function fitImage(s, p, x, y, maxW, maxH, alt, align) {
  if (!fs.existsSync(p)) throw new Error('找不到圖片 ' + p);
  const z = pngSize(p);
  let w = maxW, h = maxW * z.h / z.w;
  if (h > maxH) { h = maxH; w = maxH * z.w / z.h; }
  const dx = align === 'left' ? 0 : (maxW - w) / 2;
  s.addImage({ path: p, x: x + dx, y, w, h, altText: alt || '' });
  return { x: x + dx, y, w, h };
}

// 兩組並排：左 FreeSurfer、右 tigerbx
function pairSlide(eyebrow, title, notes, left, right, caption) {
  const s = base(eyebrow, title, notes);
  const colW = 5.95, gap = 0.2, y0 = 1.5;
  [left, right].forEach((it, i) => {
    const x = M + i * (colW + gap);
    txt(s, [{ text: it.head + '   ', options: { bold: true, color: it.color } }, { text: it.sub, options: { color: C.MUTED } }],
      { x, y: y0, w: colW, h: 0.34, fontSize: 13.5 });
    fitImage(s, it.img, x, y0 + 0.42, colW, 4.45, it.alt);
  });
  txt(s, caption, { x: M, y: 6.3, w: 12.1, h: 0.62, fontSize: 12.5 });
  return s;
}

// ════════════════════════════════════════════════════════════════
// 01 封面
{
  const s = pres.addSlide(); page += 1;
  s.background = { color: C.DARK };
  txt(s, 'MEETING REPORT  ·  2026-09', { x: 0.8, y: 0.85, w: 11, h: 0.3, fontFace: F.MONO, fontSize: 12, bold: true, color: C.DMUTED, charSpacing: 3 });
  txt(s, 'ASD 延伸實驗', { x: 0.8, y: 1.45, w: 11.5, h: 0.95, fontSize: 46, bold: true, color: C.WHITE });
  txt(s, '三包資料混合訓練  ×  tigerbx 標籤對照', { x: 0.8, y: 2.45, w: 11.5, h: 0.6, fontSize: 24, color: C.SOFT });
  txt(s, '回應上次 meeting 的紅筆問題', { x: 0.8, y: 3.1, w: 11.5, h: 0.4, fontSize: 16, color: C.DMUTED });
  const cols = [
    ['286', '位受試者\nASD 164 · DGM 54 · VNT 68'],
    [f4(S.fs_after), 'Dice（30 個結構）\nFreeSurfer 標籤，28 顆 test'],
    ['0.000%', '形變場折疊率\n兩組、全部檢查點'],
  ];
  cols.forEach(([big, small], i) => {
    const x = 0.8 + i * 3.9;
    txt(s, big, { x, y: 4.45, w: 3.6, h: 0.8, fontFace: F.MONO, fontSize: 40, bold: true, color: C.TEAL_L });
    txt(s, small, { x, y: 5.3, w: 3.6, h: 0.7, fontSize: 13, color: C.DMUTED });
  });
  txt(s, 'VoxelMorph × MNI152 2009c   ·   FreeSurfer 7.4.1   ·   tigerbx', { x: 0.8, y: 6.75, w: 11, h: 0.3, fontFace: F.MONO, fontSize: 11, color: C.DMUTED });
  s.addNotes('這次報告回應上次 meeting 老師用紅筆標的問題，並加入兩個新實驗：三包資料混合訓練，以及用 tigerbx 的分割取代 FreeSurfer 做對照。');
}

// 02 紅筆問題總覽
{
  const s = base('LAST MEETING', '上次的紅筆問題：狀態總覽',
    '上次簡報上的紅筆一共 8 件事。7 件有答案，第 2 題（頭殼沒切好）沒有直接測試，只有旁證。');
  const ok = hl('●', C.TEAL), part = hl('●', C.RUST);
  // 頁碼要等全部頁面排完才知道 → 延後畫
  later.push(() => {
    const rows = [
      ['', '位置', '問題', '答案', '頁'],
      [ok, 'p04', 'T065 標「重複」', '不是重複：建檔時編號打成 T056，資料正確 → 已納回', pg('DATA QUALITY')],
      [ok, 'p04', 'A016_1 標「拿掉」', '已排除；另查出 A016_2 是品管掃描，一併排除', pg('DATA QUALITY')],
      [ok, 'p10', '去看檔頭', 'A0131 / A0132 是同一位 5 歲男童；YT13 是 A0131 的重複匯出', rng(pg('DICOM HEADER'), pg('SAME-PERSON CHECK'))],
      [ok, 'p11', 'VoxelMorph 的 atlas 哪來的', '論文只寫「由外部資料集算出」，沒有檔案與參數，無法重現', pg('ATLAS SOURCE')],
      [ok, 'p24 ①', '跟論文差多少', '模型貢獻：我們 +0.112、作者 +0.169；另附論文全表與逐項差異', rng(pg('VS. PAPER'), pg('VS. PAPER · DIFF ②'))],
      [part, 'p24 ②', '頭殼沒切好的效果', '沒有直接測試，只有旁證', pg('SKULL STRIPPING')],
      [ok, 'p24 ③', '同資料換 tigerbx', '模型貢獻多 +0.010；比的是兩條完整流程', rng(pg('TIGERBX · DESIGN'), pg('TIGERBX · VISUAL', true))],
      [ok, 'p24 ④', '混多一點資料集', '三包共 286 位；資料 +73%，進步小於標準誤', rng(pg('DATASETS'), pg('RESULT · MIX_EXP1'))],
    ];
    table(s, rows, { x: M, y: 1.65, w: 12.1, colW: [0.4, 0.9, 2.9, 7.2, 0.7], rowH: 0.5 });
  });
  txt(s, [hl('●', C.TEAL), { text: ' 已回答     ' }, hl('●', C.RUST), { text: ' 部分回答' }], { x: M, y: 6.55, w: 6, h: 0.3, fontSize: 11.5, color: C.MUTED });
}

// 03 資料把關修正
{
  const s = base('DATA QUALITY', '資料把關修正：170 → 164',
    '這頁的去留全部有 DICOM 檔頭或檔案層級的證據。T065 是這次唯一翻案的：原本以為身分不明，其實只是編號打錯。');
  const X = hl('排除', C.RUST), IN = hl('納回', C.TEAL), KEEP = hl('已分離重跑，納入', C.TEAL);
  table(s, [
    ['受試者', '問題', '處置'],
    ['A043', '雜訊過高、灰白對比不足', X],
    ['T085', '只有 120 / 192 張切片，來源即缺', X],
    ['A016_1', '皮質面積僅中位數 54%；技師當場標註 low contrast', X],
    ['A016_2', 'ID 是 QA＋日期、性別欄 O：品管掃描，不是受試者', X],
    ['YT13', '與 A0131 是同一次掃描（見下頁）', X],
    ['A0132', '與 A0131 是同一人的第二次掃描', X],
    ['T065', '建檔時誤植為 T056；資料夾內留有說明檔', IN],
    ['A012', '資料夾混了兩次掃描', KEEP],
  ], { x: M, y: 1.65, w: 8.4, colW: [1.2, 5.3, 1.9], rowH: 0.52 });
  const X0 = 9.5;
  stat(s, '164', 'ASD 最終可用受試者', X0, 1.75, 3.2);
  stat(s, '4,903', '個 DICOM 序列逐一讀檔頭', X0, 3.15, 3.2);
  stat(s, '148 / 16', 'ASD 的 train / test（受試者層級）', X0, 4.55, 3.2, C.INK);
}

// 04 DICOM 檔頭
{
  const s = base('DICOM HEADER', '去看檔頭：同一人與重複掃描',
    'A013 是另一個人。A0131 和 A0132 是同一個小孩隔 23 天掃兩次。YT13 跟 A0131 的掃描時間精確到秒都一樣，是同一次掃描被匯出成兩個 ID。重點是最後那個框：舊模型並沒有因為這個洩漏而分數變高。');
  table(s, [
    ['資料夾', '年齡', '性別', '體重', '掃描日期與時間'],
    ['A013', '23', 'M', '78 kg', '2021-11-13  11:55'],
    [hl('A0131', C.INK), '5', 'M', '22 kg', hl('2022-11-04  12:20:18', C.RUST)],
    [hl('YT13', C.INK), '5', 'M', '22 kg', hl('2022-11-04  12:20:18', C.RUST)],
    ['A0132', '5', 'M', '22 kg', '2022-11-27  11:11'],
  ], { x: M, y: 1.7, w: 6.9, colW: [1.3, 0.9, 0.9, 1.1, 2.7], rowH: 0.55 });
  txt(s, 'A0131 與 YT13 的掃描時間精確到秒相同 —— 不可能是兩次掃描。', { x: M, y: 4.6, w: 6.9, h: 0.5, fontSize: 12.5, color: C.MUTED });
  bullets(s, [
    [{ text: 'A013 ', options: { bold: true } }, { text: '是另一個人（23 歲成人）' }],
    [{ text: 'A0131 / A0132 ', options: { bold: true } }, { text: '是同一位男童，相隔 23 天' }],
    [{ text: 'YT13 ', options: { bold: true } }, { text: '是 A0131 同一次掃描的重複匯出' }],
    [{ text: 'DGM ', options: { bold: true } }, { text: '另有兩對同一人：D015 / D037、D038 / DGM002 → 切分時綁在同一邊' }],
  ], { x: 7.95, y: 1.75, w: 4.75, h: 2.9, fontSize: 14 });
  const O = D.old17;
  card(s, 7.95, 4.75, 4.75, 1.7);
  label(s, '對舊結果的影響', 8.2, 4.95, 4.3, C.TEAL);
  txt(s, [
    { text: `舊模型的 ${O.n} 顆 test 裡，A0131 反而是最低的（${f4(O.a0131)}）。排除它，平均 `, options: {} },
    { text: sg(O.mean_excl - O.mean), options: { bold: true } },
    { text: `，小於標準誤 ${f4(O.sem)}。\n→ 這次洩漏沒有把分數灌高；方法學仍已修正。`, options: {} },
  ], { x: 8.2, y: 5.3, w: 4.3, h: 1.1, fontSize: 12.5 });
}

// 05 同人偵測
{
  const s = base('SAME-PERSON CHECK', '怎麼確認是不是同一人：atlas 空間的標籤 Dice',
    'DICOM 的人口學欄位在這批不可全信，技師會複製上一位的登錄資料。所以另外用影像本身檢查：兩顆腦對到 atlas 之後，30 個結構的標籤重疊多少。同一次掃描 0.98、成人同一人 0.85，都遠在不同人的分布之外。但 5 歲小孩的兩次掃描只有 0.73，跟不同人分不開。');
  const pairs = D.dup_pairs.slice().reverse();
  const same = pairs.map((p) => (p[2].startsWith('不同人') ? 0 : p[1]));
  const diff = pairs.map((p) => (p[2].startsWith('不同人') ? p[1] : 0));
  s.addChart(pres.charts.BAR, [
    { name: '已知同一人／同一次掃描', labels: pairs.map((p) => p[0]), values: same },
    { name: '已知不同人', labels: pairs.map((p) => p[0]), values: diff },
  ], {
    x: M, y: 1.6, w: 7.2, h: 4.6, barDir: 'bar', barGrouping: 'stacked', barGapWidthPct: 55,
    chartColors: [C.TEAL, C.RUST],
    valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: '0.0',
    showValue: true, dataLabelPosition: 'inEnd', dataLabelFormatCode: '0.000;;;', dataLabelColor: C.WHITE,
    dataLabelFontFace: F.MONO, dataLabelFontSize: 11, dataLabelFontBold: true,
    catAxisLabelFontFace: F.MONO, catAxisLabelFontSize: 11, catAxisLabelColor: C.INK,
    valAxisLabelFontFace: F.MONO, valAxisLabelFontSize: 10, valAxisLabelColor: C.MUTED,
    valGridLine: { color: 'E4E4DE', size: 0.5 }, catGridLine: { style: 'none' },
    showLegend: true, legendPos: 'b', legendFontFace: F.SANS, legendFontSize: 11,
  });
  table(s, [
    ['不同人的分布', '對數', '中位數', '第 99 百分位'],
    ['ASD', '13,366', '0.663', '0.721'],
    ['DGM', '1,431', '0.656', '0.722'],
    ['VNT', '2,278', '0.653', '0.719'],
  ], { x: 8.2, y: 1.7, w: 4.5, colW: [1.25, 0.95, 0.95, 1.35], rowH: 0.42, fontSize: 12 });
  bullets(s, [
    [{ text: '抓得到：', options: { bold: true, color: C.TEAL } }, { text: '同一次掃描（0.98）、成人同一人（0.85）' }],
    [{ text: '抓不到：', options: { bold: true, color: C.RUST } }, { text: '5 歲兒童的兩次掃描只有 0.73，落在不同人的範圍內' }],
    [{ text: '教訓：', options: { bold: true } }, { text: '一開始用 1 對同人 vs 8 對不同人定門檻，全掃之後誤報幾百對 —— 小樣本定的門檻不能用' }],
  ], { x: 8.2, y: 3.75, w: 4.5, h: 2.8, fontSize: 13 });
}

// 06 atlas 哪來的
{
  const s = base('ATLAS SOURCE', 'VoxelMorph 作者的 atlas 從哪來',
    '論文本身只交代一句，沒有提供檔案或參數，所以作者那顆 atlas 我們重現不出來。兩邊有一點相同：atlas 都跟訓練資料互相獨立。');
  card(s, M, 1.65, 5.7, 4.95);
  label(s, '論文原文', M + 0.3, 1.9, 5.2, C.TEAL);
  // 只節錄一句原文，其餘轉述 —— 這一句就是論文對 atlas 來源的全部交代
  txt(s, '“We use an atlas computed using an external dataset [1], [68].”',
    { x: M + 0.3, y: 2.25, w: 5.1, h: 0.85, fontSize: 16, italic: true });
  txt(s, 'Balakrishnan et al., IEEE TMI 38(8), 2019, §V-B', { x: M + 0.3, y: 3.1, w: 5.1, h: 0.28, fontSize: 11, color: C.MUTED });
  txt(s, '[1]  B. Fischl, “FreeSurfer,” NeuroImage 62(2), 2012\n[68] R. Sridharan et al., Quantification and analysis of large multimodal clinical image studies: Application to stroke',
    { x: M + 0.3, y: 3.45, w: 5.1, h: 0.85, fontSize: 10.5, color: C.MUTED });
  bullets(s, [
    '論文對 atlas 的交代就只有這一句：沒有檔案、參數與人數 → 無法重現',
    'repo 附的 atlas.npz 大小 160×192×224，與論文寫的裁切尺寸一致；README 沒有說明來源',
    '訓練資料是另外 8 個公開資料集共 3,731 顆，atlas 不從其中產生',
  ], { x: M + 0.3, y: 4.45, w: 5.1, h: 2.05, fontSize: 13 });
  table(s, [
    ['', '論文', '我們'],
    ['atlas 來源', '外部資料集的平均模板', 'MNI152 2009c（152 人非線性平均）'],
    ['atlas 的標籤', '隨 repo 附', '自己跑 FreeSurfer 7.4.1；tigerbx 組另跑 tigerbx'],
    ['尺寸', '160 × 192 × 224', '192 × 224 × 192'],
    ['影像值域', '最大 0.728（未正規化）', '正規化到 [0, 1]'],
    ['與訓練資料獨立', hl('是'), hl('是')],
  ], { x: 6.65, y: 1.65, w: 6.05, colW: [1.55, 2.1, 2.4], rowH: 0.62, fontSize: 12 });
  txt(s, '值域差異對 NCC 幾乎沒有影響（NCC 對線性亮度變換不敏感），若改用 MSE 才會是變因。', { x: 6.65, y: 5.6, w: 6.05, h: 0.7, fontSize: 12, color: C.MUTED });
}

// 07 三包資料
{
  const s = base('DATASETS', '混多一點資料集：ASD + DGM + VNT',
    '三包都是同一台 Skyra 3T、同一個 MPRAGE 協定。DGM 裡有兩對同一人，所以 54 個掃描是 52 個人。VNT 年紀最大，中位數 36 歲。下方那行是合併的根據：三包彼此之間的相似度分布幾乎一樣。');
  const dem = D.demo;
  const info = [
    ['ASD', '164 位', `${dem.ASD.age_median} 歲（${dem.ASD.age_min}–${dem.ASD.age_max}）`, `${dem.ASD.under18} 位`, '148 / 16', C.TEAL],
    ['DGM', '54 掃描 = 52 人', `${dem.DGM.age_median} 歲（${dem.DGM.age_min}–${dem.DGM.age_max}）`, '0', '49 / 5', C.INK],
    ['VNT', '68 位', `${dem.VNT.age_median} 歲（${dem.VNT.age_min}–${dem.VNT.age_max}）`, '0', '61 / 7', C.RUST],
  ];
  info.forEach(([name, n, age, u18, split, col], i) => {
    const x = M + i * 4.1, y = 1.65, w = 3.8;
    card(s, x, y, w, 3.35);
    txt(s, name, { x: x + 0.3, y: y + 0.25, w: 3, h: 0.6, fontFace: F.MONO, fontSize: 30, bold: true, color: col });
    const rows = [['受試者', n], ['年齡中位數', age], ['未滿 18 歲', u18], ['train / test', split]];
    rows.forEach(([k, v], j) => {
      txt(s, k, { x: x + 0.3, y: y + 1.05 + j * 0.53, w: 1.4, h: 0.4, fontSize: 12, color: C.MUTED });
      txt(s, v, { x: x + 1.65, y: y + 1.05 + j * 0.53, w: 2.0, h: 0.4, fontSize: 13.5, bold: true });
    });
  });
  card(s, M, 5.25, 12.1, 1.35, C.PAPER);
  label(s, '為什麼可以合併', M + 0.3, 5.45, 5, C.TEAL);
  txt(s, [
    { text: '同一台 Skyra 3T、同一個 MPRAGE 協定；而且三包內部「不同人」之間的標籤 Dice 分布幾乎一樣（中位數 0.663 / 0.656 / 0.653，第 99 百分位 0.721 / 0.722 / 0.719）', options: {} },
    { text: '—— 從資料本身支持三包可以合併。合計 ', options: {} },
    { text: 'train 258 / test 28', options: { bold: true } },
    { text: '。', options: {} },
  ], { x: M + 0.3, y: 5.8, w: 11.5, h: 0.75, fontSize: 13 });
}

// 08 混合訓練結果
{
  const s = base('RESULT · MIX_EXP1', '三包混合訓練：Dice 曲線',
    'epoch 0 就是只做線性對位的基準線，0.6753，跟另一條程式路徑算出來的一模一樣，代表評估流程沒錯。epoch 90 之後各點都在雜訊範圍內，所以「最佳 epoch 230」沒有特別意義。');
  const c = curve('mix_exp1');
  lineChart(s, [{ name: 'FreeSurfer 標籤', color: C.TEAL, labels: c.map((x) => String(x[0])), values: c.map((x) => x[1]) }],
    { x: M, y: 1.55, w: 7.6, h: 5.1, valAxisMinVal: 0.64, valAxisMaxVal: 0.80, valAxisMajorUnit: 0.02 });
  const X0 = 8.55;
  stat(s, f4(S.fs_base), '基準線（只做線性對位）', X0, 1.65, 4.1, C.MUTED);
  stat(s, f4(P_MIX.best), `模型後（epoch ${P_MIX.best_epoch}）`, X0, 2.85, 4.1, C.TEAL);
  stat(s, sg(S.fs_gain), '模型貢獻 · 折疊率 0.000%', X0, 4.05, 4.1, C.INK);
  txt(s, [
    { text: 'epoch ≥ 90 全距 ', options: {} }, { text: f4(P_MIX.range), options: { bold: true } },
    { text: `，小於 2 倍標準誤 ${f4(2 * S.sem_fs_base)} → 彼此分不出高下。epoch 80 已到 0.78，下次 120 epochs 就夠。`, options: {} },
  ], { x: X0, y: 5.3, w: 4.1, h: 1.3, fontSize: 12.5, color: C.INK });
}

// 09 跟論文比
{
  const s = base('VS. PAPER', '做完的效果跟論文差多少',
    '絕對值我們比較高，但那是因為起點高。看模型自己的貢獻，我們 +0.112、論文 +0.169，我們比較低。而且資料、標籤、atlas 全都不同，嚴格說不能直接比。折疊率 0 對 0.366% 也不能直接比：我們用的是 repo 預設的微分同胚版，形變場透過積分本來就不容易折疊；論文主結果是非微分同胚版。');
  table(s, [
    ['', '只做線性對位', '模型後', '模型貢獻', '折疊率'],
    ['作者（論文 Table I）', '0.584', '0.753', hl('+0.169', C.INK), '0.366%'],
    ['我們 · FreeSurfer 標籤', f3(S.fs_base), f3(S.fs_after), hl(sg(S.fs_gain, 3), C.TEAL), '0.000%'],
    ['我們 · tigerbx 標籤', f3(S.tg_base), f3(S.tg_after), hl(sg(S.tg_gain, 3), C.TEAL), '0.000%'],
  ], { x: M, y: 1.7, w: 7.6, colW: [2.6, 1.35, 1.1, 1.3, 1.25], rowH: 0.6 });
  s.addChart(pres.charts.BAR, [{ name: '模型貢獻', labels: ['作者', '我們 · FS', '我們 · tigerbx'], values: [0.169, S.fs_gain, S.tg_gain] }], {
    x: 8.55, y: 1.6, w: 4.15, h: 3.2, barDir: 'col', chartColors: [C.MUTED, C.TEAL, C.TEAL_L],
    valAxisMinVal: 0, valAxisMaxVal: 0.2, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: '0.00',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '+0.000', dataLabelFontFace: F.MONO, dataLabelFontSize: 11,
    catAxisLabelFontSize: 11, catAxisLabelFontFace: F.SANS, valAxisLabelFontSize: 10, valAxisLabelFontFace: F.MONO,
    catAxisLabelColor: C.INK, valAxisLabelColor: C.MUTED, valGridLine: { color: 'E4E4DE', size: 0.5 }, catGridLine: { style: 'none' },
    showLegend: false, showTitle: true, title: '模型貢獻（模型後 − 基準線）', titleFontSize: 12, titleColor: C.INK, titleFontFace: F.SANS,
  });
  later.push(() => bullets(s, [
    [{ text: '絕對值我們較高，', options: { bold: true } }, { text: '但那是因為起點就比論文高約 0.09' }],
    [{ text: '看模型貢獻，我們比較低：', options: { bold: true } }, { text: '+0.112 vs +0.169' }],
    '起點越高，剩下能進步的空間越小（Dice 有天花板），但這個天花板我們量化不出來',
    `資料集、標籤來源、atlas、形變場版本全都不同 → 只能參考；論文全表見第 ${pg('VS. PAPER · TABLE I')} 頁，逐項差異見第 ${pg('VS. PAPER · DIFF ①')}–${pg('VS. PAPER · DIFF ②')} 頁`,
    [{ text: '折疊率也不能直接比：', options: { bold: true } }, { text: '我們用 repo 預設的微分同胚版（int_steps=7），論文主結果是非微分同胚版（int_steps=0）' }],
  ], { x: M, y: 4.4, w: 12.1, h: 2.4, fontSize: 13 }));
}

// ── 跟作者比：論文全表、論文其他數字、逐項差異 ──────────────────────
const B = D.bench, SDP = D.sd_pooled;
if (!B) throw new Error('沒有 bench.json —— 先跑 bench.py 量推論時間');
const GPU_NAME = B.gpu.name.replace('NVIDIA GeForce ', '');
const bold = (t) => ({ text: t, options: { bold: true } });

// 論文 Table I 全表，下面接我們
{
  const s = base('VS. PAPER · TABLE I', '作者公布的數字：論文 Table I 全表，接上我們',
    '上半部是論文 Table I 原封不動的數字，括號裡是標準差。下半部是我們，用同一種算法：Dice 的標準差一樣是把所有受試者乘上 30 個結構的 Dice 攤平之後算。時間是在這台筆電上量的，論文用 TitanX 和 Xeon，硬體不同，只能看數量級；論文沒寫 CPU 用幾個線程，我們用單線程，比較保守。折疊的部分，論文的 VoxelMorph (CC) 每顆平均約 1.9 萬個 voxel 折疊；我們兩組 28 位一個都沒有，但那主要是因為我們用的是微分同胚版，不代表模型比較好。');
  const sec = (t) => [{ text: t, options: { colspan: 6, bold: true, color: C.WHITE, fill: { color: C.DARK2 }, fontSize: 11.5 } }];
  const ds = (m, sd) => `${f3(m)} (${f3(sd)})`;
  const gpu = `${B.gpu.mean.toFixed(2)} (${B.gpu.sd.toFixed(2)})`;
  const cpu = `${B.cpu.mean.toFixed(1)} (${B.cpu.sd.toFixed(1)})`;
  table(s, [
    ['方法', 'Dice（標準差）', 'GPU 秒', 'CPU 秒', '折疊 voxel 數', '折疊率 %'],
    sec('作者 · 論文 Table I      8 個公開資料集共 3,731 顆，test 250 顆  ·  TitanX GPU / Xeon E5-2680 CPU'),
    ['只做線性對位（Affine only）', '0.584 (0.157)', '0', '0', '0', '0'],
    ['ANTs SyN (CC)', '0.749 (0.136)', '–', '9059 (2023)', '9662 (6258)', '0.185 (0.091)'],
    ['NiftyReg (CC)', '0.755 (0.143)', '–', '2347 (202)', '41251 (14336)', '0.793 (0.208)'],
    [bold('VoxelMorph (CC)'), bold('0.753 (0.145)'), '0.45 (0.01)', '57 (1)', '19077 (5928)', '0.366 (0.114)'],
    ['VoxelMorph (MSE)', '0.752 (0.140)', '0.45 (0.01)', '57 (1)', '9606 (4516)', '0.184 (0.087)'],
    sec(`我們      ASD + DGM + VNT 共 286 位，test ${S.n} 位  ·  ${GPU_NAME} / 單線程 CPU`),
    ['只做線性對位 · FreeSurfer 標籤', ds(S.fs_base, SDP.fs_base), '0', '0', '0', '0'],
    [hl('VoxelMorph (CC) · FreeSurfer 標籤', C.TEAL), hl(ds(S.fs_after, SDP.fs_after), C.TEAL), gpu, cpu, '0', '0'],
    ['只做線性對位 · tigerbx 標籤', ds(S.tg_base, SDP.tg_base), '0', '0', '0', '0'],
    [hl('VoxelMorph (CC) · tigerbx 標籤', C.RUST), hl(ds(S.tg_after, SDP.tg_after), C.RUST), gpu, cpu, '0', '0'],
  ], { x: M, y: 1.5, w: 12.1, colW: [3.75, 1.95, 1.3, 1.5, 1.9, 1.7], rowH: 0.36, fontSize: 12 });
  bullets(s, [
    `標準差：跟論文同一種算法，所有受試者 × 30 個結構的 Dice 攤平後計算（我們共 ${SDP.n_fs_after} 個值）`,
    `時間：我們是 ${GPU_NAME}、單線程 CPU（論文沒寫線程數），影像也比論文大 20% → 只能看數量級；兩組網路相同，只量一次`,
    '折疊：論文的分母是腦內 520 萬個 voxel；我們兩組 28 位都是 0 個，主要因為用的是微分同胚版（int_steps=7）',
  ], { x: M, y: 6.0, w: 12.1, h: 0.95, fontSize: 11, color: C.MUTED, paraSpaceAfter: 2 });
}

// 作者的模型跑作者的資料（OASIS 4 位，手冊 §19）
{
  const AO = D.author_oasis;
  if (!AO) throw new Error('沒有 author_exp1 的 Dice CSV —— 先用 test_dice.py 算（手冊 §19.5）');
  const gainA = AO.after - AO.base;
  const s = base('VS. PAPER · AUTHOR MODEL', '作者的模型跑作者的資料：OASIS 4 位',
    `論文 Table I 那顆模型拿不到，但官方有釋出一顆預訓練的腦部模型，我拿它跑作者自己的資料 OASIS，看作者的模型本身能做到多少。${AO.subjects.length} 位平均從 ${f3(AO.base)} 升到 ${f3(AO.after)}，進步 ${sg(gainA, 3)}，跟論文 Table I 的 +0.169 很接近。我們的模型貢獻是 ${sg(S.fs_gain, 3)} 和 ${sg(S.tg_gain, 3)}，比它小。所以我們的絕對值比較高，是因為起點高，不是模型比較強。有三件事要注意：這顆不是 Table I 那顆，是官方預訓練的 MSE 微分同胚版；這幾位作者訓練時可能看過，所以不能當測試分數；作者用 TensorFlow，這台載不起來，我照原架構搬進 PyTorch，並用故意弄壞形變場的方式確認沒有搬錯。`);
  const short = (id) => id.replace('OASIS_', '').replace('_MR1', '');
  table(s, [
    ['受試者', '只做線性對位', '作者模型', '進步'],
    ...AO.subjects.map((x) => [short(x.id), f3(x.base), f3(x.after), sg(x.after - x.base, 3)]),
    [bold(`${AO.subjects.length} 位平均`), bold(f3(AO.base)), bold(f3(AO.after)), hl(sg(gainA, 3), C.INK)],
  ], { x: M, y: 1.5, w: 6.0, colW: [1.5, 1.6, 1.5, 1.4], rowH: 0.38, fontSize: 12.5 });
  txt(s, [{ text: '作者模型的貢獻接近論文、比我們大', options: { bold: true, breakLine: true } },
    { text: '→ 我們絕對值較高是因為起點高，不是模型較強' }],
  { x: M, y: 3.9, w: 6.0, h: 0.55, fontSize: 12.5 });
  s.addChart(pres.charts.BAR, [{ name: '模型貢獻', labels: ['論文 Table I', '作者模型 · OASIS', '我們 · FS', '我們 · tigerbx'], values: [0.169, gainA, S.fs_gain, S.tg_gain] }], {
    x: M, y: 4.5, w: 6.0, h: 2.4, barDir: 'col', chartColors: [C.MUTED, C.INK, C.TEAL, C.TEAL_L],
    valAxisMinVal: 0, valAxisMaxVal: 0.2, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: '0.00',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '+0.000', dataLabelFontFace: F.MONO, dataLabelFontSize: 10,
    catAxisLabelFontSize: 10, catAxisLabelFontFace: F.SANS, valAxisLabelFontSize: 9, valAxisLabelFontFace: F.MONO,
    catAxisLabelColor: C.INK, valAxisLabelColor: C.MUTED, valGridLine: { color: 'E4E4DE', size: 0.5 }, catGridLine: { style: 'none' },
    showLegend: false, showTitle: true, title: '模型貢獻（模型後 − 只做線性對位）', titleFontSize: 11, titleColor: C.INK, titleFontFace: F.SANS,
  });
  const X2 = 6.9, W2 = 5.8;
  const im = fitImage(s, AU('0395', 'labels'), X2, 1.45, W2, 3.75, 'OAS1_0395 標籤重疊：作者模型配準前後');
  txt(s, 'OAS1_0395（最接近 4 位平均）。紅 = 只有 atlas 有、綠 = 只有受試者有、黃 = 重疊；上排只做線性對位，下排加上作者模型。',
    { x: X2, y: im.y + im.h + 0.05, w: W2, h: 0.5, fontSize: 10.5, color: C.MUTED });
  bullets(s, [
    [bold('不是 Table I 那顆：'), { text: '官方預訓練的 MSE、微分同胚版' }],
    [bold('不是測試分數：'), { text: '這幾位作者訓練時可能看過（OASIS 在作者的 3,731 顆裡）' }],
    [bold('有確認沒搬錯：'), { text: '作者用 TensorFlow，照原架構搬進 PyTorch；故意弄壞形變場，Dice 就從 0.77 掉到 0.54 / 0.45' }],
  ], { x: X2, y: 5.85, w: W2, h: 1.1, fontSize: 11.5, paraSpaceAfter: 3 });
}

// 論文其他跟我們有關的數字：Table II（手動標註）、Fig. 7（λ）
{
  const s = base('VS. PAPER · OTHER', '論文裡其他跟我們有關的數字',
    '這頁放論文裡另外兩組跟我們有關的數字。左邊是 Table II：作者把同一批模型拿去測 Buckner40 的 39 顆專家手動標註，絕對值變高，但模型貢獻幾乎不變，跟我們換 tigerbx 的觀察是同一個方向。右邊是 Fig. 7 的 λ 敏感度，論文只有圖，數字是從圖上量的，誤差大約 0.001。我們用的 λ = 1.0 正好落在 CC 的最佳區。Table III 是受試者對受試者、Table IV 是訓練時加入分割標籤，設定跟我們不同，所以沒放。');
  const wL = 5.95, X2 = 6.95, wR = 5.75;
  label(s, 'TABLE II · 換成專家手動標註（Buckner40，39 顆）', M, 1.55, wL, C.TEAL);
  const t2 = [['只做線性對位（Affine only）', 0.608, 0.175], ['ANTs SyN (CC)', 0.776, 0.130], ['NiftyReg (CC)', 0.776, 0.132],
    ['VoxelMorph (MSE)', 0.766, 0.133], ['VoxelMorph (MSE) inst.', 0.776, 0.132], ['VoxelMorph (CC)', 0.774, 0.133], ['VoxelMorph (CC) inst.', 0.786, 0.132]];
  table(s, [['方法', 'Dice（標準差）', '比線性對位多'],
    ...t2.map(([n, d, sd], i) => (n === 'VoxelMorph (CC)' ? [bold(n), bold(`${f3(d)} (${f3(sd)})`), bold(sg(d - 0.608, 3))]
      : [n, `${f3(d)} (${f3(sd)})`, i === 0 ? '—' : sg(d - 0.608, 3)]))],
  { x: M, y: 1.9, w: wL, colW: [2.75, 1.7, 1.5], rowH: 0.35, fontSize: 12 });
  txt(s, 'inst. = 每一對再用梯度下降微調 100 次（GPU 23.7 秒、單線程 CPU 628 秒）。模型沿用 Table I 那批，沒有重訓。',
    { x: M, y: 4.82, w: wL, h: 0.5, fontSize: 10.5, color: C.MUTED });
  card(s, M, 5.4, wL, 1.45);
  label(s, '跟我們 tigerbx 對照的關係', M + 0.25, 5.55, wL - 0.5, C.TEAL);
  txt(s, [
    { text: '作者換成手動標註（受試者也換了）：絕對值 0.753 → 0.774，貢獻 +0.169 → +0.166，幾乎不變。' },
    { text: `我們換成 tigerbx：絕對值差 ${f3(S.tg_after - S.fs_after)}，貢獻只差 ${sg(S.gain_diff, 3)}。`, options: { bold: true } },
    { text: '方向一致：換標籤主要改變絕對值。' },
  ], { x: M + 0.25, y: 5.88, w: wL - 0.5, h: 0.95, fontSize: 11.5 });

  label(s, 'FIG. 7 · λ 敏感度（validation Dice，從圖上量出，約 ±0.001）', X2, 1.55, wR, C.TEAL);
  const cc = [['0', '0.687'], ['0.5', '0.743'], ['1', '0.746'], ['1.5', '0.746'], ['2', '0.745'], ['5', '0.737']];
  const mse = [['0', '0.688'], ['0.005', '0.742'], ['0.01', '0.744'], ['0.02', '0.745'], ['0.05', '0.734']];
  table(s, [['λ（CC）', ...cc.map((x) => (x[0] === '1' ? '1 ← 我們' : x[0]))],
    ['Dice ≈', ...cc.map((x) => (x[0] === '1' ? hl(x[1], C.TEAL) : x[1]))]],
  { x: X2, y: 1.9, w: wR, colW: [1.0, 0.7, 0.7, 1.15, 0.7, 0.75, 0.75], rowH: 0.36, fontSize: 12 });
  table(s, [['λ（MSE）', ...mse.map((x) => x[0])], ['Dice ≈', ...mse.map((x) => x[1])]],
    { x: X2, y: 2.85, w: wR, colW: [1.0, 0.95, 0.95, 0.95, 0.95, 0.95], rowH: 0.36, fontSize: 12 });
  bullets(s, [
    [bold('我們用 λ = 1.0，'), { text: '正好在論文 CC 的最佳區（1–1.5，約 0.746）' }],
    'λ 在 0.5–2 之間只差約 0.003：論文說模型對 λ 不敏感',
    [bold('λ = 0（完全不加平滑）仍有約 0.687，'), { text: '論文說這樣也明顯優於只做線性對位' }],
    'Fig. 7 是 validation set，所以最高 0.746 跟 Table I（test）的 0.753 不同',
    [bold('沒放的：'), { text: 'Table III（受試者對受試者）、Table IV（訓練時加入分割標籤），設定跟我們不同' }],
  ], { x: X2, y: 3.8, w: wR, h: 3.05, fontSize: 12.5, paraSpaceAfter: 6 });
}

// 逐項差異 ①：資料與前處理
{
  const s = base('VS. PAPER · DIFF ①', '我們和作者的差異 ①：資料與前處理',
    '這兩頁把我們跟作者的差異逐項列出來。資料這頁最重要的是三件事：第一，我們的資料比較窄，都是同一台機器、同一個協定；第二，我們沒有 validation；第三，線性對位和 atlas 都不同，而這兩個直接決定起點。所以 Dice 的絕對值不能直接比，只能比模型貢獻，而且也只能參考。');
  const dem = D.demo;
  table(s, [
    ['項目', '作者（論文）', '我們', '會影響什麼'],
    [bold('資料'), '8 個公開資料集、3,731 顆\n多中心，年齡、疾病、掃描參數各異', `3 包、286 位；同一台 Skyra 3T、同一協定\n年齡中位數 ${dem.ASD.age_median} / ${dem.DGM.age_median} / ${dem.VNT.age_median} 歲`, '我們較窄：換到別台機器、\n別的年齡層，效果未知'],
    [bold('切分'), 'train 3,231 / validation 250 / test 250', `train 258 / validation 0 / test ${S.n}`, `test 少，平均值的標準誤約 ${f3(S.fs_after_sd / Math.sqrt(S.n))}`],
    [bold('同一人'), '未說明', 'DICOM 檔頭＋標籤 Dice 檢查，同一人放同一邊', '避免 train / test 洩漏'],
    [bold('去顱骨'), 'FreeSurfer', 'FreeSurfer（norm.mgz）；tigerbx 組用 tigerbx', 'FreeSurfer 組相同'],
    [bold('線性對位'), hl('FreeSurfer 做；對到哪個模板論文沒寫', C.RUST, false), hl('ANTs Affine，直接對到評估用的 MNI152 atlas', C.RUST, false), `可能是基準線較高的原因之一\n（${f3(S.fs_base)} vs 0.584）`],
    [bold('atlas'), hl('外部資料集的平均模板；檔案與參數都沒公布', C.RUST, false), hl('MNI152 2009c；標籤自己跑 FreeSurfer／tigerbx', C.RUST, false), 'atlas 不同 → Dice 絕對值不能直接比'],
    [bold('影像大小'), '160 × 192 × 224（由 256³、1 mm 裁切）', '192 × 224 × 192（1 mm）', '只影響速度與記憶體'],
    [bold('評估標籤'), 'FreeSurfer，30 個結構\n（每位 test 都 ≥ 100 voxel）', '同一套 30 個標籤（repo 的 labels.npz）\n另有 tigerbx 組', 'FreeSurfer 組相同'],
    [bold('品質控制'), '目視檢查分割與線性對位', '目視＋DICOM 檔頭＋資料層級檢查\n（ASD 170 → 164）', '—'],
  ], { x: M, y: 1.5, w: 12.1, colW: [1.25, 3.65, 3.8, 3.4], rowH: 0.47, fontSize: 11.5 });
  txt(s, [bold('結論：'), { text: '資料、atlas、線性對位三個起點都不同 → 只能比「模型貢獻」這類相對量，而且也只能參考。' }],
    { x: M, y: 6.5, w: 12.1, h: 0.4, fontSize: 13 });
}

// 逐項差異 ②：模型、訓練與評估
{
  const s = base('VS. PAPER · DIFF ②', '我們和作者的差異 ②：模型、訓練與評估',
    '模型這頁，網路、損失、優化器都跟論文一樣。不一樣的有四件：第一，形變場版本，我們用 repo 預設的微分同胚版，論文 Table I 是位移場版本，所以折疊率不能比。第二，λ 我們直接用論文 CC 的最佳值 1.0，自己沒有挑。第三，訓練量只有論文預設的六分之一，但曲線在 epoch 90 之後已經持平。第四，我們沒有 validation，epoch 是看 test 挑的，所以數字偏樂觀。');
  table(s, [
    ['項目', '作者（論文）', '我們', '會影響什麼'],
    [bold('形變場'), hl('位移場（Table I 主結果，等於 int_steps = 0）', C.RUST, false), hl('速度場積分，微分同胚\n（repo 預設 int_steps = 7）', C.RUST, false), '我們天生不容易折疊\n→ 折疊率 0 不代表模型較好'],
    [bold('形變場解析度'), '全解析度', '半解析度（int_downsize = 2）積分後再放大', '形變可能較平滑、細節較少'],
    [bold('網路'), 'U-Net（論文 Fig. 3）', '相同（repo 預設）\n特徵數 16-32-32-32 / 32-32-32-32-32-16-16', '—'],
    [bold('損失函數'), 'CC（窗格 9）＋ λ × 形變梯度平滑', '相同', '—'],
    [bold('λ'), hl('試 6 個 λ，每個各訓練一個模型\n用 validation Dice 挑最好的（CC 約 1–1.5）', C.RUST, false), hl('只訓練一個模型，λ = 1.0\n（直接採論文的最佳區，自己沒挑）', C.RUST, false), '沒有 validation，就沒辦法自己挑 λ'],
    [bold('訓練量'), '預設 150,000 次迭代', '250 epoch × 100 步 = 25,000 次（約 1/6）', `epoch 90 之後已持平（全距 ${f4(P_MIX.range)}）`],
    [bold('優化器'), 'ADAM，lr 1e-4，batch 1', '相同', '—'],
    [bold('選模型'), hl('依 validation Dice', C.RUST, false), hl('看 test 曲線挑 epoch', C.RUST, false), '數字偏樂觀\n（但持平段內差距小於 2 倍標準誤）'],
    [bold('框架／硬體'), 'Keras + TensorFlow；TitanX', `PyTorch；${GPU_NAME}`, '只影響速度'],
    [bold('Dice 算法'), '30 個結構 × 所有受試者平均', '相同', '—'],
  ], { x: M, y: 1.5, w: 12.1, colW: [1.45, 3.6, 3.65, 3.4], rowH: 0.42, fontSize: 11.5 });
  txt(s, [bold('最會動到結論的三項：'), { text: '形變場版本（折疊率不能比）、沒有 validation（λ 沒自己挑、數字偏樂觀）、資料與 atlas 不同（絕對值不能比）。' }],
    { x: M, y: 6.55, w: 12.1, h: 0.4, fontSize: 13 });
}

// 10 壓平
{
  const s = base('RESULT · SPREAD', '模型把受試者之間的差距壓平了',
    '每個點是一位 test 受試者。橫軸是只做線性對位的 Dice，縱軸是模型後的 Dice。模型不管你從哪裡出發，都把你拉到 0.79 附近。所以改善幅度幾乎完全由起點決定，相關係數 −0.96。報告時該報模型後的絕對值，不是改善幅度。');
  const sub = D.subjects;
  s.addChart(pres.charts.SCATTER, [
    { name: '基準線', values: sub.map((x) => x.fs_base) },
    { name: '受試者', values: sub.map((x) => x.fs_after) },
  ], {
    x: M, y: 1.55, w: 7.2, h: 5.1, lineSize: 0, lineDataSymbol: 'circle', lineDataSymbolSize: 8, chartColors: [C.TEAL],
    valAxisMinVal: 0.55, valAxisMaxVal: 0.85, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: '0.00',
    catAxisMinVal: 0.55, catAxisMaxVal: 0.75, catAxisMajorUnit: 0.05, catAxisLabelFormatCode: '0.00',
    showValAxisTitle: true, valAxisTitle: '模型後 Dice', showCatAxisTitle: true, catAxisTitle: '基準線 Dice（只做線性對位）',
    valAxisTitleFontSize: 10, catAxisTitleFontSize: 10, valAxisTitleColor: C.MUTED, catAxisTitleColor: C.MUTED,
    valAxisLabelFontFace: F.MONO, catAxisLabelFontFace: F.MONO, valAxisLabelFontSize: 10, catAxisLabelFontSize: 10,
    valAxisLabelColor: C.MUTED, catAxisLabelColor: C.MUTED,
    valGridLine: { color: 'E4E4DE', size: 0.5 }, catGridLine: { color: 'E4E4DE', size: 0.5 }, showLegend: false,
  });
  const X0 = 8.2;
  stat(s, `${f3(S.fs_base_sd)} → ${f3(S.fs_after_sd)}`, '受試者之間的標準差（縮到約 36%）', X0, 1.7, 4.5, C.TEAL);
  stat(s, S.r_gain_vs_base_fs.toFixed(2), '起點越低、進步越多（相關係數，−1 代表完全反向）', X0, 3.0, 4.5, C.RUST);
  bullets(s, [
    `A0131 為 ${f3(sub.find((x) => x.id === 'A0131').fs_after)}；其餘 27 位介於 ${f3(Math.min(...sub.filter((x) => x.id !== 'A0131').map((x) => x.fs_after)))}–${f3(Math.max(...sub.map((x) => x.fs_after)))}`,
    '起點越低，改善越多 —— 改善幅度主要由起點決定',
    [{ text: '所以該報的是模型後的絕對值，', options: { bold: true } }, { text: '不是改善幅度' }],
  ], { x: X0, y: 4.4, w: 4.5, h: 2.2, fontSize: 13.5 });
}

// 11 分資料集
{
  const s = base('RESULT · BY DATASET', '分資料集來看',
    'VNT 的起點最低，所以改善最多。但模型後三包幾乎一樣。DGM 只有 5 位、VNT 只有 7 位，分組結果不宜過度解讀。');
  const ds = ['ASD', 'DGM', 'VNT'];
  s.addChart(pres.charts.BAR, [
    { name: 'FreeSurfer · 基準線', labels: ds, values: ds.map((k) => DS[k].fs_base) },
    { name: 'FreeSurfer · 模型後', labels: ds, values: ds.map((k) => DS[k].fs_after) },
    { name: 'tigerbx · 基準線', labels: ds, values: ds.map((k) => DS[k].tg_base) },
    { name: 'tigerbx · 模型後', labels: ds, values: ds.map((k) => DS[k].tg_after) },
  ], {
    x: M, y: 1.55, w: 7.4, h: 5.1, barDir: 'col', barGrouping: 'clustered', barGapWidthPct: 60,
    chartColors: ['A9C9C8', C.TEAL, 'E8C3A9', C.RUST],
    valAxisMinVal: 0.6, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: '0.00',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelFontFace: F.MONO, dataLabelFontSize: 9,
    catAxisLabelFontFace: F.MONO, catAxisLabelFontSize: 13, catAxisLabelColor: C.INK, valAxisLabelFontFace: F.MONO, valAxisLabelFontSize: 10,
    valAxisLabelColor: C.MUTED, valGridLine: { color: 'E4E4DE', size: 0.5 }, catGridLine: { style: 'none' },
    showLegend: true, legendPos: 'b', legendFontFace: F.SANS, legendFontSize: 10.5,
  });
  table(s, [
    ['', 'n', 'FS 貢獻', 'tigerbx 貢獻'],
    ...ds.map((k) => [k, String(DS[k].n), sg(DS[k].fs_after - DS[k].fs_base, 3), sg(DS[k].tg_after - DS[k].tg_base, 3)]),
  ], { x: 8.35, y: 1.7, w: 4.35, colW: [1.0, 0.7, 1.3, 1.35], rowH: 0.45, fontSize: 12.5 });
  bullets(s, [
    'VNT 改善最多，是因為起點最低（年紀最大，中位數 36 歲）',
    [{ text: '模型後三包幾乎一樣：', options: { bold: true } }, { text: `FreeSurfer 組相差 ${f3(Math.max(...ds.map((k) => DS[k].fs_after)) - Math.min(...ds.map((k) => DS[k].fs_after)))}，tigerbx 組相差 ${f3(Math.max(...ds.map((k) => DS[k].tg_after)) - Math.min(...ds.map((k) => DS[k].tg_after)))}` }],
    'DGM 5 位、VNT 7 位，人數太少，分組只作觀察',
  ], { x: 8.35, y: 3.85, w: 4.35, h: 2.7, fontSize: 13 });
}

// ── 視覺化：FreeSurfer 組，以 T053 為例 ─────────────────────────────
{
  const s = base('VISUAL · TRIPLANAR', '配準結果看起來怎樣：三平面',
    '這幾頁用 T053 當例子，它是 28 位裡 Dice 最高的一位，最差的 A0131 在後面另外講。四欄分別是受試者、atlas、配準後、差異圖。差異圖越暗越好，亮的地方集中在皮質邊緣和腦溝。');
  const t = SUB('T053');
  fitImage(s, MX_TRI('T053'), M, 1.45, 7.4, 5.5, 'T053 三平面：受試者、atlas、配準後、差異', 'left');
  label(s, 'T053 · FreeSurfer 組', 8.3, 1.6, 4.4, C.TEAL);
  later.push(() => bullets(s, [
    '四欄由左到右：受試者原影像、atlas、配準後、兩者差異',
    '差異圖越暗代表對得越好；亮的地方集中在皮質邊緣與腦溝',
    [{ text: '皮質腦溝是個體差異最大的地方，', options: { bold: true } }, { text: '也是 Dice 主要扣分之處' }],
    `T053 是 28 位中 Dice 最高的一位（${f3(t.fs_base)} → ${f3(t.fs_after)}）；最差的 A0131 見第 ${pg('CASE · A0131')} 頁`,
  ], { x: 8.3, y: 2.0, w: 4.4, h: 4.6, fontSize: 13.5 }));
}

{
  const s = base('VISUAL · OVERLAY', '線性對位 vs. 加上 VoxelMorph：疊圖',
    '紅色是 atlas，灰階是受試者。上排只做線性對位，下排加上 VoxelMorph。腦室、腦溝、小腦的邊界在下排明顯更貼合。右邊的 NCC、SSIM 是圖上標的數字；NCC 含背景會飽和，所以主要結論還是看 Dice。');
  fitImage(s, MX('T053', 'overlay'), M, 1.45, 7.9, 5.5, 'T053 疊圖：線性對位與 VoxelMorph 各自疊上 atlas', 'left');
  const X0 = 8.75;
  stat(s, '0.771 → 0.915', 'SSIM（結構相似度）', X0, 1.65, 3.95, C.TEAL);
  stat(s, '0.915 → 0.971', 'NCC（全域相關）', X0, 2.95, 3.95, C.INK);
  bullets(s, [
    '紅色是 atlas，灰階是受試者；上排只做線性對位，下排加上 VoxelMorph',
    '下排的腦室、腦溝與小腦邊界明顯更貼合',
    [{ text: 'NCC 含背景、容易飽和，', options: { bold: true } }, { text: '主要結論看 Dice，不看 NCC' }],
  ], { x: X0, y: 4.3, w: 3.95, h: 2.4, fontSize: 13 });
}

{
  const s = base('VISUAL · OUTLINES', '結構輪廓：配準前後對照',
    '實線是 atlas 的結構邊界，虛線是受試者的。上排只做線性對位，小腦和殼核的虛線明顯錯開；下排加上 VoxelMorph 之後幾乎重合。為了圖不要太亂只畫 9 個結構，Dice 是 30 個結構的平均。');
  const t = SUB('T053');
  fitImage(s, MX('T053', 'contours'), M, 1.45, 7.7, 5.5, 'T053 結構輪廓，配準前後', 'left');
  const X0 = 8.55;
  label(s, 'T053 · FreeSurfer 組', X0, 1.6, 4.15, C.TEAL);
  stat(s, `${f3(t.fs_base)} → ${f3(t.fs_after)}`, '這位受試者的 Dice（30 個結構平均）', X0, 2.0, 4.15, C.TEAL);
  bullets(s, [
    '實線是 atlas 的結構邊界，虛線是受試者的',
    '上排只做線性對位：小腦、殼核的虛線明顯錯開',
    '下排加上 VoxelMorph：虛線幾乎與實線重合',
    '只畫 9 個結構以免圖太亂；Dice 是 30 個結構的平均',
  ], { x: X0, y: 3.5, w: 4.15, h: 3.2, fontSize: 13 });
}

{
  const s = base('VISUAL · CHECKERBOARD & GRID', '棋盤格與形變網格',
    '上面是棋盤格，方格交替取自配準後影像和 atlas，交界處的輪廓接得起來就代表對得準。下面是形變網格，把規則網格套上模型的形變場，應該是平滑的彎曲，沒有打結或交叉。');
  const a = fitImage(s, MX('T053', 'checker'), M, 1.45, 7.5, 2.7, 'T053 棋盤格', 'left');
  fitImage(s, MX('T053', 'grid'), M, a.y + a.h + 0.12, 7.5, 2.7, 'T053 形變網格', 'left');
  const X0 = 8.35, w = 4.35;
  label(s, '棋盤格', X0, 1.6, w, C.TEAL);
  txt(s, '方格交替取自配準後影像與 atlas。交界處的腦輪廓若能接起來，就代表對得準。', { x: X0, y: 1.95, w, h: 1.3, fontSize: 13.5 });
  label(s, '形變網格', X0, 4.3, w, C.TEAL);
  txt(s, '把規則網格套上模型的形變場。網格應該是平滑的彎曲，不該打結、交叉或擠成一團。', { x: X0, y: 4.65, w, h: 1.3, fontSize: 13.5 });
}

{
  const s = base('VISUAL · JACOBIAN', 'Jacobian：形變場有沒有折疊',
    'Jacobian 行列式描述每個 voxel 的局部體積變化。紅色大於 1 是擴張，藍色小於 1 是壓縮。小於等於 0 就是折疊，這張圖上一個都沒有，兩組所有檢查點的折疊率都是 0。');
  fitImage(s, MX('T053', 'jacobian'), M, 1.45, 12.1, 4.3, 'T053 Jacobian 行列式圖');
  bullets(s, [
    [{ text: '紅（> 1）局部擴張，藍（< 1）局部壓縮，', options: { bold: true } }, { text: '白色 ≈ 1 表示體積不變' }],
    [{ text: '沒有任何 ≤ 0 的點：', options: { bold: true, color: C.TEAL } }, { text: '折疊率 0.000%，兩組全部檢查點都是 0' }],
    '變化最大的是腦的外緣一圈；折疊率為 0 主要來自微分同胚的積分（int_steps=7），這個版本本來就不容易折疊',
  ], { x: M, y: 5.9, w: 12.1, h: 1.05, fontSize: 12.5, paraSpaceAfter: 4 });
}

// 12 tigerbx 設計
{
  const s = base('TIGERBX · DESIGN', '同樣的資料換 tigerbx：怎麼比才公平',
    '要讓兩組只差「標籤來源」這一個變因，做了四件事。最重要的是第三件：atlas 那一端的標籤也要換成 tigerbx，不然光 atlas 就先扣掉 0.14，而那跟配準好不好無關。');
  const cards = [
    ['① 同一批人、同一個切分', `沿用 FreeSurfer 組的 train / test 切分，286 位逐一比對歸屬相同。否則兩組的 test 是不同的人，差異無法歸因。`],
    ['② 補做偏場校正', 'tigerbx 的影像是原始強度（白質變異係數 15.9%），FreeSurfer 已做過 nu 校正 → tigerbx 組補做 N4。'],
    ['③ atlas 的標籤也換成 tigerbx', '兩套方法在 atlas 上只有 0.855 一致。沿用 FreeSurfer 的 atlas 標籤，會先扣掉約 0.14。'],
    ['④ 同一人綁在同一邊', 'D015 / D037、D038 / DGM002 兩對，在兩組都放在同一邊。'],
  ];
  cards.forEach(([h, b], i) => {
    const x = M + (i % 2) * 6.15, y = 1.65 + Math.floor(i / 2) * 1.8, w = 5.95, hh = 1.6;
    card(s, x, y, w, hh);
    txt(s, h, { x: x + 0.3, y: y + 0.22, w: w - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.TEAL });
    txt(s, b, { x: x + 0.3, y: y + 0.72, w: w - 0.6, h: 1.05, fontSize: 13 });
  });
  txt(s, [
    { text: '仍然存在的差異：', options: { bold: true, color: C.RUST } },
    { text: '去顱骨範圍不同（FreeSurfer 腦遮罩是有標籤腦區的 1.53 倍，tigerbx 1.23 倍）；FreeSurfer 組多一次重採樣。' },
    { text: '所以比的是兩條完整流程，不是兩個分割演算法。', options: { bold: true } },
  ], { x: M, y: 5.4, w: 12.1, h: 0.85, fontSize: 13.5 });
}

// 13 tigerbx 結果
{
  const s = base('TIGERBX · RESULT', '同樣的資料換 tigerbx：結果（對照論文）',
    '兩組都是同一批 28 位 test，可以逐人配對比較。tigerbx 組的模型貢獻平均多 0.0097，是配對標準誤的 3.7 倍，28 位有 22 位比較大，統計上測得到但幅度很小。絕對值的差 0.072 裡，有 0.062 在訓練前就已經存在。表格最下面那列、也是左圖那條黑色水平線，是作者在論文 Table I 的 VoxelMorph (CC)。它只是一個數字，不是作者的訓練曲線；資料是另外 8 個資料集共 3,731 顆，atlas 也不同，而且論文主結果是非微分同胚版，所以只能參考。看模型貢獻，我們兩組都比論文低。我們折疊率是 0，主要是因為用了 repo 預設的微分同胚版（int_steps=7），形變場透過積分不容易折疊，不代表模型比論文好。');
  const a = curve('mix_exp1'), b = curve('tiger_exp1');
  lineChart(s, [
    { name: 'FreeSurfer 組', color: C.TEAL, labels: a.map((x) => String(x[0])), values: a.map((x) => x[1]) },
    { name: 'tigerbx 組', color: C.RUST, labels: b.map((x) => String(x[0])), values: b.map((x) => x[1]) },
    // 作者只有一個數字（論文 Table I 的 VoxelMorph (CC)），畫成水平參考線，不是作者的訓練曲線
    { name: '作者（論文 Table I）0.753', color: C.INK, labels: a.map((x) => String(x[0])), values: a.map(() => 0.753) },
  ], { x: M, y: 1.55, w: 6.7, h: 5.1, lineDataSymbol: 'none' });
  // 最下面一列是論文 Table I 的 VoxelMorph (CC)：另一批資料、另一顆 atlas，只能參考
  table(s, [
    ['', '基準線', '模型後', '貢獻', '折疊率'],
    ['FreeSurfer 組', f4(S.fs_base), f4(S.fs_after), hl(sg(S.fs_gain), C.TEAL), '0.000%'],
    ['tigerbx 組', f4(S.tg_base), f4(S.tg_after), hl(sg(S.tg_gain), C.RUST), '0.000%'],
    [bold('作者（論文）'), '0.584', '0.753', hl('+0.169', C.INK), '0.366%'],
  ], { x: 7.6, y: 1.7, w: 5.1, colW: [1.55, 0.85, 0.85, 0.95, 0.9], rowH: 0.46, fontSize: 12 });
  stat(s, sg(S.gain_diff), `兩組的貢獻差（配對標準誤 ${f4(S.gain_diff_sem)}，約 ${(S.gain_diff / S.gain_diff_sem).toFixed(1)} 倍）`, 7.6, 3.85, 5.1, C.INK);
  txt(s, [
    { text: `28 位中有 ${S.gain_diff_tg_better} 位是 tigerbx 組貢獻較大 —— 測得到，但幅度很小。\n`, options: {} },
    { text: '絕對值不能直接比：', options: { bold: true, color: C.RUST } },
    { text: `模型後差 ${f3(S.tg_after - S.fs_after)}，其中 ${f3(S.tg_base - S.fs_base)} 在訓練前就存在（兩套標籤畫邊界的方式不同）。\n` },
    { text: '作者那列＝左圖黑線：', options: { bold: true } },
    { text: '論文 Table I 的 VoxelMorph (CC)，只有一個數字。另一批資料、另一顆 atlas、非微分同胚版，只能參考。看模型貢獻我們兩組都較低；我們折疊率為 0 主要來自微分同胚版，不代表模型較好。' },
  ], { x: 7.6, y: 5.1, w: 5.1, h: 1.7, fontSize: 12 });
}

// 14 分結構
{
  const s = base('TIGERBX · BY STRUCTURE', '分結構：30 個結構全部是 tigerbx 組較高',
    '差距最大的是脈絡叢，FreeSurfer 組只有 0.38，那本來就是 FreeSurfer 分割不穩的結構。其次是殼核、蒼白球，tigerbx 在 atlas 上把殼核切得比較大。差距最小的是白質。這些差異反映的主要是標籤性質，不代表配準變好。');
  const ps = D.per_struct.map((x) => ({ n: x.name, d: x.tg_after - x.fs_after, fs: x.fs_after, tg: x.tg_after }))
    .sort((p, q) => p.d - q.d);
  s.addChart(pres.charts.BAR, [{ name: 'tigerbx − FreeSurfer', labels: ps.map((x) => x.n), values: ps.map((x) => x.d) }], {
    x: M, y: 1.45, w: 7.6, h: 5.45, barDir: 'bar', barGapWidthPct: 35, chartColors: [C.RUST],
    valAxisMinVal: 0, valAxisMaxVal: 0.28, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: '0.00',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '+0.000', dataLabelFontFace: F.MONO, dataLabelFontSize: 8,
    catAxisLabelFontFace: F.SANS, catAxisLabelFontSize: 8.5, catAxisLabelColor: C.INK,
    valAxisLabelFontFace: F.MONO, valAxisLabelFontSize: 9, valAxisLabelColor: C.MUTED,
    valGridLine: { color: 'E4E4DE', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: false,
  });
  const top = ps.slice(-4).reverse(), low = ps.slice(0, 2);
  table(s, [
    ['結構', 'FS', 'tigerbx', '差'],
    ...top.map((x) => [x.n, f3(x.fs), f3(x.tg), hl(sg(x.d, 3), C.RUST)]),
    ...low.map((x) => [x.n, f3(x.fs), f3(x.tg), sg(x.d, 3)]),
  ], { x: 8.55, y: 1.6, w: 4.15, colW: [1.35, 0.95, 0.9, 0.95], rowH: 0.42, fontSize: 12 });
  bullets(s, [
    '差距大的多半是 FreeSurfer 本身不穩、或兩套方法邊界定義不同的結構',
    [{ text: '反映的是標籤性質，', options: { bold: true } }, { text: '不代表配準變好' }],
  ], { x: 8.55, y: 4.75, w: 4.15, h: 1.9, fontSize: 13 });
}

// ── tigerbx 視覺化：同一位受試者、兩組並排 ─────────────────────────
{
  const t = SUB('T053');
  pairSlide('TIGERBX · VISUAL', '同一位受試者、兩組各自訓練：T053 的結構輪廓',
    '同一位 T053，左邊是 FreeSurfer 組的模型與標籤，右邊是 tigerbx 組。兩個模型各自用自己的影像訓練，但配準結果在圖上幾乎看不出差別。Dice 的差距主要來自兩套標籤的邊界定義。',
    { head: 'FreeSurfer 組', sub: `Dice ${f3(t.fs_base)} → ${f3(t.fs_after)}`, color: C.TEAL, img: MX('T053', 'contours'), alt: 'T053 FreeSurfer 組結構輪廓' },
    { head: 'tigerbx 組', sub: `Dice ${f3(t.tg_base)} → ${f3(t.tg_after)}`, color: C.RUST, img: TG('T053', 'contours'), alt: 'T053 tigerbx 組結構輪廓' },
    '實線 = atlas、虛線 = 受試者；上排只做線性對位，下排加上 VoxelMorph。兩個模型各自訓練，配準結果在圖上幾乎看不出差別 —— Dice 的差距主要來自兩套標籤的邊界定義。');
}

{
  const t = SUB('VNT008');
  pairSlide('TIGERBX · VISUAL', `測試集中年紀最大的一位：VNT008（${Math.round(t.age)} 歲）`,
    'VNT 組是三包裡年紀最大、起點最低的。這位 42 歲，基準線在兩組都偏低，但模型後都拉回到跟其他人差不多的水準。圖上邊緣的紅綠色就是沒對上的部分，下排明顯變少。',
    { head: 'FreeSurfer 組', sub: `Dice ${f3(t.fs_base)} → ${f3(t.fs_after)}`, color: C.TEAL, img: MX('VNT008', 'labels'), alt: 'VNT008 FreeSurfer 組標籤重疊' },
    { head: 'tigerbx 組', sub: `Dice ${f3(t.tg_base)} → ${f3(t.tg_after)}`, color: C.RUST, img: TG('VNT008', 'labels'), alt: 'VNT008 tigerbx 組標籤重疊' },
    [
      { text: '紅 = 只有 atlas 有、綠 = 只有受試者有、黃 = 重疊。' },
      { text: `起點兩組都偏低（${f3(t.fs_base)} / ${f3(t.tg_base)}），模型後拉到 ${f3(t.fs_after)} / ${f3(t.tg_after)}；`, options: { bold: true } },
      { text: '邊緣的紅綠在下排明顯變少。' },
    ]);
}

// 15 A0131
{
  const s = base('CASE · A0131', '效果最差的一位：A0131（5 歲）',
    'A0131 在兩組都是最低的。追查後排除了三個假設。右側和左側要分開看：右側是標籤本身不確定，換成 tigerbx 標籤就恢復正常；左側兩套方法有共識，模型還是低，比較像配準困難。另外，不是「兒童比較差」，24 位未成年受試者的分割一致性跟成人一樣。');
  const img = MX('A0131', 'labels');
  const sz = pngSize(img);
  const iw = 6.2, ih = iw * sz.h / sz.w;
  s.addImage({ path: img, x: M, y: 1.5, w: iw, h: ih, altText: 'A0131 配準前後的標籤重疊圖' });
  txt(s, '紅 = 只有 atlas 有 · 綠 = 只有受試者有 · 黃 = 重疊。上排只做線性對位，下排加上 VoxelMorph（FreeSurfer 組）。',
    { x: M, y: 1.55 + ih, w: iw, h: 0.5, fontSize: 10.5, color: C.MUTED });
  const A = D.a0131;
  const r = (k) => [k, f3(A[k].fs), f3(A[k].tg), `${f3(A[k].fs_others)} / ${f3(A[k].tg_others)}`];
  table(s, [
    ['結構', 'FS 組', 'tigerbx 組', '其他 27 位（FS / tigerbx）'],
    r('右杏仁核'), r('左杏仁核'),
  ], { x: 7.1, y: 1.6, w: 5.6, colW: [1.2, 0.95, 1.2, 2.25], rowH: 0.45, fontSize: 12 });
  bullets(s, [
    [{ text: '排除三個假設：', options: { bold: true } }, { text: '體積切錯（兩方法體積比 0.98）、結構太小（右杏仁核是第 40 百分位）、影像品質（右顳葉訊雜比不比左側差）' }],
    [{ text: '右側 = 標籤不確定：', options: { bold: true, color: C.RUST } }, { text: '兩套分割的右杏仁核一致性 0.525，286 位中最低；換成 tigerbx 標籤後回到 0.870' }],
    [{ text: '左側 = 配準困難：', options: { bold: true, color: C.TEAL } }, { text: '兩方法有共識（0.751），模型卻只有 0.394' }],
    [{ text: '不是「兒童較差」：', options: { bold: true } }, { text: '24 位未成年的分割一致性與成人相當（0.83 vs 0.84）' }],
  ], { x: 7.1, y: 3.15, w: 5.6, h: 3.6, fontSize: 12.5 });
}

{
  const t = SUB('A0131');
  const second = (k) => D.subjects.map((x) => x[k]).sort((p, q) => p - q)[1];
  pairSlide('CASE · A0131', 'A0131：換成 tigerbx 標籤之後',
    '同一位 A0131，兩組並排。它在兩組都是最低的，但換成 tigerbx 標籤之後，跟第二低的差距從 0.04 縮到 0.002，不再是明顯的離群值。矢狀面下方那塊紅色在兩組都存在，是 atlas 有、這位受試者沒有的區域，所以不是標籤方法造成的。',
    { head: 'FreeSurfer 組', sub: `Dice ${f3(t.fs_base)} → ${f3(t.fs_after)}`, color: C.TEAL, img: MX('A0131', 'labels'), alt: 'A0131 FreeSurfer 組標籤重疊' },
    { head: 'tigerbx 組', sub: `Dice ${f3(t.tg_base)} → ${f3(t.tg_after)}`, color: C.RUST, img: TG('A0131', 'labels'), alt: 'A0131 tigerbx 組標籤重疊' },
    [
      { text: `A0131 在兩組都是最低，但跟第二低的差距從 ${f3(second('fs_after') - t.fs_after)} 縮到 ${f3(second('tg_after') - t.tg_after)}。` },
      { text: '矢狀面下方那塊紅色（atlas 有、受試者沒有）在兩組都存在，', options: { bold: true } },
      { text: '所以不是標籤方法造成的。' },
    ]);
}

// 16 頭殼沒切好
{
  const s = base('SKULL STRIPPING', '頭殼沒切好的影像效果如何',
    '這題沒有直接測試。唯一的旁證是兩條流程的去顱骨鬆緊差很多，但兩組都能正常訓練。不過這個比較也混了標籤差異，只能說在這個範圍內沒有讓配準失敗。');
  card(s, M, 1.65, 5.6, 4.15, C.DARK);
  txt(s, '沒有直接測試', { x: M + 0.35, y: 1.95, w: 5, h: 0.6, fontSize: 26, bold: true, color: C.WHITE });
  txt(s, '原本想用的反例（A043、A016_1）已從清單排除，而且它們的問題是分割品質不好，不是去顱骨沒做好。',
    { x: M + 0.35, y: 2.7, w: 4.9, h: 1.1, fontSize: 13.5, color: C.SOFT });
  label(s, '若要正式測', M + 0.35, 4.05, 4.9, C.TEAL_L);
  txt(s, '同一批影像，把腦遮罩刻意放寬或收緊（例如膨脹／侵蝕幾個 voxel），其他條件完全不變，看 Dice 怎麼變。',
    { x: M + 0.35, y: 4.4, w: 4.9, h: 1.3, fontSize: 13.5, color: C.SOFT });
  label(s, '唯一的旁證', 6.7, 1.75, 6, C.TEAL);
  s.addChart(pres.charts.BAR, [{ name: '腦遮罩 / 有標籤腦區', labels: ['FreeSurfer', 'tigerbx'], values: [1.53, 1.23] }], {
    x: 6.6, y: 2.05, w: 6.1, h: 2.55, barDir: 'bar', chartColors: [C.TEAL], barGapWidthPct: 60,
    valAxisMinVal: 1.0, valAxisMaxVal: 1.6, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: '0.0"×"',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00"×"', dataLabelFontFace: F.MONO, dataLabelFontSize: 12,
    catAxisLabelFontFace: F.SANS, catAxisLabelFontSize: 12, catAxisLabelColor: C.INK, valAxisLabelFontFace: F.MONO, valAxisLabelFontSize: 10,
    valAxisLabelColor: C.MUTED, valGridLine: { color: 'E4E4DE', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: false,
    showTitle: true, title: '腦遮罩是有標籤腦區的幾倍', titleFontSize: 11, titleFontFace: F.SANS, titleColor: C.MUTED,
  });
  bullets(s, [
    '兩條流程的去顱骨鬆緊差很多',
    `兩組都能正常訓練：模型貢獻 ${sg(S.fs_gain, 3)} / ${sg(S.tg_gain, 3)}，折疊率皆為 0`,
    [{ text: '只能說：', options: { bold: true } }, { text: '在這個範圍內，去顱骨鬆緊沒有讓配準失敗（但同時混了標籤差異）' }],
  ], { x: 6.7, y: 4.8, w: 6.0, h: 1.85, fontSize: 13 });
}

// 17 誠實邊界與下一步
{
  const s = pres.addSlide(); page += 1;
  s.background = { color: C.DARK };
  txt(s, 'LIMITATIONS & NEXT', { x: M, y: 0.42, w: 11, h: 0.28, fontFace: F.MONO, fontSize: 11, bold: true, color: C.DMUTED, charSpacing: 3 });
  txt(s, '誠實邊界與下一步', { x: M, y: 0.74, w: 12, h: 0.66, fontSize: 28, bold: true, color: C.WHITE });
  const col = (x, head, color, items) => {
    txt(s, head, { x, y: 1.7, w: 5.8, h: 0.35, fontFace: F.MONO, fontSize: 11.5, bold: true, color, charSpacing: 2 });
    bullets(s, items, { x, y: 2.15, w: 5.8, h: 4.6, fontSize: 16, color: C.SOFT, paraSpaceAfter: 18 });
  };
  col(M, '要寫進方法學的限制', C.RUST_L, [
    '沒有 validation set：epoch 依 test 曲線挑，數字偏樂觀（論文有切 250 顆 validation）',
    '「最佳 epoch」是雜訊：epoch ≥ 90 彼此分不出高下',
    'tigerbx 對照混了影像處理差異：比的是兩條完整流程',
    'A0131 的成因只是推論（三個假設都已排除）',
    '9 顆斜切掃描在兩組都多經歷一次內插',
  ]);
  col(6.85, '下一步與待確認', C.TEAL_L, [
    'λ = 0.5 對照：看形變放寬之後 Dice 會不會再上升',
    '訓練改 120 epochs，並切出 validation set',
    '請老師確認：DGM 2023-04-25 那四位，資料夾編號與 DICOM ID 差 1',
    '請老師確認 A014 的組別：DICOM ID 是 T094',
  ]);
  txt(s, String(page).padStart(2, '0'), { x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.28, fontFace: F.MONO, fontSize: 11, color: C.DMUTED, align: 'right' });
  s.addNotes('最後是限制與下一步。有兩件事需要老師確認：DGM 那天連續四位的編號錯位，以及 A014 的組別。');
}

later.forEach((f) => f());
console.log('頁碼', JSON.stringify(PG));
pres.writeFile({ fileName: OUT }).then((f) => console.log('wrote', f, '·', page, 'slides'));
