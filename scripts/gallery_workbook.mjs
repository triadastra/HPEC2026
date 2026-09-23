// Optional Excel export after results_gallery.py; requires @oai/artifact-tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'results/gallery');
const previews = path.join(root, 'outputs/gallery-workbook-checks');
await fs.mkdir(previews, { recursive: true });
const data = JSON.parse(await fs.readFile(path.join(out, 'tables/workbook_data.json'), 'utf8'));
const wb = Workbook.create();
const font = 'Helvetica Neue';
const definitions = [
  ['Results', 'results', 'All archived result summaries; normalized MSE/MAE. Population seed SD. Missing values remain blank.'],
  ['Loss arms', 'arms', '46 configurations with paired losses. Source and revived-checkpoint qualifications are explicit.'],
  ['Pairwise tests', 'pairs', '1,035 unique pairs. Holm_all uses 1,035 tests; Holm_paper uses 136. TOST is unadjusted, MSE-based.'],
  ['Archived costs', 'costs', 'All 225 profiling records, including configurations without matching accuracy records. Not fresh profiling.'],
  ['XGBoost seeds', 'xgboost', 'Original per-seed, per-channel aggregate and rolling metrics. Tree counts and measured FLOPs were not saved.'],
  ['Loss seed metrics', 'seed_metrics', 'Per-seed means extracted from the loss archive, retaining original stored floating-point precision.'],
  ['Training diagnostics', 'training', 'Archived diagnostic values, not newly reconstructed training histories.'],
  ['LR judgments', 'lr', 'Archived learning-rate hypotheses, not proof of optimal rates. Limited to the displayed 38 configurations.'],
];
const notes = wb.worksheets.add('Read me');
const notesRows = [
  ['HPEC 2026 supplementary results', ''],
  ['Scope', 'Archived results and supplementary statistical sensitivity analysis; no new model training or inference.'],
  ['Paper', 'Are Multidimensional Models Worth Their Computational Cost in Demand Forecasting?'],
  ['Repository', 'https://github.com/triadastra/HPEC2026'],
  ['Data', 'https://huggingface.co/datasets/Celsia/HPEC2026'],
  ['Model archive', 'https://huggingface.co/Celsia/HPEC2026'],
  ['Methods', 'results/gallery/README.md'],
  ['Original tables', 'results/SOURCES.json pins source revisions and checksums; original tables remain unchanged.'],
  ['Recovered inputs', 'results/gallery/inputs/SOURCES.json; source loss archive hash and checkpoint qualifications in loss_provenance.json.'],
  ['Loss coverage', '30,087 series x 24 months. Three-seed mean losses per trained arm; persistence is deterministic.'],
  ['Provenance limitation', 'Eight arms use revived checkpoints whose source-tree fingerprints were not revalidated.'],
  ['Statistical limitation', '24 months; HAC sensitivity does not resolve all dependence. Wilcoxon assumes independent series; related trade series may violate this.'],
  ['Equivalence', 'MSE TOST uses 5% of reference MSE, is direction-dependent, and is not multiple-testing adjusted.'],
  ['FLOPs unit', 'Arithmetic operations per joint value-and-weight forecast. Not watts, runtime, or total instruction count.'],
  ['XGBoost', 'At most 1,000 arithmetic additions and separately 4,000 tree comparisons. Analytical bound, not measured historical cost.'],
  ['Statistical baselines', 'Analytical model arithmetic: persistence/seasonal copy 0; direct 12-month mean 24 for two outputs. Memory work excluded.'],
  ['Rolling pooling', 'Metrics averaged across equal-length annual folds within seed, then across seeds. RMSE is sqrt(pooled MSE) within each seed.'],
  ['sMAPE and MASE', 'Archived evaluation conventions retained. These values were not redefined or independently recomputed by this gallery.'],
  ['Displayed precision', 'Numeric cells retain stored precision; display rounding does not imply inferential precision.'],
  ['Missing values', 'Blank means not recorded/not applicable, never a substituted zero. Analytical and archived costs use different columns.'],
];
notes.getRange(`A1:B${notesRows.length}`).values = notesRows;
notes.getRange(`A1:B${notesRows.length}`).format.font = { name: font, size: 11 };
notes.getRange('A1:B1').format.font = { name: font, size: 15, bold: true };
notes.getRange('A:A').format.columnWidth = 28;
notes.getRange('B:B').format.columnWidth = 110;
notes.getRange(`A2:B${notesRows.length}`).format.wrapText = true;
notes.getRange(`A2:B${notesRows.length}`).format.rowHeight = 34;
notes.showGridLines = false;

function col(n) { let s=''; for(n++;n;n=Math.floor((n-1)/26)) s=String.fromCharCode(65+(n-1)%26)+s; return s; }
function typed(v) {
  if (v === '' || v == null) return null;
  if (typeof v !== 'string') return v;
  if (/^-?\d+(\.\d*)?([eE][+-]?\d+)?$/.test(v)) return Number(v);
  if (v === 'True' || v === 'False') return v === 'True';
  return v;
}
for (const [name, key, note] of definitions) {
  const sheet = wb.worksheets.add(name);
  const rows = data[key];
  let fields = [...new Set(rows.flatMap(r => Object.keys(r)))];
  if (key === 'results') fields = ['label', 'panel', 'variant', 'enc', 'n', 'MSE', 'MSE_std', 'MAE', 'RMSE', 'sMAPE', 'MASE', ...fields.filter(k => !['label','panel','variant','enc','n','MSE','MSE_std','MAE','RMSE','sMAPE','MASE'].includes(k))];
  const end = col(fields.length-1), last = rows.length+5;
  sheet.getRange('A1').values = [[name]];
  sheet.getRange('A2').values = [[note]];
  sheet.getRange(`A5:${end}5`).values = [fields];
  sheet.getRange(`A6:${end}${last}`).values = rows.map(r => fields.map(f => typed(r[f])));
  sheet.getRange(`A1:${end}${last}`).format.font = { name: font, size: 10 };
  sheet.getRange('A1').format.font = { name: font, size: 15, bold: true };
  sheet.getRange(`A5:${end}5`).format = { fill: '#334E6C', font: { name: font, size: 10, bold: true, color: '#FFFFFF' }, wrapText: true, rowHeight: 42 };
  sheet.getRange(`A6:${end}${last}`).format.rowHeight = 24;
  fields.forEach((field, i) => {
    const numeric = rows.some(r => typeof typed(r[field]) === 'number');
    const c = col(i);
    const width = ['source','cost_source','provenance'].includes(field) ? 62 : ['run','arm','a','b','label'].includes(field) ? 44 : ['cost_status','variant','panel','enc','model','cell','cell_verdict'].includes(field) ? 29 : 19;
    sheet.getRange(`${c}5:${c}${last}`).format.columnWidth = width;
    if (!numeric) sheet.getRange(`${c}6:${c}${last}`).format.wrapText = true;
    if (numeric) sheet.getRange(`${c}6:${c}${last}`).setNumberFormat(/(^p_|_p_|_p$)/.test(field) ? '0.000E+00' : /(^n$|seed|params|epoch|trees|comparisons|_end$|test_year|n_test)/.test(field) ? '#,##0' : '0.000000');
  });
  rows.forEach((r,i) => {
    const maxLength = Math.max(...fields.map(f => typeof typed(r[f]) === 'string' ? String(r[f]).length : 0));
    if (maxLength > 40) sheet.getRange(`A${i+6}:${end}${i+6}`).format.rowHeight = maxLength > 95 ? 58 : 38;
  });
  sheet.tables.add(`A5:${end}${last}`, true, `Gallery${key.replaceAll('_','')}`);
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(key === 'pairs' ? 2 : 1);
  sheet.showGridLines = false;
  if (key === 'results') {
    const c=col(fields.length);
    sheet.getRange(`${c}5`).values=[['Recorded MFLOPs']];
    sheet.getRange(`${c}5`).format={fill:'#334E6C',font:{color:'#FFFFFF',bold:true},wrapText:true};
    sheet.getRange(`${c}5:${c}${last}`).format.columnWidth=20;
    const original=col(fields.indexOf('flops_per_forecast'));
    sheet.getRange(`${c}6:${c}${last}`).formulas=rows.map((r,i)=>[`=IF(ISNUMBER(${original}${i+6}),${original}${i+6}/1000000,"")`]);
    sheet.getRange(`${c}6:${c}${last}`).setNumberFormat('0.000000');
  }
}

console.log((await wb.inspect({kind:'table',range:'Results!A5:K9',include:'values,formulas',tableMaxRows:5,tableMaxCols:11,maxChars:2500})).ndjson);
const errors = await wb.inspect({kind:'match', searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options:{useRegex:true,maxResults:30}, summary:'Final formula error scan'});
console.log(errors.ndjson);
await fs.writeFile(path.join(previews,'formula-scan.json'),errors.ndjson);
for (const name of ['Read me',...definitions.map(x=>x[0])]) {
  const image=await wb.render({sheetName:name,range:name==='Read me'?'A1:B10':'A1:H11',scale:1.3,format:'png'});
  await fs.writeFile(path.join(previews,name.replaceAll(' ','_')+'.png'),new Uint8Array(await image.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(out,'hpec_results.xlsx'));
console.log('Workbook saved with nine sheets.');
