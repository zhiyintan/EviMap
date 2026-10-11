// Check the catalogue's promise: every document listed under a topic shows at
// least one highlighted phrase of that topic, rendered by the real app.js
// buildHighlightedHtml. Works on a run's dashboard_ui/data (whole files) or on
// a sharded site/demo/<dataset>/data directory.
//   node scripts/check_highlight_promise.js <data_dir> [--max-unlocated 0.05] [--sample N]
// Exit 1 on any defect (listing without a phrase in the document, or a phrase
// swallowed by an overlapping span) or when the unlocated share exceeds the bound.
const fs = require("fs"); const path = require("path");
const args = process.argv.slice(2);
const D = args[0];
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const maxUnlocated = parseFloat(opt("--max-unlocated", "0.05"));
const sample = parseInt(opt("--sample", "0"), 10);
const ROOT = path.resolve(__dirname, "..");
const src = fs.readFileSync(path.join(ROOT, "experiments/dashboard_ui_template/app.js"), "utf8");
const grab = (n) => { const m = src.match(new RegExp("function " + n + "\\([^)]*\\) \\{[\\s\\S]*?\\n\\}\\n")); if (!m) throw new Error("app.js: " + n); return m[0]; };
eval(grab("escapeHtml") + grab("buildHighlightedHtml"));
const readJsonl = (f) => fs.readFileSync(f, "utf8").split("\n").filter(Boolean).map(JSON.parse);

const nodes = [];
for (const f of fs.readdirSync(D).filter(x => /^nodes(\.part\d+)?\.jsonl$/.test(x)).sort()) nodes.push(...readJsonl(path.join(D, f)));
const listed = new Map();
for (const n of nodes) {
  if (n.is_group_header || /^other/.test(n.basket_kind || "")) continue;
  for (const d of (n.support_doc_ids || [])) { if (!listed.has(d)) listed.set(d, new Set()); listed.get(d).add(n.atom_id); }
}
function* batches() {
  const idxPath = path.join(D, "doc_index.json");
  if (fs.existsSync(idxPath)) {
    const idx = JSON.parse(fs.readFileSync(idxPath, "utf8"));
    let buckets = [...new Set(Object.values(idx))].sort((a, b) => a - b);
    if (sample > 0) buckets = buckets.slice(0, sample);
    for (const b of buckets) {
      const tag = String(b).padStart(4, "0");
      const docs = new Map(readJsonl(path.join(D, "docbuckets", `doc_${tag}.jsonl`)).map(r => [r.doc_id, r.text]));
      yield [docs, readJsonl(path.join(D, "phrasebuckets", `ph_${tag}.jsonl`))];
    }
  } else {
    const docs = new Map(readJsonl(path.join(D, "documents.jsonl")).map(r => [r.doc_id, r.text]));
    yield [docs, readJsonl(path.join(D, "doc_all_phrases.jsonl"))];
  }
}
let listings = 0, missing = 0, noPhrase = 0, unlocated = 0, swallowed = 0;
const docsAffected = new Set(); const examples = [];
for (const [docs, rows] of batches()) {
  for (const r of rows) {
    const want = listed.get(r.doc_id); if (!want) continue;
    const text = docs.get(r.doc_id) || "";
    const phrases = r.phrases.map(o => ({ text: o.text, start: o.start, end: o.end, cssClass: "subject", jumpAtom: o.atom_id }));
    const { html } = buildHighlightedHtml(text, phrases);
    const rendered = new Set([...html.matchAll(/data-jump-atom="([^"]+)"/g)].map(m => m[1].replace(/&amp;/g, "&")));
    const have = r.phrases.map(o => o.atom_id);
    for (const a of want) {
      listings++;
      if (rendered.has(a)) continue;
      missing++; docsAffected.add(r.doc_id);
      const mine = r.phrases.filter(o => o.atom_id === a);
      let cause;
      if (!mine.length) cause = "no_phrase_in_doc";
      else if (mine.every(o => o.start == null)) cause = "phrase_unlocated";
      else cause = "swallowed_by_overlap";
      if (cause === "no_phrase_in_doc") noPhrase++; else if (cause === "phrase_unlocated") unlocated++; else swallowed++;
      if (examples.length < 5) examples.push({ doc: r.doc_id, topic: a, cause, phrases: mine.slice(0, 2).map(o => o.text) });
    }
  }
}
const rate = listings ? missing / listings : 0;
const report = { data_dir: D, listings, without_visible_highlight: missing, rate: (rate * 100).toFixed(2) + "%",
  cause: { no_phrase_in_doc: noPhrase, phrase_unlocated: unlocated, swallowed_by_overlap: swallowed },
  docs_affected: docsAffected.size, examples };
console.log(JSON.stringify(report, null, 1));
const defects = noPhrase + swallowed;
const unlocatedRate = listings ? unlocated / listings : 0;
if (defects > 0) { console.error(`[promise] FAIL: ${defects} listings cannot show their topic's phrase (pipeline/renderer defect)`); process.exit(1); }
if (unlocatedRate > maxUnlocated) { console.error(`[promise] FAIL: ${(unlocatedRate * 100).toFixed(2)}% listings have only unlocated phrases (bound ${maxUnlocated * 100}%)`); process.exit(1); }
console.error(`[promise] OK: ${listings} listings, ${missing} without visible highlight (${report.rate}, all unlocated phrases)`);
