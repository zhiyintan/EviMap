"use strict";
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const filename = path.join(__dirname, "../experiments/dashboard_ui_template/app.js");
const source = fs.readFileSync(filename, "utf8");
const boundary = source.indexOf("(async function boot()");
assert.ok(boundary > 0, "dashboard boot boundary must exist");
const context = vm.createContext({ console });
vm.runInContext(source.slice(0, boundary), context, { filename });

const marked = context.buildHighlightedHtml("skill 😀 skill", [
  { text: "skill", start: 8, end: 13, cssClass: "subject" },
]);
assert.equal(marked.html, 'skill 😀 <mark class="subject">skill</mark>');
assert.equal(marked.fallback.length, 0);
assert.equal(context.findWordBoundedIndex("position sit university", "sit"), 9);

const text = "😀 Alpha. Beta gamma.";
const a = [{ start: 2, end: 7 }];
const b = [{ start: 9, end: 13 }];
assert.equal(context.findCooccurrences(a, b, text, 0).length, 0);
assert.equal(context.findCooccurrences(a, b, text, -1).length, 1);
assert.equal(context.findCooccurrences(a, b, text, 2).length, 0);
assert.equal(context.findCooccurrences(a, b, text, 20).length, 1);
assert.equal(context.findCooccurrences([{ start: null }], b, text, -1).length, 0);
console.log("dashboard evidence spans, word boundaries and topic windows passed");
