"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const source = fs.readFileSync(path.join(__dirname, "../tools/browser_cvat_human_signoff_draft_acceptance.cjs"), "utf8");

test("browser signoff acceptance is loopback-only and cleans resources", () => {
  assert.match(source, /server\.listen\(0, "127\.0\.0\.1"/);
  assert.match(source, /url\.hostname === "127\.0\.0\.1"/);
  assert.match(source, /await route\.abort\("blockedbyclient"\)/);
  assert.match(source, /if \(browser\) await browser\.close\(\)/);
  assert.match(source, /if \(server\) await closeServer\(server\)/);
  assert.match(source, /fs\.rmSync\(temp, \{recursive: true, force: true\}\)/);
});

test("browser signoff acceptance checks the draft transaction and authority gate", () => {
  assert.match(source, /invalid_binding_rejected_without_state_change: true/);
  assert.match(source, /official_export_blocked_before_completion: true/);
  assert.match(source, /assert\.equal\(official\.training_authorized, false\)/);
  assert.match(source, /assert\.equal\(official\.automatic_approval, false\)/);
  assert.match(source, /assert\.equal\(official\.approved_for_training, false\)/);
  assert.match(source, /row\.approved_for_training === false/);
});

test("browser signoff acceptance refuses to overwrite an output directory", () => {
  assert.match(source, /if \(fs\.existsSync\(output\)\) throw new Error\("output directory already exists"\)/);
  assert.match(source, /flag: "wx"/);
});

test("private evidence screenshot is opt-in", () => {
  assert.match(source, /if \(key === "--screenshot"\)/);
  assert.match(source, /if \(args\.screenshot\)/);
  assert.match(source, /screenshot_captured: Boolean\(args\.screenshot\)/);
});
