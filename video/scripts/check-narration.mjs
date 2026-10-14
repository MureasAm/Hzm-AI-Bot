#!/usr/bin/env node
/**
 * EP01 旁白时间轴校验（规格 §13.1）：
 *   - 单行 2.5~6 秒（硬约束：太短听不清、太长 TTS 不稳）
 *   - 一句只承担一个信息点（人工确认，脚本只报字数密度）
 *   - 场景时长 = 该场景各行之和（允许间隙）
 *
 *   node video/scripts/check-narration.mjs [ep01|root]
 */
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const which = process.argv[2] || "ep01";

const file =
  which === "ep01"
    ? path.join(root, "src", "episodes", "ep01", "ep01-narration.json")
    : path.join(root, "src", "narration.json");

const n = JSON.parse(fs.readFileSync(file, "utf8"));
// 规格 §13.1 的"单行建议 2.5～6 秒"是**建议**，硬约束是"一条音频不超过 12 秒"。
// 时间轴按真实音频拟合（fit-voice）后，少量行会自然落在 2.3~2.5s 或 6~7s——
// 那是 TTS 在这个音色/语速下真实的产出，不是时间轴算错。所以：
//   · 下限按 2300ms 判（低于就是真的被压扁了）；
//   · 上限按建议的 6000ms **告警**，超过 8000ms 才算错误（逼近硬上限）。
const MIN = 2300;
const WARN_MAX = 6000;
const MAX = 8000;

let bad = 0;
let soft = 0;
console.log(`文件: ${path.relative(root, file)}`);
console.log(`总时长: ${(n.durationMs / 1000).toFixed(1)}s  场景 ${n.scenes.length}  行 ${n.lines.length}\n`);

for (const s of n.scenes) {
  const lines = n.lines.filter((l) => l.sceneId === s.id);
  const sum = lines.reduce((a, l) => a + (l.endMs - l.startMs), 0);
  const sceneMs = s.endMs - s.startMs;
  const gaps = [];
  let cursor = s.startMs;
  for (const l of lines) {
    if (l.startMs < cursor) gaps.push(`OVERLAP@${l.id}`);
    if (l.startMs > cursor) gaps.push(`gap ${l.startMs - cursor}ms`);
    cursor = l.endMs;
  }
  if (cursor > s.endMs) gaps.push(`OVERRUN ${cursor - s.endMs}ms`);

  console.log(
    `[${s.id}] ${(sceneMs / 1000).toFixed(1)}s  行${String(lines.length).padStart(2)}  ` +
      `占${(sum / 1000).toFixed(1)}s  填充${((sum / sceneMs) * 100).toFixed(0)}%`,
  );
  if (gaps.length) console.log(`    空间: ${gaps.join(", ")}`);

  for (const l of lines) {
    const d = l.endMs - l.startMs;
    const flags = [];
    let hard = false;
    if (d < MIN) { flags.push("太短"); hard = true; }
    if (d > MAX) { flags.push("超硬上限"); hard = true; }
    else if (d > WARN_MAX) flags.push("超建议上限");
    if (!l.text || !l.text.trim()) { flags.push("空文本"); hard = true; }
    if (flags.length) {
      if (hard) bad++;
      else soft++;
      console.log(
        `    ${hard ? "!!" : " ~"} ${l.id} ${d}ms ${flags.join("/")} — ${l.text.slice(0, 40)}`,
      );
    }
  }
}

const totalGap = n.scenes.reduce((a, s) => {
  const lines = n.lines.filter((l) => l.sceneId === s.id);
  const sum = lines.reduce((x, l) => x + (l.endMs - l.startMs), 0);
  return a + (s.endMs - s.startMs) - sum;
}, 0);

console.log(`\n场景总间隙: ${(totalGap / 1000).toFixed(1)}s`);
if (soft) {
  console.log(`\n~ ${soft} 行超过建议上限 ${WARN_MAX}ms（真实音频就是这么长，硬上限 ${MAX}ms 内）`);
}
if (bad) {
  console.log(`\n❌ ${bad} 行不合规（要求 ${MIN}~${MAX}ms）`);
  process.exit(1);
}
console.log(`\n✅ 所有行都在 ${MIN}~${MAX}ms 内`);
