#!/usr/bin/env node
/**
 * 按**真实音频时长**顺延时间轴（规格 §3：允许用 fit-voice 流程顺延）。
 *
 * 与 redistribute-narration.mjs 的分工：
 *   redistribute  = 音频之前，按字数**预算**时间轴；
 *   fit-voice     = 音频之后，按**实测时长**把时间轴钉准（本脚本）。
 *
 * 做法：每条 line 的槽 = max(最短槽, 实测音频 + 余量)，行间保持固定停顿，
 * 场景时长 = 该场景所有行之和 + 尾部呼吸。逐场景累加得到新的总时长。
 *
 * 用法：
 *   node video/scripts/fit-narration-to-audio.mjs [ep01] [--write]
 */
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const args = process.argv.slice(2);
const which = args.find((a) => !a.startsWith("--")) || "ep01";
const write = args.includes("--write");

const narrationPath =
  which === "ep01"
    ? path.join(root, "src", "episodes", "ep01", "ep01-narration.json")
    : path.join(root, "src", "narration.json");
const reportPath =
  which === "ep01"
    ? path.join(root, "public", "audio", "ep01", "voiceover-report.json")
    : path.join(root, "public", "audio", "voiceover-report.json");

const MIN_LINE = 0;   // **不再设"单行最短槽"**：槽位就是真实音频长度。
                      // 设最短槽会把短句硬撑出静音——用户反馈"保留了很多无声的阶段"，
                      // 根因就是这里 + "把富余摊进停顿"那套凑时长逻辑。
const PAD = 560;      // 音频之后再留一点。整句合成后句子变长，句间只需要"换气"级别
                      // 的余量；560ms 让每句说得完整又不至于黏在一起。
const LEAD_IN = 300;  // 场景起步
const TAIL_MIN = 1400; // 场景尾部呼吸。换场景时观众要消化上一段，
                       // 这里给足（1.4s）比在**每句之间**塞静音自然得多——
                       // 前者像段落之间的停顿，后者像卡住。
const GAP_MIN = 380;  // 行间停顿（自然说话的句间停顿）
const GAP_MAX = 560;  // 行间停顿上限（只有显式 --total 时才用到）

/**
 * 规格 §3 的章节相对比例——**只在显式给 `--total` 时才用**，用来分配富余时间。
 *
 * 用户反馈（"时间不用严格控制在 10 分钟，不然有很多不自然的地方"）后：
 * **默认不再有目标总时长**，总长随内容自然浮动。
 * 之前"把富余摊进停顿"的做法正是"无声阶段"的来源——为了凑到 600s，
 * 603~700ms 的间隙被硬塞在每一句之间，听起来就像卡住了。
 */
const SPEC_PROPORTION = {
  "cold-open": 45,
  "system-prompt": 50,
  "behavior-core": 65,
  "files-to-messages": 35,
  "three-domains": 50,
  "four-gates": 80,
  "content-forms": 135,
  "full-trace": 90,
  "reply-output": 32,
  "closing": 23,
};

const TARGET_S = (() => {
  const a = args.find((x) => x.startsWith("--total"));
  if (!a) return null;
  return Number(a.includes("=") ? a.split("=")[1] : args[args.indexOf(a) + 1]);
})();

const narration = JSON.parse(fs.readFileSync(narrationPath, "utf8"));
const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
const byId = new Map(report.map((r) => [r.line_id, r]));

/**
 * 时间轴求解。
 *
 * **默认（不传 --total）：槽位 = 实测音频长度 + 小余量，停顿用固定自然值。**
 * 总时长随内容自然浮动。这是用户反馈后的默认行为——
 * "时间不用严格控制在 10 分钟，不然有很多不自然的地方"。
 *
 * 传 `--total NN`（秒）时才启用旧的"把富余摊进停顿"行为；
 * 那会把每句之间的静音拉到 GAP_MAX，**只在明确要求固定时长时用**。
 * 之前默认走的就是那套，正是"无声阶段"的来源：
 * 为了凑 600s，每句之间被塞进 600~700ms 静音。
 */
const allLines = narration.lines;

// ---- 1. 每行的槽位：就是音频长度（短句不再被撑长）----
const slotOf = new Map();
let contentMin = 0;
let missing = 0;
let measuredTotal = 0;
for (const line of allLines) {
  const m = byId.get(line.id);
  let slot;
  if (!m || !m.audio_ms) {
    missing += 1;
    // 没有音频：用**和生成器同一个时长模型**兜底，别用"字数/语速"那种粗估
    const han = (line.text.match(/[\u4e00-\u9fa5]/g) || []).length;
    const punc = (line.text.match(/[，。、；：？！…]/g) || []).length;
    const lat = (line.text.match(/[A-Za-z0-9_]+/g) || []).reduce((a, s) => a + s.length, 0);
    slot = Math.round(332 + 355 * punc + (200 * han + 71 * lat) / 1.2) + PAD;
  } else {
    measuredTotal += m.audio_ms;
    slot = m.audio_ms + PAD;
  }
  slotOf.set(line.id, slot);
  contentMin += slot;
}
const sceneCount = narration.scenes.length;
contentMin += LEAD_IN * sceneCount + TAIL_MIN * sceneCount
  + GAP_MIN * Math.max(0, allLines.length - sceneCount);

const targetMs = TARGET_S ? Math.round(TARGET_S * 1000) : null;
const surplus = targetMs ? Math.max(0, targetMs - contentMin) : 0;
const propTotal = narration.scenes.reduce((a, s) => a + (SPEC_PROPORTION[s.id] ?? 30), 0);

// ---- 2 & 3. 铺时间轴 ----
let cursor = 0;
const gapUsed = [];
for (const scene of narration.scenes) {
  const lines = allLines.filter((l) => l.sceneId === scene.id);
  scene.startMs = cursor;
  if (!lines.length) {
    scene.endMs = cursor + LEAD_IN + TAIL_MIN;
    cursor = scene.endMs;
    continue;
  }

  let gap = GAP_MIN;
  let tailExtra = 0;
  if (targetMs) {
    const sceneSurplus = surplus * ((SPEC_PROPORTION[scene.id] ?? 30) / propTotal);
    const nGaps = Math.max(1, lines.length - 1);
    const gapExtra = Math.min(sceneSurplus * 0.8, nGaps * (GAP_MAX - GAP_MIN));
    tailExtra = Math.max(0, sceneSurplus - gapExtra);
    gap = Math.round(GAP_MIN + gapExtra / nGaps);
  }
  gapUsed.push(gap);

  let c = scene.startMs + LEAD_IN;
  lines.forEach((l, i) => {
    const slot = slotOf.get(l.id);
    l.startMs = c;
    l.endMs = c + slot;
    c += slot + (i < lines.length - 1 ? gap : 0);
  });
  scene.endMs = Math.round(c + TAIL_MIN + tailExtra);
  cursor = scene.endMs;
}

narration.durationMs = cursor;

console.log(`来源: ${path.relative(root, narrationPath)}`);
console.log(`报告: ${path.relative(root, reportPath)}`);
console.log(`实测旁白总时长: ${(measuredTotal / 1000).toFixed(1)}s`);
console.log(`内容最短总长:   ${(contentMin / 1000).toFixed(1)}s`);
console.log(`目标总长:       ${(targetMs / 1000).toFixed(1)}s（富余 ${(surplus / 1000).toFixed(1)}s 已摊进停顿）`);
console.log(`行间停顿:       ${Math.min(...gapUsed)}~${Math.max(...gapUsed)}ms`);
console.log(`拟合后总时长:   ${(cursor / 1000).toFixed(1)}s`);
console.log(`\n各场景:`);
for (const scene of narration.scenes) {
  const lines = narration.lines.filter((l) => l.sceneId === scene.id);
  const sum = lines.reduce((a, l) => a + (l.endMs - l.startMs), 0);
  console.log(
    `  [${scene.id.padEnd(18)}] ${((scene.endMs - scene.startMs) / 1000).toFixed(1)}s  ` +
      `行${String(lines.length).padStart(3)}  占${(sum / 1000).toFixed(1)}s`,
  );
}
if (missing) {
  console.log(`\n⚠️ ${missing} 行没有音频，用了字数估算兜底`);
}
const bad = narration.lines.filter((l) => {
  const m = byId.get(l.id);
  if (!m || !m.audio_ms) return false;
  return m.audio_ms > l.endMs - l.startMs + 60;
});
if (bad.length) {
  console.log(`⚠️ ${bad.length} 行的音频比槽位长（会被下一行咬到）:`);
  for (const l of bad.slice(0, 8)) {
    const m = byId.get(l.id);
    console.log(`   ${l.id} 音频${m.audio_ms}ms > 槽${l.endMs - l.startMs}ms`);
  }
}

if (write) {
  fs.writeFileSync(narrationPath, `${JSON.stringify(narration, null, 2)}\n`, "utf8");
  console.log(`\n已写入 ${path.relative(root, narrationPath)}`);
} else {
  console.log("\n（未写入；加 --write 才落盘）");
}
