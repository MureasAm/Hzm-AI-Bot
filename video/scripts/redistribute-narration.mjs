#!/usr/bin/env node
/**
 * EP01 时间轴初版求解（音频还没生成时用）：场景时长与行槽都由**朗读时长模型**算。
 *
 * 定位：这是**音频之前的预算**。音频生成完必须再跑 `fit-narration-to-audio.mjs`
 * 按实测时长钉准——那个才是最终时间轴。
 *
 * 用户反馈（"时间不用严格控制在 10 分钟，不然有很多不自然的地方"）后的原则：
 *   - **默认没有目标总时长**，总长随内容浮动；
 *   - 行槽 = 该句的预计朗读时长 + 小余量，**不设"最短槽"**（设了就会把短句撑出静音）；
 *   - 停顿用固定自然值，不为了凑时长拉伸。
 *   - 只有显式 `--total NN` 时才按章节比例分配富余（旧行为）。
 *
 * 用法：
 *   node video/scripts/redistribute-narration.mjs [ep01] [--write] [--total 600]
 */
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const argv = process.argv.slice(2);
const which = argv.find((a) => !a.startsWith("--")) || "ep01";
const write = argv.includes("--write");
const totalIdx = argv.findIndex((a) => a === "--total" || a.startsWith("--total="));
/** null = 不设目标总时长（默认，自然节奏） */
let TARGET_TOTAL_S = null;
if (totalIdx >= 0) {
  const a = argv[totalIdx];
  TARGET_TOTAL_S = Number(a.includes("=") ? a.split("=")[1] : argv[totalIdx + 1]);
}

const file =
  which === "ep01"
    ? path.join(root, "src", "episodes", "ep01", "ep01-narration.json")
    : path.join(root, "src", "narration.json");

const MIN_LINE = 0; // 不设最短槽：短句就是短，不撑静音
const MAX_LINE = 12000; // 单行最长槽（规格的硬上限是"一条音频不超过 12 秒"）
const MIN_GAP = 320; // 行间停顿（自然说话句间）
const TAIL_MIN = 380; // 场景尾部最小呼吸
const LEAD_IN = 120; // 场景起步

/**
 * 朗读时长模型（毫秒）——来自 123 行真实 HZM-SPEAK 合成结果的最小二乘拟合：
 *     audio_ms ≈ 332 + 200×汉字 + 355×标点 + 71×英文字母   （平均绝对误差 ≈ 610ms）
 *
 * 为什么不能只用"字数 / 语速"（本集实测踩过）：
 *   · 标点是真实停顿 —— "词是什么意思、她一贯的偏好、最深的经历、"只有 17 个汉字，
 *     却念了 9.3 秒；
 *   · 英文 token 很快 —— "system_prompt" 10 个字母只占约 0.7 秒，按字数算会把
 *     system-prompt-3 高估到 6.3 秒（实测 3.8 秒）。
 */
const estimateMs = (text) => {
  const han = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
  const punc = (text.match(/[，。、；：？！…]/g) || []).length;
  const lat = (text.match(/[A-Za-z0-9_]+/g) || []).reduce((a, s) => a + s.length, 0);
  return Math.round(332 + 200 * han + 355 * punc + 71 * lat);
};

/** 规格 §3 的章节相对比例（单位：秒，仅用于分配**富余**时间）。 */
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

const n = JSON.parse(fs.readFileSync(file, "utf8"));

// ---------- 1. 每场景的最小可行时长 ----------
// wantOf = 该句预计朗读时长 + 小余量，**不设最短槽**。
// 设最短槽会把"情景、反应、例句。"这种短句硬撑到 2.4s，多出来的是静音。
const wantOf = (text) => Math.min(MAX_LINE, estimateMs(text) + 250);
const need = new Map();
for (const scene of n.scenes) {
  const lines = n.lines.filter((l) => l.sceneId === scene.id);
  const slots = lines.reduce((a, l) => a + wantOf(l.text), 0);
  const gaps = Math.max(0, lines.length - 1) * MIN_GAP;
  need.set(scene.id, {
    min: LEAD_IN + slots + gaps + TAIL_MIN,
    lines,
    speech: lines.reduce((a, l) => a + estimateMs(l.text), 0),
  });
}

const needTotal = [...need.values()].reduce((a, v) => a + v.min, 0);
const targetMs = TARGET_TOTAL_S ? Math.round(TARGET_TOTAL_S * 1000) : needTotal;
const propTotal = n.scenes.reduce((a, s) => a + (SPEC_PROPORTION[s.id] ?? 30), 0);
const surplus = Math.max(0, targetMs - needTotal);

console.log(`台词预算 ${(needTotal / 1000).toFixed(1)}s`);
if (TARGET_TOTAL_S) {
  console.log(`目标总时长 ${TARGET_TOTAL_S}s → 可分配富余 ${(surplus / 1000).toFixed(1)}s（会摊进停顿）`);
} else {
  console.log("未指定 --total：总时长随内容自然浮动，停顿固定不拉伸");
}

// ---------- 2. 场景时长 = 最小需求 + 按规格比例分到的富余 ----------
let cursor = 0;
for (const scene of n.scenes) {
  const dur = Math.round(need.get(scene.id).min + surplus * ((SPEC_PROPORTION[scene.id] ?? 30) / propTotal));
  scene.startMs = cursor;
  scene.endMs = cursor + dur;
  cursor += dur;
}
n.durationMs = cursor;

// ---------- 3. 场景内分配行槽 ----------
// 槽位 = 该句预计朗读时长 + 余量。**不给最短槽**（短句就该短）。
// 只有显式 --total 时才把富余按需求比例摊上去。
for (const scene of n.scenes) {
  const {lines} = need.get(scene.id);
  if (!lines.length) continue;
  const sceneMs = scene.endMs - scene.startMs;
  const gaps = Math.max(0, lines.length - 1) * MIN_GAP;
  const avail = sceneMs - LEAD_IN - gaps - TAIL_MIN;
  const wants = lines.map((l) => wantOf(l.text));
  const wantTotal = wants.reduce((a, b) => a + b, 0) || 1;
  const surplusHere = Math.max(0, avail - wantTotal);

  let c = scene.startMs + LEAD_IN;
  lines.forEach((l, i) => {
    const want = wants[i];
    const slot = Math.min(MAX_LINE, Math.round(want + surplusHere * (want / wantTotal)));
    l.startMs = c;
    l.endMs = c + slot;
    c += slot + (i < lines.length - 1 ? MIN_GAP : 0);
  });
}

// ---------- 4. 报告 ----------
const lines = n.lines;
const over = lines.filter((l) => l.endMs - l.startMs > MAX_LINE);
const under = lines.filter((l) => l.endMs - l.startMs < MIN_LINE);
const mismatch = lines.filter((l) => estimateMs(l.text) > l.endMs - l.startMs + 80);
const overflowScenes = [];
for (const s of n.scenes) {
  const ls = lines.filter((l) => l.sceneId === s.id);
  const last = ls.length ? ls[ls.length - 1].endMs : s.startMs;
  if (last > s.endMs) overflowScenes.push(`${s.id}(+${last - s.endMs}ms)`);
}

console.log(`\n行 ${lines.length}  总时长 ${(n.durationMs / 1000).toFixed(1)}s`);
for (const s of n.scenes) {
  const ls = lines.filter((l) => l.sceneId === s.id);
  const sum = ls.reduce((a, l) => a + (l.endMs - l.startMs), 0);
  const est = ls.reduce((a, l) => a + estimateMs(l.text), 0);
  console.log(
    `  [${s.id.padEnd(18)}] ${((s.endMs - s.startMs) / 1000).toFixed(1)}s  行${String(ls.length).padStart(3)}  ` +
      `占${(sum / 1000).toFixed(1)}s  预计朗读${(est / 1000).toFixed(1)}s`,
  );
}
console.log(
  `\n最长行 ${Math.max(...lines.map((l) => l.endMs - l.startMs))}ms  ` +
    `最短行 ${Math.min(...lines.map((l) => l.endMs - l.startMs))}ms`,
);
let bad = 0;
if (over.length) { console.log(`❌ ${over.length} 行超 ${MAX_LINE}ms`); bad += over.length; }
if (under.length) { console.log(`❌ ${under.length} 行低于 ${MIN_LINE}ms`); bad += under.length; }
if (overflowScenes.length) { console.log(`❌ 场景内容溢出: ${overflowScenes.join(", ")}`); bad += overflowScenes.length; }
if (mismatch.length) {
  console.log(`⚠️ ${mismatch.length} 行的预计朗读时长超过槽位（会被下一行咬到）：`);
  for (const l of mismatch.slice(0, 10)) {
    console.log(`   ${l.id.padEnd(22)} 预计${estimateMs(l.text)}ms > 槽${l.endMs - l.startMs}ms  ${l.text}`);
  }
}
if (!bad) console.log("✅ 全部行与场景都在约束内");

if (write) {
  fs.writeFileSync(file, `${JSON.stringify(n, null, 2)}\n`, "utf8");
  console.log(`\n已写入 ${path.relative(root, file)}`);
} else {
  console.log("\n（未写入；加 --write 才落盘）");
}
