#!/usr/bin/env node
/**
 * EP01 时间轴一致性校验：场景内部的动画锚点是否还跟旁白对齐。
 *
 * 为什么需要它：场景组件里的动画延迟是**按秒写死**的（如 `sec(6.4)` 对应
 * "system_prompt：常驻…" 那句的开播时刻）。旁白时间轴会被
 * redistribute-narration.mjs / fit-narration-to-audio.mjs 两次重算，
 * 一旦重算，这些写死的秒数就可能不再落在对应那句上——画面会"讲A演B"。
 *
 * 这个脚本把每个场景里写死的动画秒数抽出来，和旁白行的真实区间对照，
 * 报出"落在没有旁白的空档里"或"偏离目标句太远"的锚点。
 *
 * 用法：
 *   node video/scripts/check-scene-timing.mjs [ep01]
 */
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const which = process.argv[2] || "ep01";

const narr = JSON.parse(
  fs.readFileSync(path.join(root, "src", "episodes", which, `${which}-narration.json`), "utf8"),
);

const scenesDir = path.join(root, "src", "episodes", which, "scenes");
const fileForScene = {
  "cold-open": "ColdOpenScene.tsx",
  "system-prompt": "SystemPromptScene.tsx",
  "behavior-core": "BehaviorCoreScene.tsx",
  "files-to-messages": "FilesToMessagesScene.tsx",
  "three-domains": "ThreeDomainsScene.tsx",
  "four-gates": "FourGatesScene.tsx",
  "content-forms": "ContentFormsScene.tsx",
  "full-trace": "FullTraceScene.tsx",
  "reply-output": "ReplyOutputScene.tsx",
  closing: "ClosingScene.tsx",
};

let problems = 0;
for (const scene of narr.scenes) {
  const file = path.join(scenesDir, fileForScene[scene.id]);
  if (!fs.existsSync(file)) continue;
  const src = fs.readFileSync(file, "utf8");

  // 抽取 sec(6.4) 形态的锚点（这是场景里唯一允许写"秒"的地方）
  const anchors = [...src.matchAll(/\bsec\((\d+(?:\.\d+)?)\)/g)].map((m) => Number(m[1]));
  const uniq = [...new Set(anchors)].sort((a, b) => a - b);
  if (!uniq.length) continue;

  const lines = narr.lines
    .filter((l) => l.sceneId === scene.id)
    .map((l) => ({
      id: l.id,
      from: (l.startMs - scene.startMs) / 1000,
      to: (l.endMs - scene.startMs) / 1000,
      text: l.text,
    }));

  const sceneDur = (scene.endMs - scene.startMs) / 1000;
  const offAnchors = [];
  for (const a of uniq) {
    const hit = lines.find((l) => a >= l.from - 0.25 && a <= l.to);
    if (!hit) {
      // 是否落在行间空档？空档里的锚点通常是刻意的"留白"，但也可能是漂移
      const near = lines.find((l) => Math.abs(l.from - a) < 1.0 || Math.abs(l.to - a) < 1.0);
      if (!near) offAnchors.push({a, why: "没有邻近的旁白行"});
      else offAnchors.push({a, why: `行间空档，最近的是 ${near.id}`});
    }
  }

  const beyond = uniq.filter((a) => a > sceneDur);
  console.log(
    `[${scene.id.padEnd(18)}] ${sceneDur.toFixed(1)}s  锚点 ${uniq.length} 个  ` +
      `${offAnchors.length === 0 && beyond.length === 0 ? "✅ 全部落在旁白区间内" : ""}`,
  );
  if (offAnchors.length) {
    problems += offAnchors.length;
    for (const o of offAnchors) console.log(`    ~ sec(${o.a}) ${o.why}`);
  }
  if (beyond.length) {
    problems += beyond.length;
    for (const a of beyond) console.log(`    !! sec(${a}) 超出场景时长 ${sceneDur.toFixed(1)}s`);
  }
}

console.log(
  problems
    ? `\n~ ${problems} 个锚点不在旁白区间内（空档锚点多为刻意留白；超时长的必须改）`
    : "\n✅ 所有场景的动画锚点都与旁白对齐",
);
process.exit(problems && problems > 0 && process.argv.includes("--strict") ? 1 : 0);
