#!/usr/bin/env node
/**
 * EP01 帧抽查：为每个场景渲染若干张 still，供人工核对布局
 * （规格 §15 Phase 5：抽查十个章节交界帧、检查字幕溢出、检查立绘是否遮挡信息）。
 *
 * 用法：
 *   node video/scripts/check-scenes.mjs            # 每场景 1 张（中点）
 *   node video/scripts/check-scenes.mjs --multi    # 每场景 3 张（1/4、1/2、3/4）
 */
import {execFileSync} from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const multi = process.argv.includes("--multi");

const n = JSON.parse(
  fs.readFileSync(path.join(root, "src", "episodes", "ep01", "ep01-narration.json"), "utf8"),
);
const outDir = path.join(root, "out", "ep01-checks");
fs.mkdirSync(outDir, {recursive: true});

const msToFrames = (ms) => Math.round((ms / 1000) * n.fps);

for (const s of n.scenes) {
  const from = msToFrames(s.startMs);
  const to = msToFrames(s.endMs);
  const span = to - from;
  const fracs = multi ? [0.12, 0.5, 0.86] : [0.5];
  for (const f of fracs) {
    const frame = Math.round(from + span * f);
    const name = `${s.id}-${String(frame).padStart(5, "0")}.png`;
    const out = path.join(outDir, name);
    process.stdout.write(`  ${s.id} frame=${frame} … `);
    try {
      execFileSync(
        process.platform === "win32" ? "npx.cmd" : "npx",
        [
          "remotion",
          "still",
          "src/index.ts",
          "HazelBotEp01PersonaInjection",
          path.relative(root, out),
          `--frame=${frame}`,
        ],
        {cwd: root, stdio: "pipe", shell: process.platform === "win32"},
      );
      console.log("ok");
    } catch (e) {
      console.log("FAIL");
      console.error(String(e.stderr || e.message).slice(-1200));
    }
  }
}
console.log(`\n输出目录: ${path.relative(root, outDir)}`);
