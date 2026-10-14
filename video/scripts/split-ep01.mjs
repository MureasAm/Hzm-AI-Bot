#!/usr/bin/env node
/**
 * EP01 分段导出：**一个场景一个 mp4**，方便逐段看细节、逐段提修改。
 *
 * 为什么需要它：整片 10 分钟，改一处要重渲全片。分段之后：
 *   - 每段只有 30~145 秒，渲得快、看得清；
 *   - 每段自带它那一段的配音（从整轨裁出来），声画仍对齐；
 *   - 文件名带序号与时长，改哪段直接说"03-behavior-core"。
 *
 * 实现：用 remotion 的 `--frames=<start>-<end>` 渲画面，再用 ffmpeg 从
 * `ep01-voiceover-track.m4a` 裁出**同一时间区间**的音轨合进去。
 * 这样分段音频与整片音频完全一致（同一份母版），不会出现"分段里语速不一样"。
 *
 * 用法：
 *   node video/scripts/split-ep01.mjs                    # 全部 10 段，0.5 倍，带配音
 *   node video/scripts/split-ep01.mjs --full             # 原始分辨率
 *   node video/scripts/split-ep01.mjs --only four-gates,closing
 *   node video/scripts/split-ep01.mjs --no-voice         # 只看版面（更快）
 *   node video/scripts/split-ep01.mjs --out-dir ep01-parts-v2
 */
import {execFileSync} from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const repo = path.resolve(root, "..");
const FFMPEG = path.join(repo, "bin", "ffmpeg.exe");

const argv = process.argv.slice(2);
const full = argv.includes("--full");
const noVoice = argv.includes("--no-voice");
const argVal = (name) => {
  const i = argv.findIndex((a) => a === name || a.startsWith(`${name}=`));
  if (i < 0) return null;
  return argv[i].includes("=") ? argv[i].split("=")[1] : argv[i + 1];
};
const onlyArg = argVal("--only");
const only = onlyArg ? new Set(onlyArg.split(",")) : null;
const outName = argVal("--out-dir") || "ep01-parts";

const narr = JSON.parse(
  fs.readFileSync(path.join(root, "src", "episodes", "ep01", "ep01-narration.json"), "utf8"),
);
const fps = narr.fps;
const track = path.join(root, "public", "audio", "ep01", "ep01-voiceover-track.m4a");
const outDir = path.join(root, "out", outName);
fs.mkdirSync(outDir, {recursive: true});

const msToFrames = (ms) => Math.round((ms / 1000) * fps);
const npx = process.platform === "win32" ? "npx.cmd" : "npx";
const total = narr.scenes.length;

let built = 0;
for (const [i, scene] of narr.scenes.entries()) {
  if (only && !only.has(scene.id)) continue;
  const idx = String(i + 1).padStart(2, "0");
  const from = msToFrames(scene.startMs);
  const to = msToFrames(scene.endMs);
  const durS = ((scene.endMs - scene.startMs) / 1000).toFixed(1);
  const base = `${idx}-${scene.id}`;
  const silent = path.join(outDir, `${base}.silent.mp4`);
  const final = path.join(outDir, `${base}.mp4`);

  if (fs.existsSync(final)) fs.unlinkSync(final);
  process.stdout.write(`[${idx}/${total}] ${scene.id.padEnd(18)} ${durS}s … `);

  // 1) 画面（用 render-props-ep01.json → 不带配音轨；分段音频单独裁，
  //    这样不会出现"整轨 + 段落裁切"两条音轨叠加）
  const renderArgs = [
    "remotion", "render", "src/index.ts", "HazelBotEp01PersonaInjection",
    path.relative(root, silent).replace(/\\/g, "/"),
    "--codec", "h264", "--concurrency", "2",
    "--x264-preset", "veryfast", "--crf", "20",
    `--frames=${from}-${to - 1}`,
    "--props=render-props-ep01.json",
  ];
  if (!full) renderArgs.push("--scale", "0.5");
  try {
    execFileSync(npx, renderArgs, {cwd: root, stdio: "pipe", shell: process.platform === "win32"});
  } catch (e) {
    console.log("画面FAIL");
    console.error(String(e.stderr || e.message).slice(-900));
    continue;
  }

  if (noVoice || !fs.existsSync(track)) {
    fs.renameSync(silent, final);
    console.log("ok（无配音）");
    built++;
    continue;
  }

  // 2) 从整轨裁出同一区间，再合进分段
  try {
    execFileSync(FFMPEG, [
      "-y", "-loglevel", "error",
      "-i", silent,
      "-ss", (scene.startMs / 1000).toFixed(3),
      "-t", ((scene.endMs - scene.startMs) / 1000).toFixed(3),
      "-i", track,
      "-map", "0:v:0", "-map", "1:a:0",
      "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
      "-shortest", final,
    ], {stdio: "pipe"});
    fs.unlinkSync(silent);
    const mb = (fs.statSync(final).size / 1024 / 1024).toFixed(1);
    console.log(`ok  ${mb} MB`);
    built++;
  } catch (e) {
    console.log("合音FAIL");
    console.error(String(e.stderr || e.message).slice(-900));
  }
}

console.log(`\n生成 ${built} 段 → ${path.relative(root, outDir)}`);
console.log(`整片：out/hazel-bot-ep01-persona-injection-with-voice.mp4`);
