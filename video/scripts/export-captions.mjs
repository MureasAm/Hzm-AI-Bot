/**
 * 导出旁白 CSV 与字幕 SRT。
 *
 * 规格 §13.2：现有导出脚本需要增加 episode 参数，不能继续只读写根级
 * narration.json 和 public/subtitles.srt。
 *
 * 用法：
 *   node scripts/export-captions.mjs            # 旧样片（根级 narration.json）
 *   node scripts/export-captions.mjs ep01       # 第一集
 */
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const episode = process.argv[2] ?? null;

const conf = episode
  ? {
      narration: path.join(root, "src", "episodes", episode, `${episode}-narration.json`),
      csv: path.join(root, "narration", `${episode}.csv`),
      srt: path.join(root, "public", "subtitles", `${episode}.srt`),
    }
  : {
      narration: path.join(root, "src", "narration.json"),
      csv: path.join(root, "narration.csv"),
      srt: path.join(root, "public", "subtitles.srt"),
    };

const narration = JSON.parse(fs.readFileSync(conf.narration, "utf8"));

const pad = (value, size = 2) => String(value).padStart(size, "0");

const formatTime = (ms, separator) => {
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  const seconds = Math.floor((ms % 60000) / 1000);
  const millis = Math.floor(ms % 1000);
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}${separator}${pad(millis, 3)}`;
};

// ---- CSV：一行一条待合成语音 ----
const rows = narration.lines.map((line) =>
  [
    line.sceneId,
    line.id,
    line.startMs,
    line.endMs,
    line.endMs - line.startMs,
    line.text.replaceAll('"', '""'),
  ]
    .map((value, index) => (index >= 5 ? `"${value}"` : value))
    .join(","),
);

fs.mkdirSync(path.dirname(conf.csv), {recursive: true});
fs.writeFileSync(
  conf.csv,
  ["scene_id,line_id,start_ms,end_ms,duration_ms,text", ...rows].join("\n") + "\n",
  "utf8",
);

// ---- SRT：字幕 ----
// 规格 §13.3：最多两行、每行尽量不超过 22 个中文字符。
// 本集旁白单行都不超长，所以一条 line 就是一条字幕；超出则按标点折成两行。
const MAX_CHARS = 22;

const wrap = (text) => {
  if (text.length <= MAX_CHARS) {
    return text;
  }
  const punct = ["，", "。", "、", "；", "：", "？", "！", "…", " "];
  const mid = Math.floor(text.length / 2);
  let bestIdx = -1;
  let bestDist = Infinity;
  for (let i = 1; i < text.length - 1; i += 1) {
    if (!punct.includes(text[i])) continue;
    const d = Math.abs(i - mid);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }
  if (bestIdx < 0) {
    return `${text.slice(0, mid)}\n${text.slice(mid)}`;
  }
  return `${text.slice(0, bestIdx + 1)}\n${text.slice(bestIdx + 1)}`;
};

const overlong = narration.lines.filter((l) => l.text.length > MAX_CHARS);
if (overlong.length) {
  console.log(`⚠️ ${overlong.length} 行超过 ${MAX_CHARS} 字，SRT 里折成两行（画面字幕仍按单行排版）：`);
  for (const l of overlong.slice(0, 8)) {
    console.log(`   ${l.id} (${l.text.length}字)`);
  }
}

const srt = narration.lines
  .map(
    (line, index) =>
      `${index + 1}\n${formatTime(line.startMs, ",")} --> ${formatTime(
        line.endMs,
        ",",
      )}\n${wrap(line.text)}\n`,
  )
  .join("\n");

fs.mkdirSync(path.dirname(conf.srt), {recursive: true});
fs.writeFileSync(conf.srt, srt, "utf8");

const totalChars = narration.lines.reduce((a, l) => a + l.text.length, 0);
console.log(`来源: ${path.relative(root, conf.narration)}`);
console.log(`行数: ${narration.lines.length}  总字数: ${totalChars}`);
console.log(`总时长: ${(narration.durationMs / 1000).toFixed(1)}s`);
console.log(`Wrote ${path.relative(root, conf.csv)}`);
console.log(`Wrote ${path.relative(root, conf.srt)}`);
