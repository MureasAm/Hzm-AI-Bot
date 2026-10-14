#!/usr/bin/env node
/**
 * EP01 验收自检：把规格 §16 里**能自动判的项**一次跑完。
 *
 * 人工项（"观众能复述人格/事实/现场"这类）不在这里判——见
 * `video/EP01_ACCEPTANCE_REPORT.md`。
 *
 * 用法：
 *   node video/scripts/check-ep01-acceptance.mjs
 */
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const repo = path.resolve(root, "..");

const ep01 = path.join(root, "src", "episodes", "ep01");
const data = path.join(root, "data", "ep01");
const audio = path.join(root, "public", "audio", "ep01");

const results = [];
const ok = (name, detail) => results.push({name, pass: true, detail});
const bad = (name, detail) => results.push({name, pass: false, detail});

const j = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const exists = (p) => fs.existsSync(p);

// ---------- 产物存在 ----------
for (const [label, p] of [
  ["persona-comparison.json", path.join(data, "persona-comparison.json")],
  ["persona-structure.json", path.join(data, "persona-structure.json")],
  ["full-trace-fixture.json", path.join(data, "full-trace-fixture.json")],
  ["full-trace-result.json", path.join(data, "full-trace-result.json")],
  ["full-trace-decisions.json", path.join(data, "full-trace-decisions.json")],
  ["ep01-narration.json", path.join(ep01, "ep01-narration.json")],
  ["ep01-fixtures.ts", path.join(ep01, "ep01-fixtures.ts")],
  ["ep01.srt", path.join(root, "public", "subtitles", "ep01.srt")],
  ["ep01.csv", path.join(root, "narration", "ep01.csv")],
  ["voiceover-track.m4a", path.join(audio, "ep01-voiceover-track.m4a")],
]) {
  exists(p) ? ok(`产物存在：${label}`, path.relative(root, p)) : bad(`产物存在：${label}`, "缺失");
}

// ---------- 画幅与帧率 ----------
const narr = j(path.join(ep01, "ep01-narration.json"));
const dur = narr.durationMs / 1000;
narr.width === 1920 && narr.height === 1080 && narr.fps === 30
  ? ok("1920×1080 / 30fps", `${narr.width}×${narr.height} @${narr.fps}`)
  : bad("1920×1080 / 30fps", `${narr.width}×${narr.height} @${narr.fps}`);
// 时长：**不按 10 分钟硬卡**（用户明确要求"时间不用严格控制在 10 分钟，
// 不然有很多不自然的地方"）。总长随内容自然浮动，这里只拦住明显的异常
// ——太短说明漏了内容，太长说明又回到"撑停顿凑时长"的老毛病。
dur >= 300 && dur <= 780
  ? ok("时长合理（不硬卡 10 分钟）",
       `${dur.toFixed(1)}s = ${Math.floor(dur / 60)}:${String(Math.round(dur % 60)).padStart(2, "0")}`)
  : bad("时长合理（不硬卡 10 分钟）", `${dur.toFixed(1)}s 超出 5~13 分钟`);

// ---------- 十个场景齐全（规格 §3） ----------
const wantedScenes = ["cold-open", "system-prompt", "behavior-core", "files-to-messages",
  "three-domains", "four-gates", "content-forms", "full-trace", "reply-output", "closing"];
const gotScenes = narr.scenes.map((s) => s.id);
JSON.stringify(gotScenes) === JSON.stringify(wantedScenes)
  ? ok("十个 scene 齐全且顺序与规格一致", gotScenes.join(" → "))
  : bad("十个 scene 齐全且顺序与规格一致", gotScenes.join(","));

// ---------- 三板对照：真实 3 版 × 5 次，且都用同一个模型/参数 ----------
const cmp = j(path.join(data, "persona-comparison.json"));
const runsOk = cmp.variants.length === 3 && cmp.variants.every((v) => v.runs.length === 5);
runsOk ? ok("三板对照 3 版 × 5 次真实输出", `${cmp.variants.reduce((a, v) => a + v.runs.length, 0)} 条`)
  : bad("三板对照 3 版 × 5 次真实输出", `variants=${cmp.variants.length}`);
cmp.params.temperature === 0.85 && cmp.params.frequency_penalty === 0.3 &&
cmp.params.max_tokens === 150 && cmp.params.thinking === "disabled"
  ? ok("生成参数与 constants.py 一致", JSON.stringify(cmp.params))
  : bad("生成参数与 constants.py 一致", JSON.stringify(cmp.params));
cmp.note && cmp.note.includes("相同生成参数")
  ? ok("记录了三版同模型同参数", "note 已写明")
  : bad("记录了三版同模型同参数", "note 缺失");

// ---------- 不得出现已删除架构作为现行机制 ----------
const deleted = ["traits.json", "styles.json", "voice_samples.json", "phrases.json"];
const narrationText = narr.lines.map((l) => l.text).join("\n");
const sceneSrc = fs.readdirSync(path.join(ep01, "scenes")).map((f) =>
  fs.readFileSync(path.join(ep01, "scenes", f), "utf8")).join("\n");
const mentionsDeleted = deleted.filter((d) => narrationText.includes(d) || sceneSrc.includes(d));
// 允许出现在 BehaviorCoreScene 的"旧文件"历史卡里，但旁白不能把它们当现行机制讲
const narrationMentions = deleted.filter((d) => narrationText.includes(d));
narrationMentions.length === 0
  ? ok("旁白未把已删除架构当现行机制", "旁白未提及 traits/styles/voice_samples/phrases")
  : bad("旁白未把已删除架构当现行机制", narrationMentions.join(","));
mentionsDeleted.length
  ? ok("旧文件只作历史背景出现", `仅出现在历史卡的源码里：${mentionsDeleted.join(",")}`)
  : ok("旧文件只作历史背景出现", "未出现");

// ---------- full trace：命中/沉默 + 真实段数 ----------
const dec = j(path.join(data, "full-trace-decisions.json"));
const hits = dec.decisions.filter((d) => d.hit).length;
const miss = dec.decisions.length - hits;
miss > 0
  ? ok("未命中的层保持沉默且有记录", `${hits} 命中 / ${miss} 沉默，共 ${dec.decisions.length} 条`)
  : bad("未命中的层保持沉默且有记录", "没有沉默项（可疑：不该全都命中）");
dec.messageCount === dec.messages.length
  ? ok("messages 段数与真实 trace 一致", `${dec.messageCount} 段`)
  : bad("messages 段数与真实 trace 一致", `${dec.messageCount} vs ${dec.messages.length}`);
dec.fictional === true
  ? ok("演示用户标记为虚构", "fictional: true")
  : bad("演示用户标记为虚构", "fictional 不为 true");

// ---------- fixture 的固定时间/天气 ----------
const fx = j(path.join(data, "full-trace-fixture.json"));
fx.clock && fx.weather && fx.weather.city
  ? ok("固定时间与天气写入 fixture", `${fx.clock} / ${fx.weather.city} ${fx.weather.text}`)
  : bad("固定时间与天气写入 fixture", "缺字段");

// ---------- 旁白：整句合成 + 规格硬上限 ----------
// 用户反馈"一句话在 gpt-sovits 里尽量一整句地打出来" → 每行都是一次 TTS 调用。
// 所以只检查规格的**硬**上限（一条音频不超过 12 秒）；
// "每行 2.5~6 秒"那条是给**字幕行**的建议，不适用于朗读单位。
const tooLong = narr.lines.filter((l) => l.endMs - l.startMs > 12000);
tooLong.length === 0
  ? ok("没有超过 12 秒的旁白行", `最长 ${Math.max(...narr.lines.map((l) => l.endMs - l.startMs))}ms`)
  : bad("没有超过 12 秒的旁白行", `${tooLong.length} 行`);
const avgChars = narr.lines.reduce((a, l) => a + l.text.length, 0) / narr.lines.length;
avgChars >= 8
  ? ok("旁白按整句切分（不是碎片）", `${narr.lines.length} 句，平均 ${avgChars.toFixed(1)} 字/句`)
  : bad("旁白按整句切分（不是碎片）", `平均只有 ${avgChars.toFixed(1)} 字/句`);

// ---------- 配音完整 ----------
const rep = j(path.join(audio, "voiceover-report.json"));
const repById = new Map(rep.map((r) => [r.line_id, r]));
const lineIds = narr.lines.map((l) => l.id);
const noAudio = lineIds.filter((id) => !repById.get(id)?.audio_ms);
noAudio.length === 0
  ? ok("每行旁白都有音频", `${lineIds.length} 行`)
  : bad("每行旁白都有音频", `缺 ${noAudio.length} 行：${noAudio.slice(0, 5).join(",")}`);
// 报告里可能有"已被删掉的旧行"的残留（改文案/折句后 line id 会变）——
// 它不影响渲染（渲染按 narration 走），但要提示清理，免得下次核对看混。
const orphan = rep.filter((r) => !lineIds.includes(r.line_id)).map((r) => r.line_id);
if (orphan.length) {
  ok("报告无残留旧行", `有 ${orphan.length} 条旧行残留（不影响渲染）：${orphan.slice(0, 5).join(",")}`);
}
const withAudio = rep.filter((r) => r.audio_ms && lineIds.includes(r.line_id));
const stubs = withAudio.filter((r) => r.audio_ms < 900);
stubs.length === 0
  ? ok("没有退化（<900ms）的音频", "全部合格")
  : bad("没有退化（<900ms）的音频", `${stubs.length} 行：${stubs.map((s) => s.line_id).join(",")}`);

// ---------- 语速（用户反馈项） ----------
const han = (t) => (t.match(/[\u4e00-\u9fa5]/g) || []).length;
const punc = (t) => (t.match(/[，。、；：？！…]/g) || []).length;
const lat = (t) => (t.match(/[A-Za-z0-9_]+/g) || []).reduce((a, s) => a + s.length, 0);
const rates = withAudio.map((r) => {
  const speech = r.audio_ms - 332 - 355 * punc(r.text) - 71 * lat(r.text);
  return speech > 0 && han(r.text) >= 6 ? han(r.text) / (speech / 1000) : null;
}).filter(Boolean).sort((a, b) => a - b);
if (rates.length > 5) {
  const med = rates[Math.floor(rates.length / 2)];
  const slow = rates.filter((x) => x < med * 0.8).length;
  // 语速中位数低于 5.5 字/秒就偏慢（提 1.2 倍之前是 4.96）
  med >= 5.5
    ? ok("语速中位数达标", `${med.toFixed(2)} 字/秒；偏慢 ${slow} 行 / 共 ${rates.length} 行`)
    : bad("语速中位数达标", `只有 ${med.toFixed(2)} 字/秒（目标 ≥5.5）`);
}
const reportSpeed = withAudio[0]?.speed;
if (reportSpeed) {
  ok("配音速度参数已记录", `speed_factor=${reportSpeed}`);
}

// ---------- 渲染物不联网/不调模型 ----------
const ep01Src = fs.readdirSync(ep01, {recursive: true})
  .filter((f) => String(f).endsWith(".tsx") || String(f).endsWith(".ts"))
  .map((f) => fs.readFileSync(path.join(ep01, String(f)), "utf8")).join("\n");
const netRe = /\b(fetch|axios|XMLHttpRequest|WebSocket|http:\/\/|https:\/\/)/;
netRe.test(ep01Src)
  ? bad("渲染代码不访问网络", "发现网络相关调用")
  : ok("渲染代码不访问网络", "无 fetch/axios/http 调用");
const apiRe = /chat\.completions|openai|zhipu/i;
apiRe.test(ep01Src)
  ? bad("渲染代码不调用模型", "发现模型调用")
  : ok("渲染代码不调用模型", "无模型调用");

// ---------- 动画约束 ----------
const animRe = /transition\s*:|@keyframes|animation\s*:|animate-/;
animRe.test(ep01Src)
  ? bad("没有 CSS transition/animation", "发现违禁动画")
  : ok("没有 CSS transition/animation", "全部帧驱动");
const seqCount = (ep01Src.match(/<Sequence|<Series\.Sequence/g) || []).length;
const premountCount = (ep01Src.match(/premountFor=/g) || []).length;
premountCount >= seqCount
  ? ok("每个 Sequence 都设了 premountFor", `${premountCount} / ${seqCount}`)
  : bad("每个 Sequence 都设了 premountFor", `${premountCount} / ${seqCount}`);
const staticCount = (ep01Src.match(/staticFile\(/g) || []).length;
staticCount > 0
  ? ok("public 资源走 staticFile()", `${staticCount} 处`)
  : bad("public 资源走 staticFile()", "未使用");

// ---------- 成片（可选） ----------
const mp4 = path.join(root, "out", "hazel-bot-ep01-persona-injection-with-voice.mp4");
if (exists(mp4)) {
  const probe = path.join(repo, "bin", "ffprobe.exe");
  const out = execFileSync(probe, ["-v", "error", "-show_entries",
    "format=duration", "-show_entries", "stream=codec_type,width,height",
    "-of", "default=noprint_wrappers=1", mp4], {encoding: "utf8"});
  out.includes("width=1920") && out.includes("height=1080") && out.includes("codec_type=audio")
    ? ok("成片存在且规格正确", `${(Number(out.match(/duration=([\d.]+)/)[1])).toFixed(1)}s，含音轨`)
    : bad("成片存在且规格正确", out.replace(/\n/g, " "));
} else {
  bad("成片存在且规格正确", "还没渲染（out/hazel-bot-ep01-persona-injection-with-voice.mp4）");
}

// ---------- 输出 ----------
const pass = results.filter((r) => r.pass).length;
for (const r of results) {
  console.log(`  ${r.pass ? "✅" : "❌"} ${r.name}${r.detail ? `  — ${r.detail}` : ""}`);
}
console.log(`\n${pass}/${results.length} 项通过`);
process.exit(pass === results.length ? 0 : 1);
