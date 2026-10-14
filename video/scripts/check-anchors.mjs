#!/usr/bin/env node
/**
 * EP01 锚点 id 校验：场景里 `at("scene", "line-id")` 引用的旁白行是否真的存在。
 *
 * 为什么必须单独校验这一项：
 *   折句 / 改文案会让 **line id 重新编号**（`cold-open-11` 变 `cold-open-10` 等），
 *   而 `at()` 的参数是**字符串 id**——TypeScript 完全查不出来。
 *   本集实测踩过：重建旁白后 ColdOpenScene 仍引用 `cold-open-12`，
 *   typecheck 通过、`remotion still` 也能过（那帧没走到），
 *   但渲染整段时 `at()` 抛错 → 段落渲染直接失败。
 *
 * 用法：
 *   node video/scripts/check-anchors.mjs [ep01]
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
const ids = new Set(narr.lines.map((l) => l.id));
const byScene = new Map();
for (const l of narr.lines) {
  if (!byScene.has(l.sceneId)) byScene.set(l.sceneId, new Set());
  byScene.get(l.sceneId).add(l.id);
}

const dirs = [
  path.join(root, "src", "episodes", which, "scenes"),
  path.join(root, "src", "episodes", which, "components"),
  path.join(root, "src", "episodes", which),
];

let bad = 0;
let total = 0;
for (const dir of dirs) {
  if (!fs.existsSync(dir)) continue;
  for (const f of fs.readdirSync(dir)) {
    if (!f.endsWith(".tsx") && !f.endsWith(".ts")) continue;
    const file = path.join(dir, f);
    const src = fs.readFileSync(file, "utf8");
    const re = /\bat(?:Sec)?\(\s*"([^"]+)"\s*,\s*"([^"]+)"/g;
    let m;
    while ((m = re.exec(src))) {
      total += 1;
      const [, sceneId, lineId] = m;
      const line = m.index;
      const problems = [];
      if (!ids.has(lineId)) problems.push(`旁白里没有这一行`);
      else if (!byScene.get(sceneId)?.has(lineId)) problems.push(`该行不属于场景 ${sceneId}`);
      if (problems.length) {
        bad += 1;
        const near = [...(byScene.get(sceneId) ?? [])].filter((i) => i.startsWith(sceneId));
        const lineNo = lineId.split("-").pop();
        const hint = near.find((i) => i.endsWith(`-${lineNo}`));
        console.log(
          `  ❌ ${path.relative(root, file)}: at("${sceneId}", "${lineId}") — ${problems.join("；")}` +
            (hint ? `（该行号现在叫 ${hint}）` : ""),
        );
      }
    }
  }
}

const sceneFiles = fs.existsSync(dirs[0]) ? fs.readdirSync(dirs[0]).length : 0;
console.log(`检查 ${total} 个锚点引用（${sceneFiles} 个场景文件）`);
if (bad) {
  console.log(`\n❌ ${bad} 个锚点引用了不存在/不匹配的旁白行`);
  process.exit(1);
}
console.log("✅ 所有锚点 id 都存在且属于对应场景");
