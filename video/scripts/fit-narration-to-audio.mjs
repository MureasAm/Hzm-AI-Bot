import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const narrationPath = path.join(root, "src", "narration.json");
const reportPath = path.join(root, "public", "audio", "voiceover-report.json");

const narration = JSON.parse(fs.readFileSync(narrationPath, "utf8"));
const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
const reportById = new Map(report.map((item) => [item.line_id, item]));

const lineGapMs = 260;
const sceneTailMs = 350;
const leadInMs = 120;

let timelineCursor = 0;

for (const scene of narration.scenes) {
  scene.startMs = timelineCursor;
  let lineCursor = scene.startMs + leadInMs;

  for (const line of narration.lines.filter((item) => item.sceneId === scene.id)) {
    const measured = reportById.get(line.id);
    const originalSlot = line.endMs - line.startMs;
    const slot = Math.max(
      originalSlot,
      measured ? measured.audio_ms + 300 : originalSlot,
    );

    line.startMs = Math.round(lineCursor);
    line.endMs = Math.round(line.startMs + slot);
    lineCursor = line.endMs + lineGapMs;
  }

  scene.endMs = Math.round(lineCursor - lineGapMs + sceneTailMs);
  timelineCursor = scene.endMs;
}

narration.durationMs = timelineCursor;

fs.writeFileSync(
  narrationPath,
  `${JSON.stringify(narration, null, 2)}\n`,
  "utf8",
);

console.log(
  `Fitted narration to audio: ${(timelineCursor / 1000).toFixed(2)}s total.`,
);
