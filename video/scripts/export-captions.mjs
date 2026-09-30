import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const narration = JSON.parse(
  fs.readFileSync(path.join(root, "src", "narration.json"), "utf8"),
);

const pad = (value, size = 2) => String(value).padStart(size, "0");

const formatTime = (ms, separator) => {
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  const seconds = Math.floor((ms % 60000) / 1000);
  const millis = Math.floor(ms % 1000);
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}${separator}${pad(millis, 3)}`;
};

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

fs.writeFileSync(
  path.join(root, "narration.csv"),
  [
    "scene_id,line_id,start_ms,end_ms,duration_ms,text",
    ...rows,
  ].join("\n") + "\n",
  "utf8",
);

const srt = narration.lines
  .map(
    (line, index) =>
      `${index + 1}\n${formatTime(line.startMs, ",")} --> ${formatTime(
        line.endMs,
        ",",
      )}\n${line.text}\n`,
  )
  .join("\n");

fs.writeFileSync(path.join(root, "public", "subtitles.srt"), srt, "utf8");
console.log("Wrote narration.csv and public/subtitles.srt");
