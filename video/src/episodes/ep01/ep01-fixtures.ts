/**
 * EP01 数据桥：把 Phase 0 冻结的证据读进来，供画面只读消费。
 *
 * 规格 §12.2：Remotion 组件不得硬编码实验回复和 trace 结果；统一读取
 *   video/data/ep01/persona-comparison.json
 *   video/data/ep01/full-trace-fixture.json
 *   video/data/ep01/full-trace-result.json
 *   video/src/episodes/ep01/ep01-narration.json
 * 且渲染过程不调用模型、不读网络、不碰真实用户数据。
 */
import personaComparisonRaw from "../../../data/ep01/persona-comparison.json";
import decisionsRaw from "../../../data/ep01/full-trace-decisions.json";
import fixtureRaw from "../../../data/ep01/full-trace-fixture.json";
import traceRaw from "../../../data/ep01/full-trace-result.json";
import personaStructureRaw from "../../../data/ep01/persona-structure.json";
import narrationRaw from "./ep01-narration.json";
import {type Domain} from "./theme";

export type ComparisonRun = {
  run: number;
  raw: string;
  cleaned: string;
  finish_reason: string | null;
};

export type ComparisonVariant = {
  id: string;
  label: string;
  sublabel: string;
  runs: ComparisonRun[];
  selectedRun: number;
  selectionReason: string;
};

export type PersonaComparison = {
  generated_at: string;
  commit: string;
  model: string;
  params: {temperature: number; frequency_penalty: number; max_tokens: number; thinking: string};
  input: string;
  behaviorSource: {
    file: string;
    name: string;
    trigger: string;
    response: string;
    sampleCount: number;
    formatted: string;
  };
  systemPromptSource: {file: string; chars: number};
  note: string;
  variants: ComparisonVariant[];
};

export type Decision = {
  id: string;
  label: string;
  domain: Domain;
  method: string;
  hit: boolean;
  reason: string;
  evidence: Record<string, unknown>;
  why: string;
};

export type MessageBlock = {
  index: number;
  role: string;
  header: string;
  chars: number;
  head: string;
};

export type TraceDecisions = {
  fixtureId: string;
  fictional: boolean;
  derivedFrom: string;
  generated_at: string;
  commit: string;
  model: string;
  clock: string;
  weather: {city: string; text: string; temperatureC: number};
  userProfile: {name: string; city: string; note: string};
  incomingBatch: string[];
  decisions: Decision[];
  messages: MessageBlock[];
  messageCount: number;
  rawReply: string;
  cleanedReply: string;
  delivery: {mode: string; shouldVoice: boolean; parts: string[]; note: string};
};

export type Fixture = {
  fixtureId: string;
  fictional: boolean;
  clock: string;
  clockDisplay: string;
  weather: {city: string; text: string; temperatureC: number};
  userProfile: {name: string; city: string; note: string};
  session: {topic: string; mood: string | null};
  scheduleNote: string;
  recentTurns: {t: string; user: string; bot: string}[];
  incomingBatch: string[];
  incomingBatchNote: string;
  expectedPaths: {shouldHit: string[]; shouldNotHit: string[]};
};

export type NarrationLine = {
  id: string;
  sceneId: string;
  startMs: number;
  endMs: number;
  text: string;
  subtitleHighlight?: string[];
};

export type Scene = {
  id: string;
  startMs: number;
  endMs: number;
  kicker: string;
  title: string;
  subtitle: string;
  visual: string;
};

export type Narration = {
  title: string;
  episode: string;
  fps: number;
  width: number;
  height: number;
  durationMs: number;
  narrationVoice: string;
  scenes: Scene[];
  lines: NarrationLine[];
};

export type PersonaStructure = {
  systemPrompt: {
    file: string;
    title: string;
    chars: number;
    headings: string[];
    sections: {name: string; chars: number}[];
    role: string;
  };
  behaviors: {
    file: string;
    count: number;
    names: string[];
    focus: {
      name: string;
      trigger: string;
      response: string;
      samples: {user: string; reply: string}[];
    };
  };
  historyNote: string;
};

export const personaComparison = personaComparisonRaw as unknown as PersonaComparison;
export const personaStructure = personaStructureRaw as unknown as PersonaStructure;
export const traceDecisions = decisionsRaw as unknown as TraceDecisions;
export const fixture = fixtureRaw as unknown as Fixture;
export const traceResult = traceRaw as unknown as TraceDecisions & {
  promptParts: Record<string, string>;
};
export const narration = narrationRaw as unknown as Narration;

export const msToFrames = (ms: number) => Math.round((ms / 1000) * narration.fps);
export const sceneFrames = (s: Scene) => msToFrames(s.endMs - s.startMs);

export const linesOfScene = (sceneId: string) =>
  narration.lines.filter((l) => l.sceneId === sceneId);

/** 某帧落在哪一行旁白上（字幕轨用）。 */
export const lineAtFrame = (frame: number): NarrationLine | undefined =>
  narration.lines.find((l) => frame >= msToFrames(l.startMs) && frame < msToFrames(l.endMs));

/**
 * 场景内的动画锚点：按**旁白行的起始时刻**取，而不是写死秒数。
 *
 * 为什么必须这样（本集踩过两次）：
 *   时间轴会被 `redistribute-narration.mjs` / `fit-narration-to-audio.mjs`
 *   重算（改文案、改语速、改停顿都会变），场景里若写死 `sec(6.4)`，
 *   重算后画面就会"讲 A 演 B"。改成按 line id 取时刻，画面永远跟着旁白走。
 *
 * @param sceneId 场景 id
 * @param lineId  旁白行 id（如 `"system-prompt-3"`）
 * @param offsetSec 相对该行起点的偏移（负数=提前，用于"准备动作先于开口"）
 * @returns 场景局部帧数
 */
export const at = (sceneId: string, lineId: string, offsetSec = 0): number => {
  const scene = narration.scenes.find((s) => s.id === sceneId);
  const line = narration.lines.find((l) => l.id === lineId);
  if (!scene || !line) {
    throw new Error(`at(): 找不到 ${sceneId} / ${lineId}`);
  }
  const localMs = line.startMs - scene.startMs + offsetSec * 1000;
  return Math.max(0, msToFrames(localMs));
};

/** 场景局部秒数（给视觉用，如遮罩进度）。 */
export const atSec = (sceneId: string, lineId: string, offsetSec = 0): number =>
  at(sceneId, lineId, offsetSec) / narration.fps;

export const variantById = (id: string) =>
  personaComparison.variants.find((v) => v.id === id)!;

/** 三板对照里"被选中"的那一条（数据里已写明选择理由）。 */
export const selectedReplyOf = (id: string) => {
  const v = variantById(id);
  return v.runs.find((r) => r.run === v.selectedRun)?.cleaned ?? "";
};

export const decisionsByDomain = (domain: Domain) =>
  traceDecisions.decisions.filter((d) => d.domain === domain);
