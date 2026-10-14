/**
 * EP01 视觉契约：颜色语义 + 帧驱动动画工具。
 *
 * 规格约束（§7、§12.3）：
 *   - 全片固定色彩语义：人格=紫/玫红，事实=蓝/青，现场=绿/橙；
 *   - 所有动画由 useCurrentFrame() 驱动；
 *   - 秒数统一乘 fps，不散落裸帧常数；
 *   - 禁止 CSS transition / animation / Tailwind 动画类；
 *   - 常规进入用 {damping: 200} 的 spring，避免所有卡片弹跳；
 *   - 用 interpolate 时设置左右 clamp。
 */
import {interpolate, spring} from "remotion";

export const FPS = 30;

/** 秒 → 帧。禁止在组件里散落裸帧常数。 */
export const sec = (s: number) => Math.round(s * FPS);

export const COLORS = {
  // 底层（沿用现有样片的纸感深色底，保持系列一致）
  ink: "#14161a",
  inkSoft: "#1b1e23",
  panel: "#22262c",
  panelSoft: "#2b3037",
  line: "#3c434c",
  paper: "#f2ede4",
  muted: "#9aa1ab",
  dim: "#5f6874",

  // 三类内容的语义色（规格 §7，全片不许另起一套）
  persona: "#c964d8", // 人格：她会怎样反应（紫/玫红）
  fact: "#4d9fe0", // 事实：她知道什么（蓝/青）
  scene: "#4cc38a", // 现场：此刻发生了什么（绿）

  // 强调与状态
  accent: "#e8734f", // 关键结论 / 当前焦点（沿用样片 coral 家族）
  warn: "#e0a63c", // 半确定 / 降级
  hit: "#5fd39a",
  miss: "#4a515b",
} as const;

export type Domain = "persona" | "fact" | "scene";

export const DOMAIN_COLOR: Record<Domain, string> = {
  persona: COLORS.persona,
  fact: COLORS.fact,
  scene: COLORS.scene,
};

export const DOMAIN_LABEL: Record<Domain, string> = {
  persona: "人格",
  fact: "事实",
  scene: "现场",
};

export const DOMAIN_MEANING: Record<Domain, string> = {
  persona: "她会怎样反应",
  fact: "她知道什么",
  scene: "此刻发生了什么",
};

export const METHOD_LABEL: Record<string, string> = {
  always: "常驻",
  keyword: "关键词 / 正则",
  vector: "向量召回",
  "vector+llm": "向量 + LLM",
  llm: "LLM 判断",
};

/** 常规进入动画：damping 200，不弹跳（规格 §12.3）。 */
export const enter = (frame: number, delayFrames = 0, durationInFrames = sec(0.5)) =>
  spring({
    frame: frame - delayFrames,
    fps: FPS,
    config: {damping: 200},
    durationInFrames,
  });

/** 左右都 clamp 的线性插值。 */
export const lin = (
  frame: number,
  range: [number, number],
  output: [number, number],
) => interpolate(frame, range, output, {extrapolateLeft: "clamp", extrapolateRight: "clamp"});

/**
 * 四段式淡入淡出（立绘/浮层的标准进出场）：淡入 → 保持 → 淡出。
 * 用于"出现一会儿就走"的元素，避免每个场景重复手写三段插值。
 */
export const fadeInOut = (
  frame: number,
  fadeInAt: number,
  holdUntil: number,
  fadeOutUntil: number,
) =>
  interpolate(frame, [fadeInAt, holdUntil, fadeOutUntil], [0, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** 场景首尾淡入淡出（帧驱动，不用 CSS）。 */
export const fadeWindow = (frame: number, durationInFrames: number, fadeFrames = sec(0.35)) =>
  Math.min(clamp01(frame / fadeFrames), clamp01((durationInFrames - frame) / fadeFrames));

/**
 * checking 状态的短脉冲（规格 §12.4：不要无限旋转动画，用帧驱动短脉冲）。
 * 返回 0~1，每 `periodFrames` 一循环。
 */
export const pulse = (frame: number, periodFrames = sec(0.9), phase = 0) => {
  const t = ((frame + phase) % periodFrames) / periodFrames;
  return 0.5 - 0.5 * Math.cos(t * Math.PI * 2);
};

export const FONT_SANS =
  '"Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", sans-serif';
export const FONT_MONO =
  '"Cascadia Mono", "JetBrains Mono", "Consolas", "Courier New", monospace';
