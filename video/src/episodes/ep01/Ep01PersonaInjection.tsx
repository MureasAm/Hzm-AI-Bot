/**
 * Ep01PersonaInjection —— 第一集主组件（规格 §3、§12）。
 *
 * 承担三件事：
 *   1. 按 ep01-narration.json 把十个 scene 串成时间轴（Series）；
 *   2. 字幕轨：内置逐句字幕，关键核心词做短高亮（§13.3）；
 *   3. 配音轨：可选开启，逐行播放 public/audio/ep01/<line-id>.wav（§13.1、§13.2）。
 *
 * 动画约束（§12.3）：
 *   - 所有动画由 useCurrentFrame() 驱动；
 *   - 每个 Sequence 都设置 premountFor；
 *   - 不使用 CSS transition / animation；
 *   - public 资源走 staticFile()；
 *   - 渲染过程不访问网络、不调模型、不读真实用户数据。
 */
import React from "react";
import {Audio} from "@remotion/media";
import {
  AbsoluteFill,
  Sequence,
  Series,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import {
  COLORS,
  FONT_MONO,
  FONT_SANS,
  enter,
  fadeWindow,
  lin,
  sec,
} from "./theme";
import {
  lineAtFrame,
  linesOfScene,
  msToFrames,
  narration,
  sceneFrames,
  type Scene,
} from "./ep01-fixtures";
import {ColdOpenScene} from "./scenes/ColdOpenScene";
import {SystemPromptScene} from "./scenes/SystemPromptScene";
import {BehaviorCoreScene} from "./scenes/BehaviorCoreScene";
import {FilesToMessagesScene} from "./scenes/FilesToMessagesScene";
import {ThreeDomainsScene} from "./scenes/ThreeDomainsScene";
import {FourGatesScene} from "./scenes/FourGatesScene";
import {ContentFormsScene} from "./scenes/ContentFormsScene";
import {FullTraceScene} from "./scenes/FullTraceScene";
import {ReplyOutputScene} from "./scenes/ReplyOutputScene";
import {ClosingScene} from "./scenes/ClosingScene";

/** 网格底纹（复用现有样片的视觉语言，保持系列一致）。 */
const GridBackground: React.FC = () => (
  <AbsoluteFill
    style={{
      background:
        "repeating-linear-gradient(90deg, rgba(255,255,255,0.030) 0 1px, transparent 1px 80px)," +
        "repeating-linear-gradient(0deg, rgba(255,255,255,0.024) 0 1px, transparent 1px 80px)",
    }}
  />
);

const renderVisual = (scene: Scene, frame: number, duration: number) => {
  switch (scene.visual) {
    case "cold-open":
      return <ColdOpenScene frame={frame} durationInFrames={duration} />;
    case "system-prompt":
      return <SystemPromptScene frame={frame} durationInFrames={duration} />;
    case "behavior-core":
      return <BehaviorCoreScene frame={frame} durationInFrames={duration} />;
    case "files-to-messages":
      return <FilesToMessagesScene frame={frame} durationInFrames={duration} />;
    case "three-domains":
      return <ThreeDomainsScene frame={frame} durationInFrames={duration} />;
    case "four-gates":
      return <FourGatesScene frame={frame} durationInFrames={duration} />;
    case "content-forms":
      return <ContentFormsScene frame={frame} durationInFrames={duration} />;
    case "full-trace":
      return <FullTraceScene frame={frame} durationInFrames={duration} />;
    case "reply-output":
      return <ReplyOutputScene frame={frame} durationInFrames={duration} />;
    case "closing":
      return <ClosingScene frame={frame} durationInFrames={duration} />;
    default:
      return null;
  }
};

/**
 * 场景外壳：章节头（kicker / title / subtitle）+ 内容 + 首尾淡入淡出。
 *
 * 字幕安全区：底部留 190px 给字幕轨，content 从 top 202 起（§13.3 字幕不得覆盖
 * messages 面板的关键内容 → 各场景自己再往下压）。
 */
const SceneShell: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const duration = sceneFrames(scene);
  const fade = fadeWindow(frame, duration);
  const titleP = enter(frame, sec(0.1));
  const subP = enter(frame, sec(0.3));
  const progress = lin(frame, [0, duration], [0, 1]);
  const sceneIndex = narration.scenes.findIndex((s) => s.id === scene.id) + 1;

  return (
    <AbsoluteFill style={{background: COLORS.ink, color: COLORS.paper, opacity: fade, overflow: "hidden"}}>
      <GridBackground />

      {/* 章节头 */}
      <div
        style={{
          position: "absolute",
          left: 76,
          right: 76,
          top: 58,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <div>
          <div
            style={{
              fontFamily: FONT_MONO,
              color: COLORS.accent,
              fontSize: 19,
              letterSpacing: 1.2,
              marginBottom: 12,
              opacity: titleP,
            }}
          >
            {scene.kicker}
          </div>
          <div
            style={{
              fontFamily: FONT_SANS,
              fontSize: scene.id === "closing" ? 48 : 58,
              fontWeight: 700,
              lineHeight: 1.08,
              opacity: titleP,
              transform: `translateY(${(1 - titleP) * 18}px)`,
            }}
          >
            {scene.title}
          </div>
        </div>
        <div
          style={{
            width: 520,
            textAlign: "right",
            fontFamily: FONT_SANS,
            color: COLORS.muted,
            fontSize: 20,
            lineHeight: 1.45,
            paddingTop: 36,
            opacity: subP,
          }}
        >
          {scene.subtitle}
        </div>
      </div>

      {/* 内容 */}
      {renderVisual(scene, frame, duration)}

      {/*
        章节进度：放在**画面最顶端的一条细线**，不再横穿标题。
        原来放在 top:148 —— 标题是 58px × lineHeight 1.08，从 y≈89 排到 y≈152，
        于是那条 148 的进度条正好压在标题字上（用户指出的问题）。
        改用贴顶细线：不占任何内容区，也不会和标题/副标题打架。
      */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 0,
          height: 3,
          background: "rgba(255,255,255,0.075)",
        }}
      >
        <div
          style={{
            width: `${progress * 100}%`,
            height: "100%",
            background: COLORS.accent,
          }}
        />
      </div>

      {/* 集数与章节序号 */}
      <div
        style={{
          position: "absolute",
          right: 76,
          top: 1000,
          fontFamily: FONT_MONO,
          fontSize: 14,
          color: COLORS.dim,
          letterSpacing: 1,
        }}
      >
        EP01 · {String(sceneIndex).padStart(2, "0")} / {String(narration.scenes.length).padStart(2, "0")}
      </div>
    </AbsoluteFill>
  );
};

/**
 * 字幕轨（§13.3）：
 *   - 最多两行：文案在标点处折成两行（与 export-captions.mjs 的 SRT 一致）；
 *   - 每行尽量不超过 22 个中文字符；
 *   - 只对核心词做短高亮，不逐字高亮整段技术旁白；
 *   - 字幕盒固定在底部安全区，不与场景内容抢位置。
 */
const SUBTITLE_MAX_CHARS = 22;

/** 在标点处把一行文案折成最多两行。 */
const wrapCaption = (text: string): string[] => {
  if (text.length <= SUBTITLE_MAX_CHARS) {
    return [text];
  }
  const punct = ["，", "。", "、", "；", "：", "？", "！", "…", " "];
  const mid = text.length / 2;
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
    const cut = Math.floor(mid);
    return [text.slice(0, cut), text.slice(cut)];
  }
  return [text.slice(0, bestIdx + 1), text.slice(bestIdx + 1)];
};

const CaptionTrack: React.FC = () => {
  const frame = useCurrentFrame();
  const line = lineAtFrame(frame);
  if (!line) {
    return null;
  }

  const startFrame = msToFrames(line.startMs);
  const endFrame = msToFrames(line.endMs);
  const local = frame - startFrame;
  const duration = endFrame - startFrame;
  const reveal = Math.min(1, Math.max(0, local / 7));
  const exit = Math.min(1, Math.max(0, (duration - local) / 7));
  const visible = Math.min(reveal, exit);

  const highlights = line.subtitleHighlight ?? [];
  const rows = wrapCaption(line.text);
  // 只在第一行匹配高亮词；跨行的高亮词按行内匹配（拆开的词不强求高亮）
  const renderRow = (row: string, rowIndex: number): React.ReactNode => {
    if (!highlights.length) {
      return row;
    }
    const parts: React.ReactNode[] = [];
    let rest = row;
    let key = 0;
    while (rest.length) {
      let hitIdx = -1;
      let hitWord = "";
      for (const h of highlights) {
        const i = rest.indexOf(h);
        if (i >= 0 && (hitIdx < 0 || i < hitIdx)) {
          hitIdx = i;
          hitWord = h;
        }
      }
      if (hitIdx < 0) {
        parts.push(rest);
        break;
      }
      if (hitIdx > 0) {
        parts.push(rest.slice(0, hitIdx));
      }
      parts.push(
        <span key={`${rowIndex}-h${key++}`} style={{color: COLORS.accent}}>
          {hitWord}
        </span>,
      );
      rest = rest.slice(hitIdx + hitWord.length);
    }
    return parts;
  };

  return (
    <div
      style={{
        position: "absolute",
        left: 300,
        right: 300,
        bottom: 40,
        minHeight: 92,
        padding: "14px 26px",
        borderRadius: 8,
        background: "rgba(10,11,13,0.91)",
        border: `1px solid ${COLORS.line}`,
        color: COLORS.paper,
        opacity: visible,
        transform: `translateY(${(1 - reveal) * 10}px)`,
        boxShadow: "0 16px 46px rgba(0,0,0,0.34)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        gap: 2,
      }}
    >
      {rows.map((row, i) => (
        <div
          key={`${line.id}-row${i}`}
          style={{
            fontFamily: FONT_SANS,
            fontSize: rows.length > 1 ? 27 : 30,
            fontWeight: 600,
            lineHeight: 1.32,
            letterSpacing: 0,
            textAlign: "center",
          }}
        >
          {renderRow(row, i)}
        </div>
      ))}
    </div>
  );
};

/**
 * 配音轨。
 *
 * ⚠️ 默认走**整轨**（`build-voiceover-track.py` 生成的 `ep01-voiceover-track.wav`），
 *    不走"151 个 Sequence 各放一个 wav"。原因（本集实测）：
 *    151 路音频输入会让最终 ffmpeg 拼接阶段失败（1080p 渲染到最后
 *    `Conversion failed!`，不产出文件；0.5 倍预览也偶发同样错误）。
 *    整轨只有 1 路输入，且静音间隔由构建脚本按时间轴算，和画面帧严格对齐
 *    （实测整轨 608.32s vs 时间轴 608.32s，差 0ms）。
 *
 * `perLine` 保留为调试开关：想逐行核对某句话时打开它。
 */
const VoiceoverTrack: React.FC<{enabled: boolean; perLine?: boolean}> = ({
  enabled,
  perLine = false,
}) => {
  if (!enabled) {
    return null;
  }
  if (!perLine) {
    // 优先用 AAC（.m4a）：整轨 wav 是 37MB 的 32kHz PCM，Remotion 拼接阶段处理它更吃力。
    // 转成 160k AAC 后 8.8MB，时长与 wav 完全一致（608.320s）。
    return (
      <Sequence from={0} durationInFrames={msToFrames(narration.durationMs)} premountFor={narration.fps}>
        <Audio src={staticFile(`audio/${narration.episode}/${narration.episode}-voiceover-track.m4a`)} />
      </Sequence>
    );
  }
  return (
    <>
      {narration.lines.map((line) => (
        <Sequence
          key={line.id}
          from={msToFrames(line.startMs)}
          durationInFrames={Math.max(1, msToFrames(line.endMs - line.startMs))}
          premountFor={narration.fps}
        >
          <Audio src={staticFile(`audio/${narration.episode}/${line.id}.wav`)} />
        </Sequence>
      ))}
    </>
  );
};

export type Ep01Props = {
  /** 是否播放配音轨（旁白音频就绪后由 render-props 打开） */
  withVoiceover?: boolean;
  /** 是否显示字幕 */
  withSubtitles?: boolean;
  /** 配音轨是否按行播放（调试用；默认整轨） */
  voicePerLine?: boolean;
};

export const Ep01PersonaInjection: React.FC<Ep01Props> = ({
  withVoiceover = false,
  withSubtitles = true,
  voicePerLine = false,
}) => {
  return (
    <AbsoluteFill style={{background: COLORS.ink}}>
      <Series>
        {narration.scenes.map((scene) => (
          <Series.Sequence
            key={scene.id}
            durationInFrames={sceneFrames(scene)}
            premountFor={narration.fps}
          >
            <SceneShell scene={scene} />
          </Series.Sequence>
        ))}
      </Series>
      <VoiceoverTrack enabled={withVoiceover} perLine={voicePerLine} />
      {withSubtitles ? <CaptionTrack /> : null}
    </AbsoluteFill>
  );
};

export {linesOfScene};
