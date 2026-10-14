/**
 * BehaviorAnatomy —— 一条真实 behavior 的解剖（规格 §5.2、§12.4）。
 *
 * 固定三个术语：情景 → 反应 → 例句
 *   - 三区域依次展开，用**一根动作路径**连接（不是树状代码图）；
 *   - 例句以**聊天气泡**出现，**不以 JSON 代码块出现**；
 *   - 不把整个 JSON 一次铺满。
 *
 * 版式（用户反馈过：例句超出底部、立绘大半在界外）：
 *   左栏 = 情景 / 反应 两张卡 + 一根竖向动作路径；
 *   右栏 = 例句（5 条聊天气泡，带**说话人标签**）+ few-shot 说明。
 *   两栏并排而不是把例句压在第三张卡下面 —— 原来那种竖排会把总高推到
 *   1000px 以上，直接越过字幕安全区（bottom 约从 890 开始）。
 *
 * 例句气泡都带说话人（`用户：` / `灰泽满：`）—— 与 `_format_behavior_rule`
 * 注入给模型的格式（"粉丝说：… → 灰泽满：…"）是同一套标签，
 * 观众看到的就是模型实际读到的样子。
 */
import React from "react";
import {
  COLORS,
  DOMAIN_COLOR,
  FONT_MONO,
  FONT_SANS,
  enter,
  lin,
  sec,
} from "../theme";
import {RichText} from "./RichText";

export type BehaviorSample = {user: string; reply: string};

/** 一栏（情景/反应）。 */
const Stage: React.FC<{
  index: number;
  term: string;
  caption: string;
  frame: number;
  delay: number;
  color: string;
  body: string;
  width: number;
}> = ({index, term, caption, frame, delay, color, body, width}) => {
  const p = enter(frame, delay);
  return (
    <div
      style={{
        width,
        opacity: p,
        transform: `translateY(${(1 - p) * 16}px)`,
        display: "flex",
        flexDirection: "column",
        gap: 7,
      }}
    >
      <div style={{display: "flex", alignItems: "center", gap: 9}}>
        <span
          style={{
            width: 24,
            height: 24,
            borderRadius: 12,
            border: `1.5px solid ${color}`,
            color,
            fontFamily: FONT_MONO,
            fontSize: 14,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flex: "0 0 auto",
          }}
        >
          {index}
        </span>
        <span style={{fontFamily: FONT_SANS, fontSize: 23, fontWeight: 700, color}}>{term}</span>
        <span style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.muted}}>{caption}</span>
      </div>
      <div
        style={{
          fontFamily: FONT_SANS,
          fontSize: 17,
          lineHeight: 1.45,
          color: COLORS.paper,
          padding: "10px 14px",
          background: "rgba(255,255,255,0.032)",
          borderLeft: `3px solid ${color}`,
          borderRadius: 5,
        }}
      >
        <RichText text={body} boldColor={color} />
      </div>
    </div>
  );
};

/** 一条例句：上下两个带说话人的气泡。 */
const SampleBubble: React.FC<{
  sample: BehaviorSample;
  color: string;
  progress: number;
  width: number;
}> = ({sample, color, progress, width}) => (
  <div
    style={{
      width,
      opacity: progress,
      transform: `translateY(${(1 - progress) * 10}px)`,
      display: "flex",
      flexDirection: "column",
      gap: 4,
    }}
  >
    <div style={{display: "flex", alignItems: "baseline", gap: 6}}>
      <span style={{fontFamily: FONT_SANS, fontSize: 13, color: COLORS.dim, flex: "0 0 auto"}}>
        用户：
      </span>
      <span style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.muted, lineHeight: 1.4}}>
        {sample.user}
      </span>
    </div>
    <div style={{display: "flex", alignItems: "baseline", gap: 6}}>
      <span style={{fontFamily: FONT_SANS, fontSize: 13, color, flex: "0 0 auto"}}>灰泽满：</span>
      <span
        style={{
          fontFamily: FONT_SANS,
          fontSize: 16,
          color: COLORS.paper,
          lineHeight: 1.4,
          background: `${color}1a`,
          border: `1px solid ${color}55`,
          borderRadius: "3px 10px 10px 10px",
          padding: "4px 9px",
        }}
      >
        {sample.reply}
      </span>
    </div>
  </div>
);

export const BehaviorAnatomy: React.FC<{
  frame: number;
  name: string;
  trigger: string;
  response: string;
  samples: BehaviorSample[];
  /** 例句逐条展开：第几条已出现（可为小数，做连续动画） */
  samplesReveal?: number;
  /** 是否显示"这不是固定回复表"的注脚 */
  showFootnote?: boolean;
  /** 左栏（情景+反应）宽度 */
  stageWidth?: number;
  /** 右栏（例句）宽度 */
  sampleWidth?: number;
}> = ({
  frame,
  name,
  trigger,
  response,
  samples,
  samplesReveal = 1,
  showFootnote = false,
  stageWidth = 780,
  sampleWidth = 520,
}) => {
  const tone = DOMAIN_COLOR.persona;
  // 动作路径：一根竖向主线，随帧推进（高度按两段卡片的实际高度给足）
  const pathProgress = lin(frame, [sec(0.4), sec(2.4)], [0, 1]);
  const pathH = 250;

  return (
    <div style={{display: "flex", gap: 44, alignItems: "flex-start"}}>
      {/* 左栏：情景 → 反应（一根动作路径串起来） */}
      <div style={{width: stageWidth, display: "flex", flexDirection: "column", gap: 18}}>
        <div style={{display: "flex", alignItems: "center", gap: 12, opacity: enter(frame, 0)}}>
          <span
            style={{
              fontFamily: FONT_SANS,
              fontSize: 21,
              fontWeight: 700,
              color: COLORS.paper,
              padding: "5px 14px",
              border: `1px solid ${tone}`,
              background: `${tone}1c`,
              borderRadius: 6,
            }}
          >
            {name}
          </span>
          <span style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.muted}}>
            behaviors.json · 本轮由 LLM 情景分类选中
          </span>
        </div>

        <div style={{position: "relative", paddingLeft: 40}}>
          {/* 动作路径（竖向主线，连接两段） */}
          <svg
            width="26"
            height={pathH}
            viewBox={`0 0 26 ${pathH}`}
            style={{position: "absolute", left: 4, top: 6}}
          >
            <line x1="13" y1="6" x2="13" y2={pathH - 8} stroke={COLORS.line} strokeWidth="2" />
            <line
              x1="13"
              y1="6"
              x2="13"
              y2={6 + (pathH - 14) * pathProgress}
              stroke={tone}
              strokeWidth="2.4"
            />
            {[6, pathH / 2, pathH - 8].map((y, i) => (
              <circle
                key={y}
                cx="13"
                cy={y}
                r="5"
                fill={pathProgress > i / 3 ? tone : COLORS.panel}
                stroke={tone}
                strokeWidth="1.6"
              />
            ))}
          </svg>

          <div style={{display: "flex", flexDirection: "column", gap: 18}}>
            <Stage
              index={1}
              term="情景"
              caption="用户创造了什么处境"
              frame={frame}
              delay={sec(0.15)}
              color={tone}
              body={trigger}
              width={stageWidth - 40}
            />
            <Stage
              index={2}
              term="反应"
              caption="她往哪个方向走"
              frame={frame}
              delay={sec(0.85)}
              color={tone}
              body={response}
              width={stageWidth - 40}
            />
          </div>
        </div>
      </div>

      {/* 右栏：例句（3 个术语的第三个） */}
      <div style={{width: sampleWidth, display: "flex", flexDirection: "column", gap: 9}}>
        <div style={{display: "flex", alignItems: "center", gap: 9, opacity: enter(frame, sec(1.3))}}>
          <span
            style={{
              width: 24,
              height: 24,
              borderRadius: 12,
              border: `1.5px solid ${tone}`,
              color: tone,
              fontFamily: FONT_MONO,
              fontSize: 14,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            3
          </span>
          <span style={{fontFamily: FONT_SANS, fontSize: 23, fontWeight: 700, color: tone}}>例句</span>
          <span style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.muted}}>
            真人示范 · {samples.length} 条（注入时的样子）
          </span>
        </div>

        <div style={{display: "flex", flexDirection: "column", gap: 9}}>
          {samples.map((s, i) => (
            <SampleBubble
              key={`${s.user}-${i}`}
              sample={s}
              color={tone}
              progress={lin(samplesReveal, [i, i + 0.8], [0, 1])}
              width={sampleWidth}
            />
          ))}
        </div>

        {showFootnote ? (
          <div
            style={{
              fontFamily: FONT_SANS,
              fontSize: 15,
              lineHeight: 1.45,
              color: COLORS.accent,
              opacity: enter(frame, sec(2.6)),
              borderTop: `1px solid ${COLORS.line}`,
              paddingTop: 8,
              marginTop: 2,
            }}
          >
            behavior 不是固定回复表：例句给的是示范，回复仍按用户当前消息生成。
          </div>
        ) : null}
      </div>
    </div>
  );
};
