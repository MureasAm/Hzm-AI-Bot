/**
 * Scene 06 · four-gates —— 四种判别（规格 §8，按用户反馈重做同步）。
 *
 * 先讲人话问题，再显示技术名词：
 *   每轮都需要吗？          → 常驻
 *   消息里明确提到了吗？    → 关键词 / 正则
 *   意思相近吗？            → 向量召回
 *   这是哪种人际情景？      → LLM 判断
 *
 * 用户反馈（"06 部分，动画和声音完全不匹配，保留了很多无声的阶段"）：
 *   原来的 `at` 值是**相对旧时间轴手算的常数**（其中一个竟是 `49.6 - 45.5`），
 *   改文案/改语速后全部错位——旁白讲到第三种时，画面可能还在演第一种，
 *   于是看起来就是"动画和声音不匹配 + 一段段没声音"。
 *
 *   现在每个 Gate 的动画直接绑到**它自己那句旁白**（`at()` 反查），
 *   并且同一时刻只让**当前那句**对应的 Gate 处于 checking/hit，
 *   其余保持 idle——一个时刻只有一个焦点，这也符合规格 §2.2 第 3 条。
 *
 * 版式：四个 Gate 排成 2×2，下方一条组合路径说明。
 * 可用高度：top 214 起、字幕安全区约 y=890 → 约 670px。
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
import {at} from "../ep01-fixtures";
import {DecisionGate, type GateMethod, type GateStatus} from "../components/DecisionGate";
import {FileCard} from "../components/FileCard";
import type {Domain} from "../theme";

type GateSpec = {
  id: string;
  label: string;
  question: string;
  method: GateMethod;
  color: string;
  files: {name: string; domain: Domain}[];
  /** 讲这个 Gate 的那句旁白 id —— 动画绑在它上面 */
  line: string;
  /** 降级方向小标签 */
  degrade: string;
};

const GATES: GateSpec[] = [
  {
    id: "always",
    label: "AlwaysGate",
    question: "每轮都需要吗？",
    method: "always",
    color: COLORS.warn,
    line: "four-gates-3",
    files: [
      {name: "system_prompt.txt", domain: "persona"},
      {name: "schedule.json", domain: "fact"},
      {name: "当前时间上下文", domain: "scene"},
    ],
    degrade: "内容缺失时宁可不注入",
  },
  {
    id: "keyword",
    label: "KeywordGate",
    question: "消息里明确提到了吗？",
    method: "keyword",
    color: DOMAIN_COLOR.fact,
    line: "four-gates-4",
    files: [
      {name: "terms.json", domain: "fact"},
      {name: "preferences.json", domain: "fact"},
      {name: "corpus 钩子", domain: "fact"},
      {name: "legendary 粗筛", domain: "fact"},
    ],
    degrade: "宁可漏，不误触",
  },
  {
    id: "vector",
    label: "VectorGate",
    question: "意思相近吗？",
    method: "vector",
    color: DOMAIN_COLOR.fact,
    line: "four-gates-5",
    files: [
      {name: "statement_final.json", domain: "fact"},
      {name: "core_stories.json", domain: "fact"},
    ],
    degrade: "过阈值才带",
  },
  {
    id: "llm",
    label: "LlmGate",
    question: "这是哪种人际情景？候选真的相关吗？",
    method: "llm",
    color: DOMAIN_COLOR.persona,
    line: "four-gates-6",
    files: [
      {name: "behaviors.json 情景分类", domain: "persona"},
      {name: "corpus 确认", domain: "fact"},
      {name: "歧义词确认", domain: "fact"},
    ],
    degrade: "判不出来一律不带（宁可漏不可错）",
  },
];

/** 组合路径：主线画粗、辅助线画细（规格 §8.2）。 */
const COMBINED = [
  {name: "corpus", main: "向量候选", aux: "词面补位 → LLM 确认", line: "four-gates-9"},
  {name: "legendary", main: "关键词粗筛", aux: "LLM 语境确认", line: "four-gates-10"},
  {name: "behavior", main: "LLM 情景分类", aux: "判别词兜底", line: "four-gates-10"},
];

const GatePanel: React.FC<{g: GateSpec; frame: number}> = ({g, frame}) => {
  // 动画锚点 = 这句旁白开始的时刻
  const start = at("four-gates", g.line, -0.5);
  const settled = at("four-gates", g.line, 3.0);
  // 同一时刻只让当前这句对应的 Gate 活跃；已讲过的保持 hit，没讲到的 idle
  const status: GateStatus =
    frame < start - sec(0.4)
      ? "idle"
      : frame < settled
        ? "checking"
        : "hit";

  return (
    <div style={{display: "flex", flexDirection: "column", gap: 9, width: 470}}>
      <DecisionGate
        label={g.label}
        question={g.question}
        method={g.method}
        status={status}
        color={g.color}
        frame={frame}
        delayFrames={start - sec(0.3)}
        width={470}
        detail={status === "hit" ? `${g.files.length} 类内容走这条路` : undefined}
      />

      {/* 进入该 Gate 的文件：命中亮起 / 未命中灰色 */}
      <div style={{display: "flex", flexDirection: "column", gap: 5, paddingLeft: 4}}>
        {g.files.map((f, i) => {
          const ok = status === "hit";
          const p = lin(frame, [start + sec(0.3) + i * 3, start + sec(0.9) + i * 3], [0, 1]);
          return (
            <FileCard
              key={f.name}
              name={f.name}
              domain={f.domain}
              dimmed={!ok}
              highlight={ok}
              progress={p}
              width={462}
              fontSize={14}
              style={{padding: "8px 12px"}}
            />
          );
        })}
      </div>

      {/* 降级方向 + 组合路径说明（小标签，本集不展开异常处理） */}
      <div style={{display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap"}}>
        <span
          style={{
            fontFamily: FONT_SANS,
            fontSize: 12,
            color: COLORS.dim,
            padding: "2px 7px",
            border: `1px dashed ${COLORS.miss}`,
            borderRadius: 4,
            opacity: lin(frame, [start + sec(1.2), start + sec(1.8)], [0, 1]),
          }}
        >
          失败降级
        </span>
        <span
          style={{
            fontFamily: FONT_SANS,
            fontSize: 13,
            color: COLORS.muted,
            opacity: lin(frame, [start + sec(1.2), start + sec(1.8)], [0, 1]),
          }}
        >
          {g.degrade}
        </span>
      </div>
    </div>
  );
};

export const FourGatesScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  const introP = enter(frame, 0);
  // 「命中就点亮 / 没命中保持灰色」那句
  const legendP = enter(frame, at("four-gates", "four-gates-7", -0.4));
  // 「别理解成每个文件只用一种判别」那句
  const combinedP = enter(frame, at("four-gates", "four-gates-8", -0.4));

  // 组合路径逐条点亮（讲 corpus / 梗库·行为 那两句时）
  const combinedActive = (line: string) =>
    frame >= at("four-gates", line, -0.4) && frame < at("four-gates", line, 5.5);

  return (
    <div
      style={{
        position: "absolute",
        left: 84,
        right: 84,
        top: 210,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      {/* 开场一句：这段在讲什么 */}
      <div
        style={{
          opacity: introP,
          transform: `translateY(${(1 - introP) * 10}px)`,
          fontFamily: FONT_SANS,
          fontSize: 19,
          color: COLORS.muted,
        }}
      >
        四种判别 —— 每种前面都是一个很朴素的问题
      </div>

      {/* 四个 Gate：2×2 */}
      <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px 44px"}}>
        {GATES.map((g) => (
          <GatePanel key={g.id} g={g} frame={frame} />
        ))}
      </div>

      {/* 结果去向 */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 26,
          opacity: legendP,
          transform: `translateY(${(1 - legendP) * 10}px)`,
        }}
      >
        <div style={{display: "flex", alignItems: "center", gap: 9}}>
          <span style={{width: 22, height: 3, background: COLORS.hit, display: "inline-block"}} />
          <span style={{fontFamily: FONT_SANS, fontSize: 17, color: COLORS.hit}}>
            命中 → 进入待注入轨道
          </span>
        </div>
        <div style={{display: "flex", alignItems: "center", gap: 9}}>
          <span style={{width: 22, height: 3, background: COLORS.miss, display: "inline-block"}} />
          <span style={{fontFamily: FONT_SANS, fontSize: 17, color: COLORS.dim}}>
            未命中 → 保持灰色，留在本轮不注入
          </span>
        </div>
      </div>

      {/* 组合路径（§8.2）：主线画粗、辅助线画细 */}
      <div
        style={{
          opacity: combinedP,
          transform: `translateY(${(1 - combinedP) * 10}px)`,
          borderTop: `1px solid ${COLORS.line}`,
          paddingTop: 10,
          display: "flex",
          alignItems: "center",
          gap: 22,
        }}
      >
        <span style={{fontFamily: FONT_SANS, fontSize: 16, color: COLORS.muted, flex: "0 0 auto"}}>
          有些路径是组合的（主线粗、辅助线细）：
        </span>
        {COMBINED.map((c) => {
          const active = combinedActive(c.line);
          return (
            <div
              key={c.name}
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: 7,
                padding: "4px 10px",
                borderRadius: 5,
                border: `1px solid ${active ? COLORS.warn : COLORS.line}`,
                background: active ? `${COLORS.warn}18` : "rgba(255,255,255,0.02)",
              }}
            >
              <span
                style={{
                  fontFamily: FONT_MONO,
                  fontSize: 13,
                  color: active ? COLORS.warn : COLORS.dim,
                }}
              >
                {c.name}
              </span>
              <span style={{fontFamily: FONT_SANS, fontSize: 14, color: COLORS.paper, fontWeight: 600}}>
                {c.main}
              </span>
              <span style={{fontFamily: FONT_SANS, fontSize: 13, color: COLORS.dim}}>→ {c.aux}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
