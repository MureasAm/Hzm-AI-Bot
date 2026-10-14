/**
 * InjectionTrace —— 完整链路演示的核心画面（规格 §10、§12.4）。
 *
 * 布局（规格明确要求）：
 *   左：输入批次
 *   中：判别器与被点亮的文件
 *   右：messages 栈
 *   下：未命中灰色区
 *   → **画面最终必须停留至少 2 秒**，让观众看清完整结果（由调用方保证停留时长）
 *
 * 数据全部来自视频数据文件：decisions（命中/未命中 + 真实 reason）+ messages（真实 15 段）。
 */
import React from "react";
import {
  COLORS,
  DOMAIN_COLOR,
  DOMAIN_LABEL,
  FONT_MONO,
  FONT_SANS,
  type Domain,
  enter,
  lin,
  sec,
} from "../theme";
import type {Decision, MessageBlock} from "../ep01-fixtures";
import {MethodTag, MissTag} from "./KeywordTag";
import {ROLE_COLOR} from "./MessageCard";

const METHOD_SHORT: Record<string, string> = {
  always: "常驻",
  keyword: "关键词",
  vector: "向量",
  "vector+llm": "向量+LLM",
  llm: "LLM",
};

/** 左：输入批次（三条消息 + 读秒窗口说明）。 */
export const InputBatch: React.FC<{
  frame: number;
  lines: string[];
  note: string;
  clock: string;
  weather: {city: string; text: string; temperatureC: number};
  width?: number;
}> = ({frame, lines, note, clock, weather, width = 380}) => {
  const p = enter(frame, 0);
  const clockLabel = clock.replace("T", " ").slice(5, 16);
  return (
    <div style={{width, opacity: p, display: "flex", flexDirection: "column", gap: 9}}>
      <div style={{fontFamily: FONT_MONO, fontSize: 15, color: COLORS.accent, letterSpacing: 1}}>
        输入批次
      </div>
      {lines.map((t, i) => {
        const ip = enter(frame, sec(0.12) * i);
        return (
          <div
            key={t}
            style={{
              fontFamily: FONT_SANS,
              fontSize: 16,
              lineHeight: 1.4,
              color: COLORS.paper,
              padding: "9px 12px",
              borderRadius: "10px 10px 10px 3px",
              background: COLORS.panelSoft,
              border: `1px solid ${COLORS.line}`,
              opacity: ip,
              transform: `translateX(${(1 - ip) * -12}px)`,
            }}
          >
            {t}
          </div>
        );
      })}
      <div
        style={{
          fontFamily: FONT_SANS,
          fontSize: 13,
          color: COLORS.muted,
          opacity: enter(frame, sec(0.5)),
          lineHeight: 1.4,
        }}
      >
        {note}
      </div>
      <div
        style={{
          marginTop: 2,
          padding: "8px 11px",
          borderRadius: 5,
          background: `${DOMAIN_COLOR.scene}14`,
          border: `1px solid ${DOMAIN_COLOR.scene}55`,
          fontFamily: FONT_MONO,
          fontSize: 13,
          color: COLORS.paper,
          opacity: enter(frame, sec(0.6)),
          lineHeight: 1.5,
        }}
      >
        <div>{clockLabel}</div>
        <div>
          {weather.city} · {weather.text} {weather.temperatureC}℃
        </div>
      </div>
    </div>
  );
};

/** 中：判别器列表（真实 decisions，命中点亮、未命中保持灰）。 */
export const GateColumn: React.FC<{
  frame: number;
  decisions: Decision[];
  /** 每条决策开始被判定的帧（相对场景局部帧） */
  startAt: number;
  step: number;
  width?: number;
  /** 紧凑行高：本集把 21 条判别压进一屏，行高必须小 */
  compact?: boolean;
}> = ({frame, decisions, startAt, step, width = 420, compact = true}) => {
  const rowH = compact ? 24 : 34;
  return (
    <div style={{width, display: "flex", flexDirection: "column", gap: compact ? 2 : 5}}>
      <div
        style={{
          fontFamily: FONT_MONO,
          fontSize: 15,
          color: COLORS.accent,
          letterSpacing: 1,
          marginBottom: 3,
        }}
      >
        判别器 · 被点亮的文件
      </div>
      {decisions.map((d, i) => {
        const t = startAt + i * step;
        const appeared = frame >= t;
        const p = enter(frame, t);
        const tone = d.hit ? DOMAIN_COLOR[d.domain as Domain] : COLORS.dim;
        return (
          <div
            key={d.id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 7,
              height: rowH,
              padding: compact ? "0 8px" : "6px 9px",
              borderRadius: 4,
              background: d.hit ? `${tone}16` : "rgba(255,255,255,0.018)",
              border: `1px solid ${d.hit ? tone : COLORS.line}`,
              opacity: appeared ? p * (d.hit ? 1 : 0.55) : 0,
              transform: `translateX(${(1 - p) * 14}px)`,
              boxSizing: "border-box",
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: 3,
                background: d.hit ? tone : COLORS.miss,
                flex: "0 0 auto",
              }}
            />
            <span
              style={{
                fontFamily: FONT_SANS,
                fontSize: compact ? 13 : 15,
                color: d.hit ? COLORS.paper : COLORS.muted,
                width: compact ? 112 : 132,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                flex: "0 0 auto",
              }}
            >
              {d.label}
            </span>
            <MethodTag method={METHOD_SHORT[d.method] ?? d.method} color={tone} />
            <span
              style={{
                fontFamily: FONT_SANS,
                fontSize: compact ? 11 : 12,
                color: d.hit ? tone : COLORS.dim,
                marginLeft: "auto",
                flex: "0 0 auto",
              }}
            >
              {d.hit ? "注入" : "沉默"}
            </span>
          </div>
        );
      })}
    </div>
  );
};

/** 右：messages 栈（真实段数与段头）。 */
export const MessageStack: React.FC<{
  frame: number;
  messages: MessageBlock[];
  startAt: number;
  step: number;
  width?: number;
  maxVisible?: number;
  compact?: boolean;
}> = ({frame, messages, startAt, step, width = 460, maxVisible = 15, compact = true}) => {
  const shown = messages.slice(0, maxVisible);
  const rowH = compact ? 24 : 30;
  return (
    <div style={{width, display: "flex", flexDirection: "column", gap: compact ? 2 : 3}}>
      <div
        style={{
          fontFamily: FONT_MONO,
          fontSize: 15,
          color: COLORS.accent,
          letterSpacing: 1,
          marginBottom: 3,
        }}
      >
        messages 栈 · {messages.length} 段
      </div>
      {shown.map((m, i) => {
        const t = startAt + i * step;
        const p = enter(frame, t);
        const roleColor = ROLE_COLOR[m.role] ?? COLORS.muted;
        return (
          <div
            key={m.index}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              height: rowH,
              padding: compact ? "0 8px" : "4px 8px",
              borderRadius: 4,
              background: COLORS.panel,
              borderLeft: `3px solid ${roleColor}`,
              opacity: frame >= t ? p : 0,
              transform: `translateX(${(1 - p) * 10}px)`,
              boxSizing: "border-box",
            }}
          >
            <span
              style={{
                fontFamily: FONT_MONO,
                fontSize: 11,
                color: COLORS.dim,
                width: 18,
                flex: "0 0 auto",
              }}
            >
              {String(m.index).padStart(2, "0")}
            </span>
            <span
              style={{
                fontFamily: FONT_MONO,
                fontSize: 12,
                color: roleColor,
                width: 54,
                flex: "0 0 auto",
              }}
            >
              {m.role}
            </span>
            <span
              style={{
                fontFamily: FONT_SANS,
                fontSize: 13,
                color: COLORS.paper,
                flex: 1,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {m.header}
            </span>
            <span style={{fontFamily: FONT_MONO, fontSize: 11, color: COLORS.dim, flex: "0 0 auto"}}>
              {m.chars}
            </span>
          </div>
        );
      })}
      {/* 空栈占位：让"还在拼装"这件事可见 */}
      {frame < startAt ? (
        <div
          style={{
            fontFamily: FONT_SANS,
            fontSize: 14,
            color: COLORS.dim,
            paddingTop: 8,
          }}
        >
          还没开始拼装
        </div>
      ) : null}
    </div>
  );
};

/** 下：未命中灰色区（规格 §10.4）。 */
export const MissZone: React.FC<{
  frame: number;
  decisions: Decision[];
  startAt: number;
  /** 是否已经全部揭示 */
  revealed: boolean;
  style?: React.CSSProperties;
}> = ({frame, decisions, startAt, revealed, style}) => {
  const miss = decisions.filter((d) => !d.hit);
  const p = enter(frame, startAt);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "9px 16px",
        borderRadius: 7,
        background: "rgba(255,255,255,0.016)",
        border: `1px dashed ${COLORS.miss}`,
        opacity: p,
        flexWrap: "wrap",
        ...style,
      }}
    >
      <span style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.dim, flex: "0 0 auto"}}>
        本轮不注入
      </span>
      {miss.map((d, i) => (
        <MissTag key={d.id} progress={lin(frame, [startAt + i * 4, startAt + i * 4 + 10], [0, 1])}>
          {d.label}
        </MissTag>
      ))}
      {revealed ? (
        <span style={{fontFamily: FONT_SANS, fontSize: 14, color: COLORS.dim, marginLeft: "auto"}}>
          保持沉默 · 不相关的内容不会被塞进去
        </span>
      ) : null}
    </div>
  );
};

/** 顶部：本轮固定场景摘要。 */
export const FixtureHeader: React.FC<{
  frame: number;
  clock: string;
  weather: {city: string; text: string};
  user: string;
  topic: string;
}> = ({frame, clock, weather, user, topic}) => {
  const p = enter(frame, 0);
  const clockLabel = clock.replace("T", " ").slice(0, 16);
  return (
    <div
      style={{
        display: "flex",
        gap: 26,
        alignItems: "center",
        opacity: p,
        fontFamily: FONT_MONO,
        fontSize: 15,
        color: COLORS.muted,
      }}
    >
      <span style={{color: COLORS.accent}}>固定 fixture · 全部虚构</span>
      <span>{clockLabel}</span>
      <span>
        {weather.city} {weather.text}
      </span>
      <span>用户 {user}</span>
      <span
        style={{
          fontFamily: FONT_SANS,
          color: COLORS.paper,
          maxWidth: 640,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {topic}
      </span>
    </div>
  );
};

export const DOMAIN_LABEL_MAP = DOMAIN_LABEL;
export {sec};
