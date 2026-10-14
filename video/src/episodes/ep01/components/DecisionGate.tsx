/**
 * DecisionGate —— 四种判别的统一组件（规格 §8.1、§12.4）。
 *
 * 统一 props：label / question / method / status / color
 * status 仅允许 idle | hit | miss | checking
 * checking 用**帧驱动短脉冲**，不做无限旋转动画。
 *
 * 视觉语义：
 *   hit      → 点亮并进入"待注入"轨道
 *   miss     → 保持灰色，移入"本轮不注入"
 *   idle     → 还没轮到它
 *   checking → 正在判断（LLM / 向量）
 */
import React from "react";
import {COLORS, FONT_MONO, FONT_SANS, enter, lin, pulse, sec} from "../theme";

export type GateStatus = "idle" | "hit" | "miss" | "checking";

export type GateMethod = "always" | "keyword" | "vector" | "llm" | "vector+llm";

const METHOD_LABEL: Record<GateMethod, string> = {
  always: "常驻",
  keyword: "关键词 / 正则",
  vector: "向量召回",
  "vector+llm": "向量 + LLM",
  llm: "LLM 判断",
};

/** 四种判别各自的图标（矢量，语义化，不用 emoji）。 */
const MethodIcon: React.FC<{method: GateMethod; color: string; frame: number; status: GateStatus}> = ({
  method,
  color,
  frame,
  status,
}) => {
  const glow = status === "checking" ? 0.45 + 0.55 * pulse(frame, sec(0.9)) : 1;
  const common = {stroke: color, strokeWidth: 1.9, fill: "none", strokeLinejoin: "round" as const};
  return (
    <svg width="30" height="30" viewBox="0 0 30 30" style={{flex: "0 0 auto", opacity: glow}}>
      {method === "always" ? (
        // 常驻：无限环
        <>
          <circle cx="15" cy="15" r="9" {...common} />
          <circle cx="15" cy="15" r="3" fill={color} stroke="none" />
        </>
      ) : null}
      {method === "keyword" ? (
        // 关键词：文字游标与下划线
        <>
          <path d="M5 9 H25" {...common} />
          <path d="M9 16 H25" {...common} />
          <path d="M5 23 H17" {...common} strokeWidth={2.6} />
        </>
      ) : null}
      {method === "vector" ? (
        // 向量：同向的箭头族
        <>
          <path d="M6 24 L14 8" {...common} />
          <path d="M12 24 L20 8" {...common} opacity={0.7} />
          <path d="M18 24 L26 8" {...common} opacity={0.45} />
        </>
      ) : null}
      {method === "llm" || method === "vector+llm" ? (
        // LLM 判断：分叉然后收敛
        <>
          <path d="M6 8 L15 15" {...common} />
          <path d="M6 22 L15 15" {...common} />
          <path d="M15 15 L25 15" {...common} strokeWidth={2.4} />
          <circle cx="15" cy="15" r="2.6" fill={color} stroke="none" />
        </>
      ) : null}
    </svg>
  );
};

export const DecisionGate: React.FC<{
  label: string;
  /** 人话问题（先讲人话，再显示技术名词 —— 规格 §8） */
  question: string;
  method: GateMethod;
  status: GateStatus;
  color: string;
  /** 局部帧（用于进入动画与脉冲） */
  frame: number;
  delayFrames?: number;
  /** 命中原因 / 未命中原因（真实 trace 的 reason） */
  detail?: string;
  width?: number;
  style?: React.CSSProperties;
}> = ({
  label,
  question,
  method,
  status,
  color,
  frame,
  delayFrames = 0,
  detail,
  width = 420,
  style,
}) => {
  const p = enter(frame, delayFrames);
  const active = status === "hit" || status === "checking";
  const tone =
    status === "hit" ? color : status === "checking" ? COLORS.warn : status === "miss" ? COLORS.dim : COLORS.muted;

  return (
    <div
      style={{
        width,
        padding: "14px 16px",
        borderRadius: 8,
        background: active ? `${color}14` : "rgba(255,255,255,0.022)",
        border: `1px solid ${status === "hit" ? color : COLORS.line}`,
        opacity: p * (status === "idle" ? 0.72 : 1),
        transform: `translateY(${(1 - p) * 16}px)`,
        display: "flex",
        gap: 14,
        alignItems: "flex-start",
        boxSizing: "border-box",
        ...style,
      }}
    >
      <MethodIcon method={method} color={tone} frame={frame} status={status} />
      <div style={{minWidth: 0, flex: 1}}>
        <div style={{display: "flex", alignItems: "baseline", gap: 8, justifyContent: "space-between"}}>
          <span style={{fontFamily: FONT_SANS, fontSize: 20, fontWeight: 600, color: COLORS.paper}}>
            {label}
          </span>
          <span
            style={{
              fontFamily: FONT_MONO,
              fontSize: 12,
              color: tone,
              padding: "2px 7px",
              border: `1px solid ${tone}66`,
              borderRadius: 4,
              whiteSpace: "nowrap",
            }}
          >
            {METHOD_LABEL[method]}
          </span>
        </div>
        <div style={{fontFamily: FONT_SANS, fontSize: 16, color: COLORS.muted, marginTop: 5, lineHeight: 1.35}}>
          {question}
        </div>
        {detail ? (
          <div
            style={{
              fontFamily: FONT_SANS,
              fontSize: 14,
              color: status === "hit" ? color : COLORS.dim,
              marginTop: 6,
              lineHeight: 1.35,
            }}
          >
            {detail}
          </div>
        ) : null}
        {/* 状态条：命中点亮 / 未命中保持灰 */}
        <div
          style={{
            marginTop: 9,
            height: 3,
            borderRadius: 2,
            background: COLORS.line,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${lin(status === "hit" ? p : status === "checking" ? 0.55 : 0, [0, 1], [0, 100])}%`,
              background: status === "hit" ? color : status === "checking" ? COLORS.warn : COLORS.dim,
            }}
          />
        </div>
      </div>
    </div>
  );
};
