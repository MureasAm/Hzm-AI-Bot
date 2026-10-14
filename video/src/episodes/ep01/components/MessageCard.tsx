/**
 * MessageCard —— "messages 才是模型真正接收的单位"。
 *
 * role 只有 system / user / assistant 三种；段头显示注入段名（规格 §9.3 只标来源→段名）。
 */
import React from "react";
import {COLORS, FONT_MONO, FONT_SANS} from "../theme";

export const ROLE_COLOR: Record<string, string> = {
  system: COLORS.warn,
  user: COLORS.fact,
  assistant: COLORS.persona,
};

export const MessageCard: React.FC<{
  role: string;
  header?: string;
  text: string;
  progress?: number;
  width?: number;
  tone?: string;
  compact?: boolean;
  /** 右侧对齐（用户消息），用于三板对照三栏布局 */
  alignRight?: boolean;
  style?: React.CSSProperties;
}> = ({
  role,
  header,
  text,
  progress = 1,
  width = 640,
  tone,
  compact = false,
  alignRight = false,
  style,
}) => {
  const color = tone ?? ROLE_COLOR[role] ?? COLORS.muted;
  return (
    <div
      style={{
        width,
        padding: compact ? "8px 12px" : "12px 16px",
        borderRadius: 7,
        background: COLORS.panel,
        border: `1px solid ${COLORS.line}`,
        borderLeft: alignRight ? undefined : `4px solid ${color}`,
        borderRight: alignRight ? `4px solid ${color}` : undefined,
        opacity: progress,
        transform: `translateX(${(1 - progress) * (alignRight ? 12 : -12)}px)`,
        boxSizing: "border-box",
        ...style,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 10,
          marginBottom: compact ? 3 : 6,
          justifyContent: alignRight ? "flex-end" : "flex-start",
        }}
      >
        <span style={{fontFamily: FONT_MONO, fontSize: compact ? 13 : 15, color, letterSpacing: 0.4}}>
          {role}
        </span>
        {header ? (
          <span
            style={{
              fontFamily: FONT_SANS,
              fontSize: compact ? 13 : 14,
              color: COLORS.muted,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {header}
          </span>
        ) : null}
      </div>
      <div
        style={{
          fontFamily: FONT_SANS,
          fontSize: compact ? 15 : 18,
          lineHeight: 1.42,
          color: COLORS.paper,
          wordBreak: "break-word",
        }}
      >
        {text}
      </div>
    </div>
  );
};
