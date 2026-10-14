/**
 * FileCard —— "文件是人维护内容的单位"。
 *
 * 规格 §12.4 / §14：文件卡是公共视觉模块之一；未命中的卡**保持灰色**留在
 * "本轮不注入"区，不做对错判断动画。
 */
import React from "react";
import {COLORS, DOMAIN_COLOR, FONT_MONO, FONT_SANS, type Domain} from "../theme";

export const FileCard: React.FC<{
  name: string;
  note?: string;
  domain: Domain;
  /** 0~1 进入进度（由调用方的 spring 提供，帧驱动） */
  progress?: number;
  /** 未命中：保持灰色（规格 §8.1） */
  dimmed?: boolean;
  width?: number;
  /** 本轮点亮 */
  highlight?: boolean;
  fontSize?: number;
  style?: React.CSSProperties;
}> = ({
  name,
  note,
  domain,
  progress = 1,
  dimmed = false,
  width = 340,
  highlight = false,
  fontSize = 19,
  style,
}) => {
  const tone = dimmed ? COLORS.dim : DOMAIN_COLOR[domain];
  return (
    <div
      style={{
        width,
        padding: "13px 15px",
        borderRadius: 7,
        background: dimmed ? "rgba(255,255,255,0.028)" : COLORS.panelSoft,
        border: `1px solid ${highlight ? tone : COLORS.line}`,
        borderLeft: `4px solid ${tone}`,
        boxShadow: highlight ? `0 0 0 1px ${tone}44, 0 10px 26px rgba(0,0,0,0.28)` : "none",
        opacity: (dimmed ? 0.5 : 1) * progress,
        transform: `translateY(${(1 - progress) * 14}px)`,
        display: "flex",
        alignItems: "center",
        gap: 12,
        ...style,
      }}
    >
      {/* 折角文件图标（矢量，不用 emoji、不用模板化 AI 视觉） */}
      <svg width="24" height="30" viewBox="0 0 26 32" style={{flex: "0 0 auto"}}>
        <path
          d="M2 2 H16 L24 10 V30 H2 Z"
          fill="none"
          stroke={tone}
          strokeWidth="1.8"
          strokeLinejoin="round"
        />
        <path d="M16 2 V10 H24" fill="none" stroke={tone} strokeWidth="1.8" strokeLinejoin="round" />
      </svg>
      <div style={{minWidth: 0}}>
        <div
          style={{
            fontFamily: FONT_MONO,
            fontSize,
            color: dimmed ? COLORS.muted : COLORS.paper,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {name}
        </div>
        {note ? (
          <div style={{fontFamily: FONT_SANS, fontSize: 14, color: COLORS.muted, marginTop: 3}}>
            {note}
          </div>
        ) : null}
      </div>
    </div>
  );
};
