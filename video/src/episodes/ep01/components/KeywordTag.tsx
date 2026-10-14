/**
 * 小标签组件：关键词标注 / 域标签 / 状态徽标。
 *
 * 规格 §4.4：对照不用红叉绿勾粗暴判优，使用关键词标注差异。
 */
import React from "react";
import {COLORS, DOMAIN_COLOR, DOMAIN_LABEL, FONT_MONO, FONT_SANS, type Domain} from "../theme";

export const KeywordTag: React.FC<{
  children: React.ReactNode;
  color: string;
  progress?: number;
  style?: React.CSSProperties;
}> = ({children, color, progress = 1, style}) => (
  <span
    style={{
      display: "inline-block",
      padding: "3px 10px",
      borderRadius: 999,
      border: `1px solid ${color}`,
      background: `${color}1c`,
      color,
      fontFamily: FONT_SANS,
      fontSize: 15,
      opacity: progress,
      transform: `scale(${0.92 + 0.08 * progress})`,
      transformOrigin: "left center",
      ...style,
    }}
  >
    {children}
  </span>
);

export const DomainTag: React.FC<{domain: Domain; progress?: number}> = ({domain, progress = 1}) => (
  <span
    style={{
      display: "inline-block",
      padding: "2px 9px",
      borderRadius: 4,
      background: `${DOMAIN_COLOR[domain]}22`,
      border: `1px solid ${DOMAIN_COLOR[domain]}77`,
      color: DOMAIN_COLOR[domain],
      fontFamily: FONT_SANS,
      fontSize: 14,
      opacity: progress,
    }}
  >
    {DOMAIN_LABEL[domain]}
  </span>
);

export const MethodTag: React.FC<{
  method: string;
  color: string;
  progress?: number;
}> = ({method, color, progress = 1}) => (
  <span
    style={{
      fontFamily: FONT_MONO,
      fontSize: 12,
      color,
      padding: "2px 7px",
      border: `1px solid ${color}66`,
      borderRadius: 4,
      opacity: progress,
      whiteSpace: "nowrap",
    }}
  >
    {method}
  </span>
);

/** 灰色"本轮不注入"标记（规格 §8.1 / §10.4）。 */
export const MissTag: React.FC<{children: React.ReactNode; progress?: number}> = ({
  children,
  progress = 1,
}) => (
  <span
    style={{
      fontFamily: FONT_SANS,
      fontSize: 14,
      color: COLORS.dim,
      padding: "2px 8px",
      borderRadius: 4,
      border: `1px dashed ${COLORS.miss}`,
      opacity: progress,
    }}
  >
    {children}
  </span>
);
