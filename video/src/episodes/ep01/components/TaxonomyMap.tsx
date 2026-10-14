/**
 * TaxonomyMap —— 三类内容地图：人格 / 事实 / 现场（规格 §7）。
 *
 * 全片固定色彩语义：人格=紫/玫红，事实=蓝/青，现场=绿/橙。
 * 用途：让观众能复述这三类，并且后面每个场景的颜色都有出处。
 */
import React from "react";
import {
  COLORS,
  DOMAIN_COLOR,
  DOMAIN_LABEL,
  DOMAIN_MEANING,
  FONT_MONO,
  FONT_SANS,
  type Domain,
  enter,
  lin,
  sec,
} from "../theme";

export type TaxonomyEntry = {
  domain: Domain;
  name: string;
  /** 该层负责什么（一句话） */
  duty: string;
  items: string[];
};

export const TAXONOMY: TaxonomyEntry[] = [
  {
    domain: "persona",
    name: "人格",
    duty: "她会怎样反应",
    items: ["system_prompt.txt", "behaviors.json"],
  },
  {
    domain: "fact",
    name: "事实",
    duty: "她知道什么",
    items: [
      "terms.json",
      "preferences.json",
      "core_stories.json",
      "statement_final.json",
      "schedule.json",
      "legendary.json",
    ],
  },
  {
    domain: "scene",
    name: "现场",
    duty: "此刻发生了什么",
    items: [
      "用户长期记忆",
      "会话记忆",
      "最近几轮原文",
      "时间 / 天气 / 直播状态",
      "群聊现场",
      "图片描述",
      "批量归纳",
      "指代补全",
      "连续索要次数",
    ],
  },
];

const Column: React.FC<{
  entry: TaxonomyEntry;
  frame: number;
  delay: number;
  /** 展开明细的进度 0~1（由场景控制） */
  reveal: number;
  width: number;
}> = ({entry, frame, delay, reveal, width}) => {
  const p = enter(frame, delay);
  const color = DOMAIN_COLOR[entry.domain];
  return (
    <div
      style={{
        width,
        opacity: p,
        transform: `translateY(${(1 - p) * 22}px)`,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          padding: "16px 18px",
          borderRadius: 8,
          background: `${color}18`,
          border: `1px solid ${color}`,
          borderTop: `4px solid ${color}`,
        }}
      >
        <div style={{fontFamily: FONT_SANS, fontSize: 34, fontWeight: 700, color}}>{entry.name}</div>
        <div style={{fontFamily: FONT_SANS, fontSize: 20, color: COLORS.paper, marginTop: 4}}>
          {entry.duty}
        </div>
      </div>
      <div style={{display: "flex", flexDirection: "column", gap: 7}}>
        {entry.items.map((it, i) => {
          const ip = lin(reveal, [i / entry.items.length, Math.min(1, (i + 1.6) / entry.items.length)], [0, 1]);
          const isFile = it.endsWith(".json") || it.endsWith(".txt");
          return (
            <div
              key={it}
              style={{
                fontFamily: isFile ? FONT_MONO : FONT_SANS,
                fontSize: isFile ? 16 : 17,
                color: COLORS.paper,
                padding: "7px 12px",
                borderRadius: 5,
                background: "rgba(255,255,255,0.032)",
                borderLeft: `3px solid ${color}${isFile ? "" : "88"}`,
                opacity: 0.25 + 0.75 * ip,
                transform: `translateX(${(1 - ip) * 10}px)`,
              }}
            >
              {it}
            </div>
          );
        })}
      </div>
      <div style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.muted}}>
        {DOMAIN_MEANING[entry.domain]}
      </div>
    </div>
  );
};

export const TaxonomyMap: React.FC<{
  frame: number;
  /** 明细展开进度（0 = 只有三栏标题，1 = 明细全出） */
  reveal?: number;
  columnWidth?: number;
  gap?: number;
}> = ({frame, reveal = 1, columnWidth = 500, gap = 46}) => {
  return (
    <div style={{display: "flex", gap, justifyContent: "center", alignItems: "flex-start"}}>
      {TAXONOMY.map((e, i) => (
        <Column
          key={e.domain}
          entry={e}
          frame={frame}
          delay={sec(0.15) * i}
          reveal={reveal}
          width={columnWidth}
        />
      ))}
    </div>
  );
};

/** 小尺寸的三色图例（其他场景复用，保证语义一致）。 */
export const DomainLegend: React.FC<{frame: number; style?: React.CSSProperties}> = ({frame, style}) => (
  <div style={{display: "flex", gap: 20, ...style}}>
    {(["persona", "fact", "scene"] as Domain[]).map((d, i) => {
      const p = enter(frame, i * 4);
      return (
        <div key={d} style={{display: "flex", alignItems: "center", gap: 8, opacity: p}}>
          <span
            style={{
              width: 12,
              height: 12,
              borderRadius: 3,
              background: DOMAIN_COLOR[d],
              display: "inline-block",
            }}
          />
          <span style={{fontFamily: FONT_SANS, fontSize: 16, color: COLORS.muted}}>
            {DOMAIN_LABEL[d]} · {DOMAIN_MEANING[d]}
          </span>
        </div>
      );
    })}
  </div>
);
