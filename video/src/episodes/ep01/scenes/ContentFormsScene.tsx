/**
 * Scene 07 · content-forms —— 文件与内容形式（规格 §9，按用户反馈重构）。
 *
 * 用户要求：
 *   ① **记忆部分不讲**（它是第二集的主题）→ 本段完全移除；
 *   ② 把**所有可能注入的文件/条目**都列成卡片；
 *   ③ 主要说明**判别方式**与**内容形式**。
 *
 * 所以这一段的画面是一张"注入清单"：三列（人格 / 事实 / 现场），
 * 每条一张卡，卡上只放三行 —— 文件（或条目）+ 判别 + 形式。
 * 旁白讲到哪一条，哪一条就点亮，其余压暗（自己会循环）。
 * 底部只留"命中才带 / 命中就直发"这两条通用结论。
 *
 * 版式：三列各 552px（3×552 + 2×36 = 1728，正好等于可用宽度）；
 * 每张卡两行以内，总高压在字幕安全区之上。
 */
import React from "react";
import {
  COLORS,
  DOMAIN_COLOR,
  FONT_MONO,
  FONT_SANS,
  type Domain,
  enter,
  lin,
  sec,
} from "../theme";
import {at, atSec} from "../ep01-fixtures";

type Item = {
  /** 卡片标题（文件名或条目名） */
  name: string;
  /** 判别方式 */
  gate: string;
  /** 内容形式 */
  form: string;
  /** 旁白讲到它的那一行（用于点亮 + 触发进入动画） */
  line: string;
  /** 文件名（用等宽字体）/ 中文条目（用常规字体） */
  mono?: boolean;
};

const COLUMNS: {domain: Domain; title: string; duty: string; items: Item[]}[] = [
  {
    domain: "persona",
    title: "人格",
    duty: "她会怎样反应",
    items: [
      {
        name: "core/system_prompt.txt",
        gate: "常驻",
        form: "稳定说明 → 首条 system message",
        line: "content-forms-1",
        mono: true,
      },
      {
        name: "behavior/behaviors.json",
        gate: "LLM 判当前人际情景",
        form: "情景 — 反应 — 例句 → 行为指令",
        line: "content-forms-3",
        mono: true,
      },
    ],
  },
  {
    domain: "fact",
    title: "事实",
    duty: "她知道什么",
    items: [
      {
        name: "world/statement_final.json",
        gate: "向量候选 + 词面补位 + LLM 确认",
        form: "过去发生过的事实陈述",
        line: "content-forms-4",
        mono: true,
      },
      {
        name: "world/terms.json",
        gate: "关键词 / 别名 / 正则（always 条目常驻）",
        form: "词、人物、梗的意思 + 她的态度",
        line: "content-forms-4",
        mono: true,
      },
      {
        name: "world/preferences.json",
        gate: "keyword 子串命中",
        form: "去掉「某次直播」仍成立的一贯偏好",
        line: "content-forms-8",
        mono: true,
      },
      {
        name: "world/core_stories.json",
        gate: "低阈值向量召回",
        form: "印象最深的经历（事件陈述）",
        line: "content-forms-8",
        mono: true,
      },
      {
        name: "world/schedule.json",
        gate: "常驻（weekly 制，永不过期）",
        form: "当前周表 →「几点播」的地面真值",
        line: "content-forms-9",
        mono: true,
      },
      {
        name: "world/legendary.json",
        gate: "关键词 / 正则粗筛 + LLM 语境确认",
        form: "固定该怎么答 —— 不进 messages，命中直接发",
        line: "content-forms-9",
        mono: true,
      },
    ],
  },
  {
    domain: "scene",
    title: "现场",
    duty: "此刻发生了什么",
    items: [
      {
        name: "用户长期记忆",
        gate: "类型化条目，跨会话存活",
        form: "你是谁、稳定事实与承诺",
        line: "content-forms-10",
      },
      {
        name: "会话记忆",
        gate: "会话未冷场（3h 内）",
        form: "这一场在聊什么、发生过什么",
        line: "content-forms-10",
      },
      {
        name: "最近几轮短期原文",
        gate: "常驻（最近几轮）",
        form: "真实对话原文 + 每条的时间间隔",
        line: "content-forms-10",
      },
      {
        name: "当前时间 / 天气 / 直播状态",
        gate: "常驻（每分钟会变）",
        form: "一行感知 → 放 messages 最末尾",
        line: "content-forms-10",
      },
      {
        name: "群聊现场",
        gate: "群聊才有（私聊为空）",
        form: "在场成员 + 群内近况",
        line: "content-forms-10",
      },
      {
        name: "图片描述",
        gate: "本批有图片才有",
        form: "并入 user 消息，不单独占段",
        line: "content-forms-10",
      },
      {
        name: "一批消息的归纳",
        gate: "本批 ≥2 条才有",
        form: "整批的理解提示",
        line: "content-forms-10",
      },
      {
        name: "指代性短消息补全",
        gate: "短/指代消息才有",
        form: "把「那个呢」补成完整意思",
        line: "content-forms-10",
      },
      {
        name: "连续索要次数",
        gate: "LLM 顺带判（≥2 轮才有）",
        form: "【客观情况】只给事实，不给指令",
        line: "content-forms-10",
      },
    ],
  },
];

/** 一张卡：文件 + 判别 + 形式（三行，克制）。 */
const ItemCard: React.FC<{
  item: Item;
  domain: Domain;
  frame: number;
  /** 是否当前旁白正讲到它 */
  active: boolean;
  width: number;
}> = ({item, domain, frame, active, width}) => {
  const tone = DOMAIN_COLOR[domain];
  const p = enter(frame, at("content-forms", item.line, -0.35));
  return (
    <div
      style={{
        width,
        opacity: p * (active ? 1 : 0.5),
        transform: `translateY(${(1 - p) * 10}px)`,
        padding: active ? "8px 11px" : "7px 11px",
        borderRadius: 6,
        background: active ? `${tone}1c` : "rgba(255,255,255,0.026)",
        border: `1px solid ${active ? tone : COLORS.line}`,
        borderLeft: `3px solid ${tone}`,
        display: "flex",
        flexDirection: "column",
        gap: 3,
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          fontFamily: item.mono ? FONT_MONO : FONT_SANS,
          fontSize: item.mono ? 15 : 16,
          fontWeight: item.mono ? 400 : 600,
          color: active ? COLORS.paper : COLORS.muted,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {item.name}
      </div>
      <div style={{display: "flex", gap: 7, alignItems: "flex-start"}}>
        <span
          style={{
            fontFamily: FONT_SANS,
            fontSize: 12,
            color: COLORS.dim,
            flex: "0 0 auto",
            width: 30,
          }}
        >
          判别
        </span>
        <span style={{fontFamily: FONT_SANS, fontSize: 13, color: active ? COLORS.paper : COLORS.muted, lineHeight: 1.35}}>
          {item.gate}
        </span>
      </div>
      <div style={{display: "flex", gap: 7, alignItems: "flex-start"}}>
        <span
          style={{
            fontFamily: FONT_SANS,
            fontSize: 12,
            color: COLORS.dim,
            flex: "0 0 auto",
            width: 30,
          }}
        >
          形式
        </span>
        <span style={{fontFamily: FONT_SANS, fontSize: 13, color: active ? COLORS.paper : COLORS.muted, lineHeight: 1.35}}>
          {item.form}
        </span>
      </div>
    </div>
  );
};

const ColumnHeader: React.FC<{col: (typeof COLUMNS)[number]; frame: number; width: number; count: number}> = ({
  col,
  frame,
  width,
  count,
}) => {
  const tone = DOMAIN_COLOR[col.domain];
  const p = enter(frame, 0);
  return (
    <div
      style={{
        width,
        opacity: p,
        transform: `translateY(${(1 - p) * 12}px)`,
        display: "flex",
        alignItems: "baseline",
        gap: 10,
        paddingBottom: 5,
        borderBottom: `2px solid ${tone}`,
      }}
    >
      <span style={{fontFamily: FONT_SANS, fontSize: 25, fontWeight: 700, color: tone}}>{col.title}</span>
      <span style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.muted}}>{col.duty}</span>
      <span style={{fontFamily: FONT_MONO, fontSize: 13, color: COLORS.dim, marginLeft: "auto"}}>
        {count} 项
      </span>
    </div>
  );
};

export const ContentFormsScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  // 当前旁白讲到哪张卡（用行 id 判断，避免写死秒数）
  const t = frame / 30;
  const activeLine = (() => {
    for (const col of COLUMNS) {
      for (const it of col.items) {
        const from = atSec("content-forms", it.line, -0.35);
        const to = from + 3.2; // 一条旁白大约讲这么久
        if (t >= from && t < to) return it.line;
      }
    }
    return null;
  })();

  const colWidth = 552;
  const sceneOnlyP = enter(frame, at("content-forms", "content-forms-11", -0.3));

  return (
    <div style={{position: "absolute", left: 96, right: 96, top: 214, display: "flex", flexDirection: "column", gap: 14}}>
      <div style={{display: "flex", gap: 36, alignItems: "flex-start"}}>
        {COLUMNS.map((col) => (
          <div key={col.domain} style={{width: colWidth, display: "flex", flexDirection: "column", gap: 7}}>
            <ColumnHeader col={col} frame={frame} width={colWidth} count={col.items.length} />
            {col.items.map((it) => (
              <ItemCard
                key={`${col.domain}-${it.name}`}
                item={it}
                domain={col.domain}
                frame={frame}
                active={activeLine === it.line}
                width={colWidth}
              />
            ))}
          </div>
        ))}
      </div>

      {/* 底部两条通用结论（不长，只占一行） */}
      <div
        style={{
          display: "flex",
          gap: 26,
          alignItems: "center",
          opacity: sceneOnlyP,
          transform: `translateY(${(1 - sceneOnlyP) * 10}px)`,
          borderTop: `1px solid ${COLORS.line}`,
          paddingTop: 10,
        }}
      >
        <span style={{fontFamily: FONT_SANS, fontSize: 17, color: COLORS.accent}}>
          命中才带 —— 没命中的条目不会出现在 messages 里
        </span>
        <span style={{fontFamily: FONT_SANS, fontSize: 17, color: COLORS.muted}}>
          例外只有 legendary：命中直接发出去，不进 messages
        </span>
      </div>
    </div>
  );
};
