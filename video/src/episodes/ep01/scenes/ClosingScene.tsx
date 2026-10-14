/**
 * Scene 10 · closing —— 收束（规格 §1.1、§18；用户反馈版式不美观，已重排）。
 *
 * 结尾必须让观众记住两句话：
 *   > 基础提示词负责让她始终是同一个人；behavior 负责让这个人在不同情景下做出不同反应。
 *   > 文件名模型看不见。模型真正读到的，是内容的文本形态、message role、注入位置和当前对话。
 *
 * 预告（用户要求）：**下一集主要讲记忆**。
 *
 * 版式（用户反馈"不美观"）：
 *   - 两条结论做成结构一致的左右对照行（文件卡 → 负责 → 结论），不再一长一短错落；
 *   - 第二部分（文件名看不见）用细线分隔并居中，role 标签等宽排列；
 *   - 预告独立成带边框的卡片，列出下一集的三个看点；
 *   - 立绘贴左边界内，不再有一半在界外。
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
import {FileCard} from "../components/FileCard";
import {HazelAvatar} from "../components/HazelAvatar";
import {at} from "../ep01-fixtures";

const ROLE_CHIPS = [
  {r: "system", c: COLORS.warn},
  {r: "user", c: COLORS.fact},
  {r: "assistant", c: COLORS.persona},
];

const NEXT_TOPICS = [
  {k: "长期记忆记什么", v: "层级与类型化条目"},
  {k: "怎么被取出来", v: "判别方式与注入位置"},
  {k: "怎么影响回复", v: "补的是现场，不是再造人格"},
];

/** 一条结论行：文件卡 → 「负责」→ 结论。两条结构完全一致，读起来才齐。 */
const ConclusionRow: React.FC<{
  frame: number;
  file: string;
  duty: string;
  delay: number;
}> = ({frame, file, duty, delay}) => {
  const p = enter(frame, delay);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 22,
        opacity: p,
        transform: `translateY(${(1 - p) * 14}px)`,
      }}
    >
      <FileCard name={file} domain="persona" width={400} fontSize={16} />
      <span style={{fontFamily: FONT_SANS, fontSize: 19, color: COLORS.muted, flex: "0 0 auto"}}>
        负责
      </span>
      <span
        style={{
          fontFamily: FONT_SANS,
          fontSize: 28,
          fontWeight: 700,
          color: DOMAIN_COLOR.persona,
          flex: "0 0 auto",
        }}
      >
        {duty}
      </span>
    </div>
  );
};

export const ClosingScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  const rowP = enter(frame, 0);
  const line2P = enter(frame, at("closing", "closing-3", -0.4));
  const nextP = enter(frame, at("closing", "closing-4", -0.6));

  // 纵向分布：内容必须整体落在**字幕安全区之上**。
  // 字幕盒在 bottom:40、高约 92~140px → 底部安全线约 y=900。
  // 所以 top 150 + height 745 = 895，正好贴住安全线。
  // （曾经 top160/height900 → 底边 1060，预告卡被字幕整个盖住。）
  return (
    <div
      style={{
        position: "absolute",
        left: 140,
        right: 140,
        top: 150,
        height: 745,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
      }}
    >
      {/* 结论一：两份人格核心的分工（两行同构） */}
      <div style={{display: "flex", flexDirection: "column", gap: 12, opacity: rowP}}>
        <div style={{fontFamily: FONT_MONO, fontSize: 15, color: COLORS.accent, letterSpacing: 1}}>
          结尾要记住的第一句
        </div>
        <ConclusionRow frame={frame} file="core/system_prompt.txt" duty="始终是同一个人" delay={0} />
        <ConclusionRow
          frame={frame}
          file="behavior/behaviors.json"
          duty="不同情景下做出不同反应"
          delay={sec(0.3)}
        />
      </div>

      {/* 结论二：文件名模型看不见 */}
      <div
        style={{
          paddingTop: 18,
          borderTop: `1px solid ${COLORS.line}`,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 12,
          opacity: line2P,
          transform: `translateY(${(1 - line2P) * 14}px)`,
        }}
      >
        <div style={{fontFamily: FONT_MONO, fontSize: 15, color: COLORS.accent, letterSpacing: 1}}>
          结尾要记住的第二句
        </div>
        <div style={{fontFamily: FONT_SANS, fontSize: 40, fontWeight: 700, color: COLORS.accent}}>
          文件名模型看不见
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          <span style={{fontFamily: FONT_SANS, fontSize: 22, color: COLORS.paper}}>
            它真正读到的，是
          </span>
          {["内容的文本形态", "message role", "注入位置", "当前对话"].map((x, i) => {
            const p = enter(frame, at("closing", "closing-3", -0.3) + i * 5);
            return (
              <span
                key={x}
                style={{
                  fontFamily: x === "message role" ? FONT_MONO : FONT_SANS,
                  fontSize: 22,
                  color: COLORS.paper,
                  padding: "5px 14px",
                  borderRadius: 5,
                  background: "rgba(255,255,255,0.045)",
                  border: `1px solid ${COLORS.line}`,
                  opacity: p,
                  transform: `translateY(${(1 - p) * 8}px)`,
                }}
              >
                {x}
              </span>
            );
          })}
        </div>
        {/* role 只有三种——和 messages 的协议字段对齐 */}
        <div style={{display: "flex", alignItems: "center", gap: 12}}>
          <span style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.dim}}>role 只有三种：</span>
          {ROLE_CHIPS.map((x, i) => {
            const p = enter(frame, at("closing", "closing-3", 0.2) + i * 5);
            return (
              <span
                key={x.r}
                style={{
                  fontFamily: FONT_MONO,
                  fontSize: 18,
                  color: x.c,
                  padding: "3px 14px",
                  borderRadius: 4,
                  border: `1px solid ${x.c}88`,
                  background: `${x.c}16`,
                  opacity: p,
                }}
              >
                {x.r}
              </span>
            );
          })}
        </div>
      </div>

      {/* 下一集预告：记忆专题 */}
      <div
        style={{
          alignSelf: "center",
          opacity: nextP,
          transform: `translateY(${(1 - nextP) * 12}px)`,
          padding: "16px 28px",
          borderRadius: 9,
          border: `1px solid ${COLORS.line}`,
          background: "rgba(255,255,255,0.028)",
          display: "flex",
          flexDirection: "column",
          gap: 11,
          minWidth: 1240,
          boxSizing: "border-box",
        }}
      >
        <div style={{display: "flex", alignItems: "baseline", gap: 14}}>
          <span
            style={{
              fontFamily: FONT_MONO,
              fontSize: 15,
              color: COLORS.accent,
              letterSpacing: 1.4,
              flex: "0 0 auto",
            }}
          >
            下一集
          </span>
          <span style={{fontFamily: FONT_SANS, fontSize: 23, color: COLORS.paper}}>
            这解释了她为什么会这样说。下一集再看，
            <span style={{color: COLORS.accent}}>她为什么还记得你之前说过什么。</span>
          </span>
        </div>
        <div style={{display: "flex", gap: 34}}>
          {NEXT_TOPICS.map((t, i) => {
            const p = enter(frame, at("closing", "closing-4", -0.2) + i * 6);
            return (
              <div
                key={t.k}
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: 8,
                  opacity: p,
                  transform: `translateX(${(1 - p) * 10}px)`,
                }}
              >
                <span style={{fontFamily: FONT_SANS, fontSize: 17, color: COLORS.paper}}>{t.k}</span>
                <span style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.dim}}>{t.v}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* 立绘：放在右下角的空白处。
          曾经贴左边（left:-18）—— 但内容是从 left:140 起的，立绘会压到
          两张结论卡上（用户反馈"不美观"包含这一处）。右下角在预告卡之下是空的，
          既不遮挡文字，也不会有一半在界外。 */}
      <HazelAvatar
        frame={frame}
        delayFrames={sec(0.3)}
        width={200}
        mood="soft"
        tone={DOMAIN_COLOR.persona}
        opacity={lin(frame, [sec(0.3), sec(1.2)], [0, 0.9])}
        style={{position: "absolute", right: 0, top: 372}}
      />
    </div>
  );
};
