/**
 * Scene 04 · files-to-messages —— 关键转折（规格 §6）。
 *
 * 论点：
 *   文件 = 人维护内容的单位
 *   messages = 模型实际接收的单位
 *
 * 画面：文件卡进入一个 "formatter" 区域，出来后变成 messages 卡。
 * **禁止画成"文件直接飞进大模型"**——中间必须出现"读取、判断、格式化"三个动作。
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
import {MessageCard} from "../components/MessageCard";
import {personaStructure} from "../ep01-fixtures";

const STEPS = ["读取", "判断", "格式化"];

/** formatter 区域：文件卡进去、messages 出来。 */
const Formatter: React.FC<{frame: number; tone: string}> = ({frame, tone}) => {
  const p = enter(frame, sec(1.6));
  // 三个动作依次点亮，形成"文件 → messages"的过渡
  const stepAt = [sec(3.0), sec(4.2), sec(5.4)];

  return (
    <div
      style={{
        width: 330,
        opacity: p,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 14,
        padding: "20px 18px",
        borderRadius: 9,
        background: COLORS.panel,
        border: `1px solid ${tone}`,
        boxShadow: `0 0 0 1px ${tone}33`,
      }}
    >
      <div style={{fontFamily: FONT_MONO, fontSize: 14, color: tone, letterSpacing: 1.4}}>
        FORMATTER
      </div>

      <svg width="60" height="34" viewBox="0 0 60 34">
        <line x1="4" y1="17" x2="44" y2="17" stroke={tone} strokeWidth="2" />
        <path d="M42 9 L54 17 L42 25" fill="none" stroke={tone} strokeWidth="2" strokeLinejoin="round" />
      </svg>

      <div style={{display: "flex", flexDirection: "column", gap: 9, width: "100%"}}>
        {STEPS.map((s, i) => {
          const sp = enter(frame, stepAt[i]);
          return (
            <div
              key={s}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                opacity: 0.32 + 0.68 * sp,
                transform: `translateX(${(1 - sp) * -12}px)`,
              }}
            >
              <span
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: 11,
                  border: `1.5px solid ${sp > 0.4 ? tone : COLORS.line}`,
                  color: sp > 0.4 ? tone : COLORS.dim,
                  fontFamily: FONT_MONO,
                  fontSize: 12,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flex: "0 0 auto",
                }}
              >
                {i + 1}
              </span>
              <span
                style={{
                  fontFamily: FONT_SANS,
                  fontSize: 21,
                  color: sp > 0.4 ? COLORS.paper : COLORS.muted,
                }}
              >
                {s}
              </span>
            </div>
          );
        })}
      </div>

      <div
        style={{
          marginTop: 4,
          fontFamily: FONT_SANS,
          fontSize: 13,
          color: COLORS.dim,
          textAlign: "center",
          lineHeight: 1.4,
          opacity: enter(frame, sec(6.0)),
        }}
      >
        路径与文件名到此为止
        <br />
        不再往下传
      </div>
    </div>
  );
};

export const FilesToMessagesScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  const tone = COLORS.accent;
  const arrowP = lin(frame, [sec(6.4), sec(7.4)], [0, 1]);

  return (
    <div style={{position: "absolute", left: 96, right: 96, top: 250, display: "flex", alignItems: "center", gap: 40}}>
      {/* 左：文件（人维护内容的单位） */}
      <div style={{width: 400, display: "flex", flexDirection: "column", gap: 12}}>
        <div style={{fontFamily: FONT_MONO, fontSize: 15, color: DOMAIN_COLOR.persona, letterSpacing: 1}}>
          文件 = 人维护内容的单位
        </div>
        <FileCard
          name="persona/core/system_prompt.txt"
          domain="persona"
          progress={enter(frame, 0)}
          width={400}
        />
        <FileCard
          name="persona/behavior/behaviors.json"
          note="被选中的那一条"
          domain="persona"
          progress={enter(frame, sec(0.3))}
          width={400}
        />
        <div
          style={{
            marginTop: 6,
            fontFamily: FONT_SANS,
            fontSize: 15,
            lineHeight: 1.5,
            color: COLORS.muted,
            opacity: enter(frame, sec(0.8)),
          }}
        >
          模型不知道这些文件叫什么，
          <br />
          也不知道这段文字原来在哪个目录。
        </div>
      </div>

      {/* 中：formatter */}
      <Formatter frame={frame} tone={tone} />

      {/* 右：messages（模型实际接收的单位） */}
      <div style={{flex: 1, display: "flex", flexDirection: "column", gap: 12, opacity: arrowP}}>
        <div style={{fontFamily: FONT_MONO, fontSize: 15, color: DOMAIN_COLOR.fact, letterSpacing: 1}}>
          messages = 模型实际接收的单位
        </div>
        <MessageCard
          role="system"
          header="首条 system message"
          text={`system_prompt 全文（${personaStructure.systemPrompt.chars} 字）`}
          progress={enter(frame, sec(7.0))}
          width={620}
        />
        <MessageCard
          role="system"
          header="【当前情境下的行为指令】"
          text={`${personaStructure.behaviors.focus.name} · 回应模式 + 真人示范`}
          progress={enter(frame, sec(7.6))}
          width={620}
        />
        <MessageCard
          role="user"
          text="你今天唱得特别好，我真的很喜欢。"
          progress={enter(frame, sec(8.2))}
          width={620}
        />
        <div
          style={{
            marginTop: 6,
            display: "flex",
            flexDirection: "column",
            gap: 5,
            opacity: enter(frame, sec(8.8)),
          }}
        >
          {[
            "模型能看到的：文本内容",
            "模型能看到的：system / user / assistant 这三种 role",
            "模型能看到的：内容在 messages 里的位置",
            "模型能看到的：当前消息与历史上下文",
          ].map((t) => (
            <div
              key={t}
              style={{
                fontFamily: FONT_SANS,
                fontSize: 16,
                color: COLORS.paper,
                paddingLeft: 10,
                borderLeft: `2px solid ${DOMAIN_COLOR.fact}`,
              }}
            >
              {t}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
