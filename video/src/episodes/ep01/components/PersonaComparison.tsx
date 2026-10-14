/**
 * PersonaComparison —— 开场三板对照（规格 §4.4、§12.4）。
 *
 * 约束：
 *   - 三栏固定等宽布局；
 *   - 同一用户消息复制到三栏，三栏用**同一个模型标识**；
 *   - 回复逐句出现（配合各行旁白/语音）；
 *   - 突出"礼貌回应 / 稳定骨架 / 完整反应动作"，**不显示胜负图标**（不用红叉绿勾）；
 *   - 三栏出现顺序：裸模型 → 仅基础提示词 → 基础提示词 + behavior。
 *
 * 数据全部来自 video/data/ep01/persona-comparison.json（Phase 0 真实生成）。
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
import {KeywordTag} from "./KeywordTag";
import {MessageCard} from "./MessageCard";

export type ComparisonColumn = {
  id: string;
  label: string;
  sublabel: string;
  /** 该栏注入的上下文摘要（按真实 messages 组装） */
  context: string[];
  /** 选中的回复（raw 的展示副本，见 displayNote） */
  reply: string;
  /** 差异关键词（不判优） */
  tags: {text: string; color: string}[];
  /**
   * 展示层裁剪说明。空/未设 = 画面显示的与模型输出完全一致。
   *
   * 为什么需要这个字段（规格 §4.3 说"不人工改写模型回复"）：
   * 仅基础提示词那版的实际输出是
   *   `"灰泽满……你今天是不是吃错药了，突然夸得这么认真。"`
   * 开头那个自称紧接着卡片上的说话人标签（`灰泽满：`）连读，
   * 就成了"灰泽满：灰泽满……"，读起来像在喊别人。
   *
   * 处置：**只裁掉展示文本开头的自称**，并在卡片上用一行小字写明裁了什么；
   * 原始输出（raw）与 clean_reply 结果（cleaned）仍然在下方证据条里完整呈现。
   * 这不是改写模型输出，是排版取舍 + 明示。
   */
  displayNote?: string;
};

export const PersonaComparison: React.FC<{
  frame: number;
  userInput: string;
  model: string;
  params: {temperature: number; frequency_penalty: number; max_tokens: number};
  columns: ComparisonColumn[];
  /** 每栏出现进度阈值（帧），第 i 栏在 thresholds[i] 后进入 */
  thresholds: [number, number, number];
  /** 每栏回复展开进度 (0~1) */
  replyReveal: [number, number, number];
  /** 每栏标签展开进度 */
  tagReveal: [number, number, number];
  width?: number;
}> = ({
  frame,
  userInput,
  model,
  params,
  columns,
  thresholds,
  replyReveal,
  tagReveal,
  width = 520,
}) => {
  return (
    <div style={{display: "flex", gap: 34, justifyContent: "center", alignItems: "flex-start"}}>
      {columns.map((c, i) => {
        const p = enter(frame, thresholds[i]);
        const active = frame >= thresholds[i];
        const tone = i === 0 ? COLORS.muted : i === 1 ? DOMAIN_COLOR.fact : DOMAIN_COLOR.persona;
        return (
          <div
            key={c.id}
            style={{
              width,
              opacity: p,
              transform: `translateY(${(1 - p) * 24}px)`,
              display: "flex",
              flexDirection: "column",
              gap: 12,
              boxSizing: "border-box",
            }}
          >
            {/* 栏头：从出现时就直接标注，不制造猜谜（规格 §4.2） */}
            <div
              style={{
                padding: "11px 15px",
                borderRadius: 7,
                background: active ? `${tone}16` : "rgba(255,255,255,0.02)",
                border: `1px solid ${active ? tone : COLORS.line}`,
                borderTop: `3px solid ${tone}`,
              }}
            >
              <div style={{fontFamily: FONT_SANS, fontSize: 23, fontWeight: 700, color: active ? tone : COLORS.muted}}>
                {c.label}
              </div>
              <div style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.muted, marginTop: 2}}>
                {c.sublabel}
              </div>
            </div>

            {/* 上下文摘要：这栏模型额外看到了什么 */}
            <div style={{display: "flex", flexDirection: "column", gap: 4, minHeight: 62}}>
              {c.context.map((t, k) => {
                const cp = lin(frame, [thresholds[i] + k * 4, thresholds[i] + k * 4 + 12], [0, 1]);
                return (
                  <div
                    key={t}
                    style={{
                      fontFamily: FONT_MONO,
                      fontSize: 14,
                      color: COLORS.paper,
                      padding: "4px 10px",
                      borderRadius: 4,
                      background: "rgba(255,255,255,0.034)",
                      borderLeft: `3px solid ${tone}`,
                      opacity: cp,
                    }}
                  >
                    {t}
                  </div>
                );
              })}
            </div>

            {/* 用户消息：三栏完全相同 */}
            <MessageCard
              role="user"
              header="用户："
              text={userInput}
              width={width}
              progress={lin(frame, [thresholds[i], thresholds[i] + 10], [0.25, 1])}
              compact
            />

            {/*
              回复：必须有说话人标签（用户反馈：光有文本看不出谁在说）。
              「灰泽满：」这个前缀和 behavior 例句里的注入格式是同一套
              （`_format_behavior_rule` 用 "粉丝说：… → 灰泽满：…"），
              所以画面里见到它就等于见到模型实际读到的样子。

              `role="assistant"` 会把 role 名渲染成 "assistant"——那是**协议字段**，
              不是人看的称呼，所以这里改用 `role="灰泽满"`，真正对应 messages 里的
              assistant 位置由卡片颜色与下面的 raw 对照说明。
            */}
            <MessageCard
              role="灰泽满"
              text={c.reply}
              tone={tone}
              width={width}
              progress={enter(frame, thresholds[i] + 8)}
              style={{
                clipPath: `inset(0 ${(1 - lin(replyReveal[i], [0, 1], [0, 1])) * 100}% 0 0)`,
              }}
            />

            {/* 裁掉了什么，明写出来（不然就是偷偷改写模型输出） */}
            {c.displayNote ? (
              <div
                style={{
                  fontFamily: FONT_SANS,
                  fontSize: 13,
                  color: COLORS.warn,
                  opacity: lin(replyReveal[i], [0.6, 1], [0, 1]),
                  marginTop: -4,
                }}
              >
                {c.displayNote}
              </div>
            ) : null}

            {/* 关键词标注（不判优） */}
            <div style={{display: "flex", gap: 8, flexWrap: "wrap", minHeight: 30}}>
              {c.tags.map((t) => (
                <KeywordTag key={t.text} color={t.color} progress={lin(tagReveal[i], [0, 1], [0, 1])}>
                  {t.text}
                </KeywordTag>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};

/**
 * 证据条：把三栏选中那条回复的 **raw → cleaned** 放在一起。
 *
 * 为什么需要（用户反馈引出的一处失真）：raw 是
 * `"灰泽满……你今天是不是吃错药了，突然夸得这么认真。"`，
 * 而 `clean_reply` 把省略号归一成了逗号（`……` → `，`）。
 * 画面如果只显示 cleaned，观众会以为模型就是那么写的。
 * 两条并排显示，才是"未做人工改写"的诚实呈现。
 */
export const RawVsCleaned: React.FC<{
  frame: number;
  variants: {label: string; raw: string; cleaned: string}[];
}> = ({frame, variants}) => {
  const p = enter(frame, 0);
  return (
    <div
      style={{
        display: "flex",
        gap: 16,
        opacity: p,
        transform: `translateY(${(1 - p) * 8}px)`,
        flexWrap: "wrap",
      }}
    >
      {variants.map((v, i) => {
        const same = v.raw === v.cleaned;
        return (
          <div
            key={v.label}
            style={{
              border: `1px solid ${COLORS.line}`,
              borderRadius: 6,
              background: "rgba(255,255,255,0.022)",
              padding: "7px 12px",
              minWidth: 420,
              maxWidth: 620,
            }}
          >
            <div style={{display: "flex", gap: 10, alignItems: "baseline", marginBottom: 3}}>
              <span style={{fontFamily: FONT_SANS, fontSize: 14, color: COLORS.muted}}>
                {v.label}
              </span>
              <span
                style={{
                  fontFamily: FONT_MONO,
                  fontSize: 12,
                  color: same ? COLORS.dim : COLORS.warn,
                }}
              >
                {same ? "raw = cleaned" : "raw → cleaned"}
              </span>
            </div>
            <div style={{fontFamily: FONT_MONO, fontSize: 13, color: COLORS.dim, lineHeight: 1.4}}>
              {v.raw}
            </div>
            {!same ? (
              <div
                style={{
                  fontFamily: FONT_MONO,
                  fontSize: 13,
                  color: COLORS.paper,
                  lineHeight: 1.4,
                  marginTop: 3,
                }}
              >
                {v.cleaned}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
};

/** 三栏共用的模型标识条（强调"同一个模型，只是上下文不同"）。 */
export const SharedModelBar: React.FC<{
  frame: number;
  model: string;
  params: {temperature: number; frequency_penalty: number; max_tokens: number};
  input: string;
}> = ({frame, model, params, input}) => {
  const p = enter(frame, 0);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 22,
        opacity: p,
        transform: `translateY(${(1 - p) * -12}px)`,
        padding: "10px 18px",
        border: `1px solid ${COLORS.line}`,
        borderRadius: 7,
        background: COLORS.panel,
      }}
    >
      <span style={{fontFamily: FONT_MONO, fontSize: 16, color: COLORS.accent}}>同一个模型</span>
      <span style={{fontFamily: FONT_MONO, fontSize: 16, color: COLORS.paper}}>{model}</span>
      <span style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.muted}}>
        temperature {params.temperature} · frequency_penalty {params.frequency_penalty} · max_tokens{" "}
        {params.max_tokens} · thinking disabled
      </span>
      <span style={{fontFamily: FONT_SANS, fontSize: 16, color: COLORS.muted, marginLeft: "auto"}}>
        输入：{input}
      </span>
    </div>
  );
};

/** 章节转场用的小结（开场收尾 / closing 复用）。 */
export const ComparisonSummary: React.FC<{
  frame: number;
  lines: string[];
  delay?: number;
  color?: string;
}> = ({frame, lines, delay = 0, color = COLORS.accent}) => (
  <div style={{display: "flex", flexDirection: "column", gap: 8}}>
    {lines.map((t, i) => {
      const p = enter(frame, delay + sec(0.2) * i);
      return (
        <div
          key={t}
          style={{
            fontFamily: FONT_SANS,
            fontSize: 22,
            color,
            opacity: p,
            transform: `translateX(${(1 - p) * -14}px)`,
          }}
        >
          {t}
        </div>
      );
    })}
  </div>
);
