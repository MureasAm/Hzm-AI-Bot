/**
 * Scene 01 · cold-open —— 三板真实回复（规格 §4）。
 *
 * 段落任务：直接证明"同一个模型因上下文不同而改变"。
 * 旁白只完成三件事：① 三栏是同一个模型；② 改变的是它当时看到的上下文；
 * ③ 提出本集问题。**不在开场解释 RAG、embedding 或 JSON**。
 *
 * 三栏顺序：裸模型 → 仅基础提示词 → 基础提示词 + behavior。
 * 数据：persona-comparison.json（Phase 0 真实生成，每版 5 次）。
 *
 * 版式注意（用户反馈过两轮）：
 *   · 回复必须带说话人标签（`灰泽满：`），否则看不出是谁在说；
 *   · 进度条已挪到画面顶端，标题区不再被压；
 *   · 底部有一条**证据条**显示 raw → cleaned —— `clean_reply` 会把
 *     `……` 归一成 `，`，只显示 cleaned 会让观众以为模型就是那么写的。
 *     它必须整体落在字幕安全区（bottom:40 起、约 150px 高）之上。
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
import {personaComparison, selectedReplyOf, at} from "../ep01-fixtures";
import {
  PersonaComparison,
  RawVsCleaned,
  SharedModelBar,
  type ComparisonColumn,
} from "../components/PersonaComparison";

export const ColdOpenScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  const c = personaComparison;

  /**
   * 展示用回复：裁掉**开头的自称**。
   *
   * 仅基础提示词那版的实际输出是 `"灰泽满……你今天是不是吃错药了…"`——
   * 她用第三人称自称是风格，但紧接着卡片上的说话人标签 `灰泽满：` 连读，
   * 就变成"灰泽满：灰泽满……"，读起来像在喊别人。
   *
   * 只裁**展示文本**；`raw` 与 `cleaned` 原样保留在下方证据条里，
   * 并在卡片下标明裁了什么（规格 §4.3 不改写模型输出，这里不藏）。
   */
  const trimSelfRef = (text: string): {reply: string; note?: string} => {
    const m = text.match(/^(灰泽满|hzm)(……|…|，|,|：|:)+/i);
    if (!m) {
      return {reply: text};
    }
    return {
      reply: text.slice(m[0].length),
      note: `画面裁掉了开头的自称「${m[0]}」（原始输出见下方证据条）`,
    };
  };

  const sysPromptReply = trimSelfRef(selectedReplyOf("system_prompt"));

  const columns: ComparisonColumn[] = [
    {
      id: "bare",
      label: "裸模型",
      sublabel: "只给用户消息",
      context: ["（没有任何 system 内容）"],
      reply: selectedReplyOf("bare"),
      tags: [{text: "礼貌回应", color: COLORS.muted}],
    },
    {
      id: "system_prompt",
      label: "仅基础提示词",
      sublabel: "system_prompt + user",
      context: [`system_prompt.txt（${c.systemPromptSource.chars} 字）`],
      reply: sysPromptReply.reply,
      displayNote: sysPromptReply.note,
      tags: [
        {text: "稳定骨架", color: DOMAIN_COLOR.fact},
        {text: "压低评价", color: DOMAIN_COLOR.fact},
      ],
    },
    {
      id: "system_prompt_behavior",
      label: "基础提示词 + behavior",
      sublabel: "system_prompt + behavior + user",
      context: [
        `system_prompt.txt（${c.systemPromptSource.chars} 字）`,
        `【当前情境下的行为指令】${c.behaviorSource.name}`,
      ],
      reply: selectedReplyOf("system_prompt_behavior"),
      tags: [
        {text: "完整反应动作", color: DOMAIN_COLOR.persona},
        {text: "嘴硬 + 别扭 + 软话", color: DOMAIN_COLOR.persona},
      ],
    },
  ];

  // 三栏依次出现（规格 §4.4：裸模型 → 基础提示词 → 基础提示词加 behavior）。
  // 锚点**跟着旁白行走**（at() 从 ep01-narration.json 取时刻），
  // 不写死秒数——时间轴每次重算后画面仍然对齐（见 ep01-fixtures.ts 的 at()）。
  const thresholds: [number, number, number] = [
    at("cold-open", "cold-open-4", -0.4),
    at("cold-open", "cold-open-6", -0.4),
    at("cold-open", "cold-open-7", -0.4),
  ];
  // 回复逐句出现，与旁白/语音同步（锚点同样跟着旁白行）
  const reveal = (lineId: string, offsetIn: number, offsetOut: number) =>
    lin(frame, [at("cold-open", lineId, offsetIn), at("cold-open", lineId, offsetOut)], [0, 1]);
  const questionP = enter(frame, at("cold-open", "cold-open-8", -0.6));

  // 证据条：三栏选中那条的 raw 与 cleaned（raw 只在两者不同时才需要看）
  const evidence = c.variants.map((v) => {
    const r = v.runs.find((x) => x.run === v.selectedRun);
    return {label: v.label, raw: r?.raw ?? "", cleaned: r?.cleaned ?? ""};
  });
  const evidenceP = enter(frame, at("cold-open", "cold-open-8", 0.2));

  return (
    <div
      style={{
        position: "absolute",
        left: 76,
        right: 76,
        top: 210,
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <SharedModelBar frame={frame} model={c.model} params={c.params} input={c.input} />

      <PersonaComparison
        frame={frame}
        userInput={c.input}
        model={c.model}
        params={c.params}
        columns={columns}
        thresholds={thresholds}
        replyReveal={[
          reveal("cold-open-5", 0, 2.4),
          reveal("cold-open-6", 0, 2.6),
          reveal("cold-open-7", 0, 2.6),
        ]}
        tagReveal={[
          reveal("cold-open-5", 0, 0.6),
          reveal("cold-open-6", 0, 0.6),
          reveal("cold-open-8", 0, 0.6),
        ]}
      />

      {/* 本集问题（规格 §4.5 第 3 件事）：放在字幕安全区之上 */}
      <div
        style={{
          marginTop: 2,
          textAlign: "center",
          opacity: questionP,
          transform: `translateY(${(1 - questionP) * 12}px)`,
        }}
      >
        <span style={{fontFamily: FONT_SANS, fontSize: 27, fontWeight: 700, color: COLORS.accent}}>
          这些文件，到底怎样走到回复里？
        </span>
        <span style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.muted, marginLeft: 18}}>
          {c.variants.reduce((a, v) => a + v.runs.length, 0)} 条原始输出全部保留 · 未做人工改写
        </span>
      </div>

      {/* 证据条：raw vs cleaned（诚实呈现 clean_reply 的归一） */}
      <div style={{opacity: evidenceP}}>
        <div
          style={{
            fontFamily: FONT_MONO,
            fontSize: 13,
            color: COLORS.warn,
            marginBottom: 5,
          }}
        >
          模型原始输出 → clean_reply 之后（画面上的回复用的是 cleaned）
        </div>
        <RawVsCleaned frame={frame} variants={evidence} />
      </div>
    </div>
  );
};
