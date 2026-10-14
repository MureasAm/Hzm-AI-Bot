/**
 * Scene 03 · behavior-core —— 情景行为（规格 §5.2、§5.3）。
 *
 * 固定三个术语：情景 → 反应 → 例句。
 * 必须解释 few-shot，但不讲成"复制几个句子"：
 *   - few-shot 给的是示范；示例必须绑定正确情景；behavior 不是固定回复表；
 *   - 示例影响节奏、转折和动作形态，最终回复仍按当前用户消息生成。
 *
 * §5.3：历史演变只占一句，旧文件卡缩成两摞分别并入 system_prompt 与 behaviors，
 * 不做"垃圾桶式嘲讽"。
 *
 * 版式（用户反馈：例句超出底部边界、立绘大半在界外）：
 *   - 例句不再压在"情景/反应"下面，而是**右栏**，与左栏并排 → 总高从 1000+ 降到 ~450；
 *   - 立绘宽度与位置收进画面内（原来是 right:-96 / bottom:-210，几乎全在界外）。
 *   可用内容高度：top 218 起、字幕安全区约在 890 → 约 670px。
 */
import React from "react";
import {
  COLORS,
  DOMAIN_COLOR,
  FONT_MONO,
  FONT_SANS,
  enter,
  fadeInOut,
  lin,
  sec,
} from "../theme";
import {personaStructure, at} from "../ep01-fixtures";
import {BehaviorAnatomy} from "../components/BehaviorAnatomy";
import {FileCard} from "../components/FileCard";
import {HazelAvatar} from "../components/HazelAvatar";

const OLD_FILES = ["traits.json", "styles.json", "voice_samples.json", "phrases.json"];

export const BehaviorCoreScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  const b = personaStructure.behaviors.focus;
  const tone = DOMAIN_COLOR.persona;

  // 三段展开：情景(behavior-core-3) → 反应(behavior-core-4) → 例句(behavior-core-6，逐条)
  // 例句逐条揭示，与旁白同步
  const samplesReveal = lin(
    frame,
    [at("behavior-core", "behavior-core-4", -0.3), at("behavior-core", "behavior-core-5", -0.5)],
    [0, b.samples.length],
  );

  // §5.3 历史演变：只占 15~25 秒的一段，收在场景尾部
  const historyP = enter(frame, at("behavior-core", "behavior-core-10", -0.4));

  return (
    <div style={{position: "absolute", left: 76, right: 76, top: 214}}>
      {/* 上半：BehaviorAnatomy（左=情景/反应，右=例句） */}
      <BehaviorAnatomy
        frame={frame}
        name={b.name}
        trigger={b.trigger}
        response={b.response}
        samples={b.samples}
        samplesReveal={samplesReveal}
        stageWidth={880}
        sampleWidth={520}
      />

      {/* 下半：few-shot 说明 + 历史演变（横向两栏，压在 ~250px 内） */}
      <div style={{display: "flex", gap: 44, marginTop: 22, alignItems: "flex-start"}}>
        <div
          style={{
            width: 880,
            padding: "12px 16px",
            borderRadius: 7,
            background: COLORS.panel,
            border: `1px solid ${COLORS.line}`,
            opacity: enter(frame, at("behavior-core", "behavior-core-7", -0.4)),
          }}
        >
          <div style={{display: "flex", alignItems: "baseline", gap: 12, marginBottom: 7}}>
            <span style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.accent}}>few-shot</span>
            <span style={{fontFamily: FONT_SANS, fontSize: 14, color: COLORS.muted}}>
              例句不是"让模型复制的句子"
            </span>
          </div>
          <div style={{display: "flex", flexWrap: "wrap", gap: "5px 24px"}}>
            {[
              "给的是示范，不是让模型复制句子",
              "示范必须绑在正确的情景上",
              "behavior 不是固定回复表",
              "影响节奏、转折和动作形态",
              "最终回复仍按当前用户消息生成",
            ].map((t, i) => {
              const p = enter(frame, at("behavior-core", "behavior-core-7", -0.2) + i * 6);
              return (
                <div
                  key={t}
                  style={{
                    fontFamily: FONT_SANS,
                    fontSize: 15,
                    lineHeight: 1.45,
                    color: COLORS.paper,
                    opacity: p,
                    transform: `translateX(${(1 - p) * 10}px)`,
                    paddingLeft: 10,
                    borderLeft: `2px solid ${tone}`,
                    flex: "0 0 auto",
                    maxWidth: 400,
                  }}
                >
                  {t}
                </div>
              );
            })}
          </div>
        </div>

        {/* §5.3 历史演变：旧文件卡缩成两摞并入两个去向 */}
        <div
          style={{
            flex: 1,
            padding: "12px 16px",
            borderRadius: 7,
            background: "rgba(255,255,255,0.022)",
            border: `1px dashed ${COLORS.miss}`,
            opacity: historyP,
            transform: `translateY(${(1 - historyP) * 12}px)`,
          }}
        >
          <div style={{fontFamily: FONT_MONO, fontSize: 13, color: COLORS.dim, marginBottom: 8}}>
            旧的四个文件（2026-10-04 拆分）
          </div>
          <div style={{display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10}}>
            {OLD_FILES.map((f, i) => {
              const p = enter(frame, at("behavior-core", "behavior-core-10", 0.4) + i * 4);
              return (
                <span
                  key={f}
                  style={{
                    fontFamily: FONT_MONO,
                    fontSize: 13,
                    color: COLORS.muted,
                    padding: "3px 8px",
                    borderRadius: 4,
                    border: `1px solid ${COLORS.miss}`,
                    background: "rgba(255,255,255,0.02)",
                    opacity: p * (1 - 0.35 * lin(
                      frame,
                      [at("behavior-core", "behavior-core-10", 0.4), at("behavior-core", "behavior-core-10", 2.4)],
                      [0, 1],
                    )),
                  }}
                >
                  {f}
                </span>
              );
            })}
          </div>
          <div style={{display: "flex", flexDirection: "column", gap: 6}}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                opacity: lin(
                  frame,
                  [at("behavior-core", "behavior-core-10", 0.8), at("behavior-core", "behavior-core-10", 2.1)],
                  [0, 1],
                ),
              }}
            >
              <span style={{fontFamily: FONT_SANS, fontSize: 14, color: COLORS.muted}}>稳定内容 →</span>
              <FileCard name="core/system_prompt.txt" domain="persona" width={300} fontSize={14} />
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                opacity: lin(
                  frame,
                  [at("behavior-core", "behavior-core-10", 1.4), at("behavior-core", "behavior-core-10", 2.7)],
                  [0, 1],
                ),
              }}
            >
              <span style={{fontFamily: FONT_SANS, fontSize: 14, color: COLORS.muted}}>真人示范 →</span>
              <FileCard name="behavior/behaviors.json" domain="persona" width={300} fontSize={14} />
            </div>
          </div>
        </div>
      </div>

      {/* 立绘：例句发声时短暂出现（规格 §2.2 第 4 条）。
          位置收进画面内 —— 原来 right:-96 / bottom:-210 让立绘大半在界外；
          现在贴在右下角的空白处，不与例句面板和字幕重叠。 */}
      <HazelAvatar
        frame={frame}
        delayFrames={at("behavior-core", "behavior-core-4", -0.4)}
        width={200}
        mood="smug"
        opacity={fadeInOut(
          frame,
          at("behavior-core", "behavior-core-4", -0.4),
          at("behavior-core", "behavior-core-5", 0.4),
          at("behavior-core", "behavior-core-6", 0.2),
        )}
        style={{position: "absolute", right: 8, top: 30}}
      />
    </div>
  );
};
