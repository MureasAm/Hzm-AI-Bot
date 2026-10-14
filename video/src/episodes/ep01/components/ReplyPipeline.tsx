/**
 * ReplyPipeline —— 输出侧的**只预告**画面（规格 §11）。
 *
 * 本集必须展示的正确顺序：
 *   最终 messages
 *     → generate_reply
 *     → 防复读检测，必要时重新生成
 *     → clean_reply
 *     → should_voice
 *         ├─ 成功：GPT-SoVITS → QQ 语音条
 *         └─ 不走语音：split_reply → 多个 QQ 文字气泡
 *     → 写回短期/长期记忆
 *
 * 禁止画成"先拆文字，再顺便生成语音"（语音成功时不会继续发送拆分文字）。
 * 不展开 split_reply 正则、TTS HTTP 请求、缓存或重试逻辑。
 */
import React from "react";
import {COLORS, FONT_MONO, FONT_SANS, enter, lin, sec} from "../theme";

const Step: React.FC<{
  index: number;
  label: string;
  note?: string;
  frame: number;
  delay: number;
  color?: string;
  active?: boolean;
  width?: number;
}> = ({index, label, note, frame, delay, color = COLORS.line, active = false, width = 520}) => {
  const p = enter(frame, delay);
  return (
    <div
      style={{
        width,
        display: "flex",
        alignItems: "center",
        gap: 12,
        opacity: p,
        transform: `translateY(${(1 - p) * 14}px)`,
      }}
    >
      <span
        style={{
          width: 26,
          height: 26,
          borderRadius: 13,
          border: `1.5px solid ${active ? color : COLORS.line}`,
          color: active ? color : COLORS.muted,
          fontFamily: FONT_MONO,
          fontSize: 14,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flex: "0 0 auto",
        }}
      >
        {index}
      </span>
      <div style={{minWidth: 0}}>
        <div
          style={{
            fontFamily: FONT_MONO,
            fontSize: 21,
            color: active ? COLORS.paper : COLORS.muted,
          }}
        >
          {label}
        </div>
        {note ? (
          <div style={{fontFamily: FONT_SANS, fontSize: 15, color: COLORS.muted, marginTop: 2}}>
            {note}
          </div>
        ) : null}
      </div>
    </div>
  );
};

const Connector: React.FC<{frame: number; delay: number; height?: number}> = ({
  frame,
  delay,
  height = 16,
}) => (
  <div
    style={{
      width: 2,
      height,
      background: COLORS.line,
      marginLeft: 12,
      opacity: enter(frame, delay),
    }}
  />
);

export const ReplyPipeline: React.FC<{
  frame: number;
  /** 真实原始回复 */
  rawReply: string;
  /** 真实清洗结果 */
  cleanedReply: string;
  /** 实际投递方式（来自 trace 结果） */
  delivery: {mode: string; shouldVoice: boolean; parts: string[]};
  /** 是否走到"写回记忆"（本集展示到这一步就够了） */
  showWriteBack?: boolean;
  width?: number;
}> = ({frame, rawReply, cleanedReply, delivery, showWriteBack = true, width = 560}) => {
  const voice = delivery.shouldVoice;

  return (
    <div style={{display: "flex", gap: 54, alignItems: "flex-start"}}>
      {/* 左：流程 */}
      <div style={{display: "flex", flexDirection: "column"}}>
        <Step index={1} label="最终 messages" note="15 段 system + user" frame={frame} delay={0} active width={width} />
        <Connector frame={frame} delay={sec(0.2)} />
        <Step index={2} label="generate_reply" note="真实模型生成" frame={frame} delay={sec(0.3)} active color={COLORS.accent} width={width} />
        <Connector frame={frame} delay={sec(0.5)} />
        <Step
          index={3}
          label="防复读检测"
          note="重复就换动作重新生成"
          frame={frame}
          delay={sec(0.6)}
          color={COLORS.warn}
          width={width}
        />
        <Connector frame={frame} delay={sec(0.8)} />
        <Step
          index={4}
          label="clean_reply"
          note="去多余括号前缀 · 换行归一"
          frame={frame}
          delay={sec(0.9)}
          color={COLORS.fact}
          width={width}
        />
        <Connector frame={frame} delay={sec(1.1)} />
        <Step
          index={5}
          label="should_voice"
          note="成句(≥30字)朗读语音条；否则文字"
          frame={frame}
          delay={sec(1.2)}
          color={COLORS.persona}
          width={width}
        />

        {/* 二选一分支：明确画成互斥 */}
        <div style={{display: "flex", gap: 26, marginLeft: 38, marginTop: 10}}>
          <div
            style={{
              opacity: enter(frame, sec(1.5)),
              padding: "10px 14px",
              borderRadius: 6,
              border: `1px solid ${voice ? COLORS.persona : COLORS.line}`,
              background: voice ? `${COLORS.persona}16` : "rgba(255,255,255,0.02)",
              minWidth: 210,
            }}
          >
            <div style={{fontFamily: FONT_SANS, fontSize: 17, color: voice ? COLORS.persona : COLORS.muted}}>
              GPT-SoVITS → QQ 语音条
            </div>
            <div style={{fontFamily: FONT_SANS, fontSize: 13, color: COLORS.dim, marginTop: 3}}>
              成功就不再发拆分文字
            </div>
          </div>
          <div
            style={{
              opacity: enter(frame, sec(1.7)),
              padding: "10px 14px",
              borderRadius: 6,
              border: `1px solid ${!voice ? COLORS.fact : COLORS.line}`,
              background: !voice ? `${COLORS.fact}16` : "rgba(255,255,255,0.02)",
              minWidth: 210,
            }}
          >
            <div style={{fontFamily: FONT_SANS, fontSize: 17, color: !voice ? COLORS.fact : COLORS.muted}}>
              split_reply → 多个文字气泡
            </div>
            <div style={{fontFamily: FONT_SANS, fontSize: 13, color: COLORS.dim, marginTop: 3}}>
              语音没走成时的兜底
            </div>
          </div>
        </div>

        {showWriteBack ? (
          <>
            <Connector frame={frame} delay={sec(2.1)} height={20} />
            <Step
              index={6}
              label="写回短期 / 长期记忆"
              note="这一集到这里就够了"
              frame={frame}
              delay={sec(2.2)}
              color={COLORS.scene}
              width={width}
            />
          </>
        ) : null}
      </div>

      {/* 右：真实数据（原始 vs 清洗后 vs 实际投递） */}
      <div style={{display: "flex", flexDirection: "column", gap: 14, width: 620}}>
        <div style={{opacity: enter(frame, sec(0.4))}}>
          <div style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.accent, marginBottom: 6}}>
            模型原始回复（真实）
          </div>
          <div
            style={{
              fontFamily: FONT_SANS,
              fontSize: 17,
              lineHeight: 1.5,
              color: COLORS.muted,
              padding: "11px 14px",
              background: COLORS.panel,
              border: `1px solid ${COLORS.line}`,
              borderRadius: 6,
              whiteSpace: "pre-wrap",
            }}
          >
            {rawReply}
          </div>
        </div>

        <div style={{opacity: enter(frame, sec(1.0))}}>
          <div style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.fact, marginBottom: 6}}>
            clean_reply 之后
          </div>
          <div
            style={{
              fontFamily: FONT_SANS,
              fontSize: 17,
              lineHeight: 1.5,
              color: COLORS.paper,
              padding: "11px 14px",
              background: COLORS.panel,
              border: `1px solid ${COLORS.fact}55`,
              borderRadius: 6,
            }}
          >
            {cleanedReply}
          </div>
        </div>

        {!voice ? (
          <div style={{opacity: enter(frame, sec(1.5))}}>
            <div style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.fact, marginBottom: 6}}>
              split_reply 拆出 {delivery.parts.length} 条 QQ 气泡
            </div>
            <div style={{display: "flex", flexDirection: "column", gap: 7, alignItems: "flex-end"}}>
              {delivery.parts.map((t, i) => (
                <div
                  key={t}
                  style={{
                    maxWidth: "86%",
                    fontFamily: FONT_SANS,
                    fontSize: 16,
                    color: COLORS.paper,
                    padding: "8px 13px",
                    borderRadius: "12px 12px 3px 12px",
                    background: `${COLORS.fact}1e`,
                    border: `1px solid ${COLORS.fact}66`,
                    opacity: lin(frame, [sec(1.6) + i * 8, sec(1.6) + i * 8 + 12], [0, 1]),
                    transform: `translateY(${
                      (1 - lin(frame, [sec(1.6) + i * 8, sec(1.6) + i * 8 + 12], [0, 1])) * 10
                    }px)`,
                  }}
                >
                  {t}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div
          style={{
            opacity: enter(frame, sec(2.0)),
            fontFamily: FONT_SANS,
            fontSize: 15,
            color: COLORS.muted,
            borderTop: `1px solid ${COLORS.line}`,
            paddingTop: 10,
          }}
        >
          代码工程篇继续：消息攒批、清洗规则、拆分正则、GPT-SoVITS 的实现细节。
        </div>
      </div>
    </div>
  );
};
