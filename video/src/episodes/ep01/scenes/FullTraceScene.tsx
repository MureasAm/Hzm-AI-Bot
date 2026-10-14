/**
 * Scene 08 · full-trace —— 完整链路演示（规格 §10）。
 *
 * 画面全部来自真实 trace（data/ep01/full-trace-decisions.json + full-trace-result.json）：
 *   左：输入批次
 *   中：判别器与被点亮的文件
 *   右：messages 栈
 *   下：未命中灰色区
 *
 * §10.4 关键旁白对应画面：不相关的内容保持灰色、保持沉默。
 * §12.4：画面最终必须停留至少 2 秒（本场景尾部由 FullTraceScene 的 hold 段保证）。
 */
import React from "react";
import {
  COLORS,
  FONT_MONO,
  FONT_SANS,
  enter,
  fadeInOut,
  lin,
  sec,
} from "../theme";
import {fixture, traceDecisions, at} from "../ep01-fixtures";
import {
  FixtureHeader,
  GateColumn,
  InputBatch,
  MessageStack,
  MissZone,
} from "../components/InjectionTrace";
import {HazelAvatar} from "../components/HazelAvatar";

export const FullTraceScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  // 判别器逐条点亮：从"读秒窗口"那句之后开始，每条 0.55s
  // （旁白一句里会点好几条，所以步长比"每句一条"要快）
  const gateStart = at("full-trace", "full-trace-3", 3.0);
  const gateStep = sec(0.55);
  // messages 栈：从"连续索要"那段开始堆叠（此时判别已讲完）
  const stackStart = at("full-trace", "full-trace-10", -1.0);
  const stackStep = sec(0.8);
  // 未命中区：与判别流程同步出现
  const missStart = at("full-trace", "full-trace-6", 0.0);
  // 尾部停留（§10.4 / §12.4：至少 2 秒）
  const holdStart = at("full-trace", "full-trace-11", 1.2);

  const holdP = lin(frame, [holdStart, holdStart + sec(0.6)], [0, 1]);
  // 停留期间轻微压暗非重点区域，把注意力收在结果上
  const dimOthers = lin(frame, [holdStart + sec(0.4), holdStart + sec(1.4)], [1, 0.5]);

  return (
    <div
      style={{
        position: "absolute",
        left: 62,
        right: 62,
        top: 196,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <FixtureHeader
        frame={frame}
        clock={fixture.clock}
        weather={fixture.weather}
        user={fixture.userProfile.name}
        topic={fixture.session.topic}
      />

      {/* 三栏：左输入 / 中判别 / 右 messages。
          21 条判别必须压在一屏内（用 compact 行高），否则会压到底部字幕安全区。 */}
      <div style={{display: "flex", gap: 24, alignItems: "flex-start"}}>
        <div style={{opacity: dimOthers}}>
          <InputBatch
            frame={frame}
            lines={fixture.incomingBatch}
            note={fixture.incomingBatchNote}
            clock={fixture.clock}
            weather={fixture.weather}
          />
        </div>

        <div style={{opacity: dimOthers}}>
          <GateColumn
            frame={frame}
            decisions={traceDecisions.decisions}
            startAt={gateStart}
            step={gateStep}
            compact
          />
        </div>

        <div style={{opacity: 1}}>
          <MessageStack
            frame={frame}
            messages={traceDecisions.messages}
            startAt={stackStart}
            step={stackStep}
            compact
          />
        </div>
      </div>

      <MissZone
        frame={frame}
        decisions={traceDecisions.decisions}
        startAt={missStart}
        revealed={frame >= at("full-trace", "full-trace-9", 0.0)}
      />

      {/* 关键结论：只在尾部停留段出现，且**覆盖在三栏之上**，避免与字幕安全区打架 */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 232,
          opacity: holdP,
          transform: `translateY(${(1 - holdP) * 12}px)`,
          display: "flex",
          justifyContent: "center",
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            maxWidth: 1420,
            padding: "22px 30px",
            borderRadius: 10,
            background: "rgba(12,13,16,0.94)",
            border: `1px solid ${COLORS.accent}`,
            boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            gap: 30,
            boxSizing: "border-box",
          }}
        >
          <div
            style={{
              flex: "0 1 auto",
              fontFamily: FONT_SANS,
              fontSize: 25,
              lineHeight: 1.5,
              color: COLORS.accent,
            }}
          >
            完整链路不是把所有文件都塞进去。
            <span style={{color: COLORS.paper}}>
              {" "}
              真正完整的是：每一层都有机会判断，
              <br />
              而不相关的内容能够保持沉默。
            </span>
          </div>
          <div
            style={{
              marginLeft: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 5,
              fontFamily: FONT_MONO,
              fontSize: 13,
              color: COLORS.muted,
              textAlign: "right",
              whiteSpace: "nowrap",
              flex: "0 0 auto",
            }}
          >
            <span>decisions {traceDecisions.decisions.length} 条</span>
            <span>
              命中 {traceDecisions.decisions.filter((d) => d.hit).length} · 沉默{" "}
              {traceDecisions.decisions.filter((d) => !d.hit).length}
            </span>
            <span>messages {traceDecisions.messageCount} 段</span>
          </div>
        </div>
      </div>

      {/* 立绘：开场转场，放在左下角的空白处，不遮挡判别器与 messages */}
      <HazelAvatar
        frame={frame}
        delayFrames={at("full-trace", "full-trace-2", -0.6)}
        width={168}
        mood="flat"
        opacity={fadeInOut(
          frame,
          at("full-trace", "full-trace-2", -0.6),
          at("full-trace", "full-trace-3", 1.0),
          at("full-trace", "full-trace-4", 0.4),
        )}
        style={{position: "absolute", left: -6, top: 470}}
      />
    </div>
  );
};
