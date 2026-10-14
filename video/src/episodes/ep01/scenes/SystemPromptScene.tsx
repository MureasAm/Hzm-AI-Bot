/**
 * Scene 02 · system-prompt —— 稳定骨架（规格 §5.1）。
 *
 * 本段必须讲清：每轮常驻 / 规定不随话题变化的东西 / 擅长保证"始终是同一个人" /
 * 不适合穷举具体人际情景 / 不是"越长越好"。
 *
 * 画面：左侧文件目录，右侧把章节压成一张人物骨架图；"身份/自称/节奏"等标签**固定在骨架上**；
 * 用户输入变化时**骨架保持不动**。
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
import {FileCard} from "../components/FileCard";
import {HazelAvatar} from "../components/HazelAvatar";

/** 骨架图上的固定标签（由真实章节名映射，不凭空写）。
 *  y 是**标签列的纵向间距**，不是绝对坐标——骨架图旁并排一列，
 *  避免标签压在骨架上互相重叠。 */
const SKELETON_TAGS = [
  {label: "身份", heading: "身份：人设 × 真实经历"},
  {label: "位置", heading: "身份与位置"},
  {label: "节奏", heading: "说话节奏"},
  {label: "自称", heading: "自我称呼"},
  {label: "括号", heading: "括号与省略号：例外不是常态"},
  {label: "底色", heading: "行为底色"},
];

/** 输入在变、骨架不动 —— 用右侧的三个"飘过的用户输入"表示。 */
const DRIFT_INPUTS = [
  "你今天唱得特别好，我真的很喜欢。",
  "你今晚几点播来着？",
  "下播后念一段《乌色月》给我听呗",
];

export const SystemPromptScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  const sp = personaStructure.systemPrompt;
  const tone = DOMAIN_COLOR.persona;

  const dirP = enter(frame, 0);
  const skeletonP = enter(frame, sec(0.9));

  // 骨架从不变化：这里只让"输入"流过，骨架的 tags 位置恒定
  const driftPhase = lin(frame, [sec(3.0), sec(9.0)], [0, 1]);

  return (
    <div style={{position: "absolute", left: 76, right: 76, top: 232, display: "flex", gap: 46}}>
      {/* 左：文件目录 + 常驻标记 */}
      <div style={{width: 430, opacity: dirP, display: "flex", flexDirection: "column", gap: 14}}>
        <div style={{fontFamily: FONT_MONO, fontSize: 15, color: COLORS.accent, letterSpacing: 1}}>
          文件目录 · persona/
        </div>
        <FileCard
          name="core/system_prompt.txt"
          note={`${sp.chars} 字 · ${sp.sections.length} 个章节`}
          domain="persona"
          highlight
          progress={dirP}
          width={430}
        />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 6,
            paddingLeft: 4,
          }}
        >
          {sp.sections.map((s, i) => {
            const p = enter(frame, sec(0.3) + i * 5);
            return (
              <div
                key={s.name}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "baseline",
                  gap: 10,
                  opacity: p * 0.92,
                  transform: `translateX(${(1 - p) * -10}px)`,
                  borderLeft: `2px solid ${tone}88`,
                  paddingLeft: 10,
                }}
              >
                <span style={{fontFamily: FONT_SANS, fontSize: 16, color: COLORS.paper}}>
                  {s.name}
                </span>
                <span style={{fontFamily: FONT_MONO, fontSize: 12, color: COLORS.dim}}>
                  {s.chars}
                </span>
              </div>
            );
          })}
        </div>
        <div
          style={{
            marginTop: 4,
            padding: "9px 12px",
            border: `1px solid ${COLORS.warn}`,
            background: `${COLORS.warn}14`,
            borderRadius: 6,
            fontFamily: FONT_SANS,
            fontSize: 16,
            color: COLORS.warn,
            opacity: enter(frame, sec(1.6)),
          }}
        >
          首条 system message · 每轮都在场 · 不参与任何检索判断
        </div>
      </div>

      {/* 右：人物骨架图（章节压上去，输入在变它不动） */}
      <div
        style={{
          flex: 1,
          position: "relative",
          height: 470,
          opacity: skeletonP,
          display: "flex",
          gap: 30,
          justifyContent: "center",
        }}
      >
        <div style={{position: "relative", width: 560, height: 460, display: "flex", gap: 20}}>
          {/* 骨架：一根脊柱 + 肩 + 头，极简线条，不做模板化 AI 视觉 */}
          <svg width="200" height="460" viewBox="0 0 200 460" style={{flex: "0 0 auto"}}>
            <circle cx="100" cy="52" r="30" fill="none" stroke={tone} strokeWidth="2" />
            <line x1="100" y1="82" x2="100" y2="330" stroke={tone} strokeWidth="2.4" />
            <line x1="30" y1="140" x2="170" y2="140" stroke={tone} strokeWidth="2" />
            <line x1="100" y1="330" x2="56" y2="440" stroke={tone} strokeWidth="2" />
            <line x1="100" y1="330" x2="144" y2="440" stroke={tone} strokeWidth="2" />
            {/* 固定落点：标签列的连线终点，位置恒定 */}
            {[40, 118, 196, 274, 330, 402].map((y) => (
              <circle key={y} cx="100" cy={y} r="4" fill={tone} />
            ))}
            {/* 骨架 → 标签的引线（等长，视觉上把标签钉在骨架上） */}
            {[40, 118, 196, 274, 330, 402].map((y) => (
              <line key={`l${y}`} x1="104" y1={y} x2="188" y2={y} stroke={`${tone}66`} strokeWidth="1.2" />
            ))}
          </svg>

          {/* 固定标签列：位置恒定，不随输入变化，互不重叠 */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              height: 400,
              paddingTop: 22,
            }}
          >
            {SKELETON_TAGS.map((t, i) => {
              const p = enter(frame, sec(1.4) + i * 4);
              return (
                <div
                  key={t.label}
                  style={{
                    opacity: p,
                    transform: `translateX(${(1 - p) * 14}px)`,
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                  }}
                >
                  <span
                    style={{
                      display: "inline-block",
                      padding: "4px 12px",
                      borderRadius: 999,
                      border: `1px solid ${tone}`,
                      background: COLORS.ink,
                      fontFamily: FONT_SANS,
                      fontSize: 16,
                      fontWeight: 600,
                      color: tone,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {t.label}
                  </span>
                  <span
                    style={{
                      fontFamily: FONT_SANS,
                      fontSize: 13,
                      color: COLORS.dim,
                      whiteSpace: "nowrap",
                      maxWidth: 200,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {t.heading}
                  </span>
                </div>
              );
            })}
          </div>

          <div
            style={{
              position: "absolute",
              left: 0,
              bottom: -8,
              width: 200,
              textAlign: "center",
              fontFamily: FONT_SANS,
              fontSize: 17,
              color: COLORS.muted,
              opacity: enter(frame, sec(2.6)),
            }}
          >
            骨架不动
          </div>
        </div>

        {/* 流动的输入：让观众看到"输入一直在换，骨架没换" */}
        <div
          style={{
            width: 420,
            display: "flex",
            flexDirection: "column",
            gap: 12,
            paddingTop: 40,
            opacity: enter(frame, sec(2.0)),
          }}
        >
          <div style={{fontFamily: FONT_MONO, fontSize: 14, color: COLORS.muted, letterSpacing: 1}}>
            用户输入一直在换
          </div>
          {DRIFT_INPUTS.map((t, i) => {
            // 依次高亮流过，但右侧骨架完全静止
            const slot = (driftPhase * DRIFT_INPUTS.length - i) % DRIFT_INPUTS.length;
            const near = Math.max(0, 1 - Math.abs(slot) / 1.4);
            return (
              <div
                key={t}
                style={{
                  fontFamily: FONT_SANS,
                  fontSize: 17,
                  lineHeight: 1.4,
                  color: near > 0.5 ? COLORS.paper : COLORS.muted,
                  padding: "9px 13px",
                  borderRadius: "10px 10px 10px 3px",
                  background: COLORS.panel,
                  border: `1px solid ${near > 0.5 ? COLORS.fact : COLORS.line}`,
                  opacity: 0.5 + 0.5 * near,
                  transform: `translateX(${(1 - near) * 12}px)`,
                }}
              >
                {t}
              </div>
            );
          })}

          <div
            style={{
              marginTop: 6,
              fontFamily: FONT_SANS,
              fontSize: 17,
              lineHeight: 1.45,
              color: COLORS.accent,
              opacity: enter(frame, at("system-prompt", "system-prompt-7", -0.5)),
              borderTop: `1px solid ${COLORS.line}`,
              paddingTop: 12,
            }}
          >
            它保证的是不变。
            <br />
            穷举每一种具体人际情景，不是它的活。
          </div>
        </div>
      </div>

      {/* 立绘：章节转场时出现，不常驻、不遮挡信息（规格 §12.4） */}
      <HazelAvatar
        frame={frame}
        delayFrames={at("system-prompt", "system-prompt-5", -0.4)}
        width={250}
        mood="neutral"
        opacity={fadeInOut(
          frame,
          at("system-prompt", "system-prompt-5", -0.4),
          at("system-prompt", "system-prompt-7", 0.6),
          at("system-prompt", "system-prompt-8", -0.2),
        )}
        style={{position: "absolute", right: -8, bottom: -232}}
      />
    </div>
  );
};
