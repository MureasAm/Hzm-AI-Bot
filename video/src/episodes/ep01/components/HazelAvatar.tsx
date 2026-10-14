/**
 * HazelAvatar —— 立绘（规格 §12.4）。
 *
 * 约束：
 *   - 只控制位置、裁切、轻微呼吸和表情贴片；
 *   - **不常驻**（只在开场、吐槽、例句发声、章节转场和结尾出现）；
 *   - 不做持续上下漂浮；
 *   - 不遮挡字幕、代码和 messages。
 *
 * 呼吸是帧驱动的轻微缩放（不是 CSS animation、不是漂浮）。
 */
import React from "react";
import {Img, staticFile} from "remotion";
import {COLORS, DOMAIN_COLOR, FONT_SANS, enter, lin} from "../theme";

export type HazelMood = "neutral" | "smug" | "soft" | "flat";

const MOOD_LABEL: Record<HazelMood, string> = {
  neutral: "",
  smug: "嘴硬",
  soft: "松口",
  flat: "无语",
};

export const HazelAvatar: React.FC<{
  /** 场景局部帧 */
  frame: number;
  delayFrames?: number;
  width?: number;
  side?: "left" | "right";
  mood?: HazelMood;
  /** 呼吸幅度（0 = 完全静止，适合严肃结论） */
  breath?: number;
  opacity?: number;
  /** 立绘边框色（跟随当前内容域） */
  tone?: string;
  style?: React.CSSProperties;
}> = ({
  frame,
  delayFrames = 0,
  width = 300,
  side = "right",
  mood = "neutral",
  breath = 0.006,
  opacity = 1,
  tone = DOMAIN_COLOR.persona,
  style,
}) => {
  const p = enter(frame, delayFrames);
  // 呼吸：一个很慢的正弦，仅缩放 0.6%，不位移（避免"持续漂浮"）
  const s = 1 + breath * Math.sin((frame / 30) * Math.PI * 0.5);
  const height = width * 1.25;

  return (
    <div
      style={{
        width,
        height,
        position: "relative",
        opacity: p * opacity,
        transform: `translateX(${(1 - p) * (side === "right" ? 26 : -26)}px)`,
        flex: "0 0 auto",
        ...style,
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          border: `1px solid ${tone}55`,
          background: `${tone}10`,
          borderRadius: 8,
        }}
      />
      <Img
        src={staticFile("hazel_stand.png")}
        style={{
          position: "absolute",
          left: "6%",
          top: "5%",
          width: "88%",
          height: "90%",
          objectFit: "contain",
          transform: `scale(${s})`,
          transformOrigin: "50% 80%",
        }}
      />
      {mood !== "neutral" && MOOD_LABEL[mood] ? (
        <div
          style={{
            position: "absolute",
            left: 12,
            bottom: 12,
            padding: "3px 11px",
            borderRadius: 999,
            background: COLORS.ink,
            border: `1px solid ${tone}`,
            fontFamily: FONT_SANS,
            fontSize: 15,
            color: tone,
            opacity: lin(p, [0.4, 1], [0, 1]),
          }}
        >
          {MOOD_LABEL[mood]}
        </div>
      ) : null}
    </div>
  );
};
