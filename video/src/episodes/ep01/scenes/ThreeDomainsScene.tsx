/**
 * Scene 05 · three-domains —— 三类内容：人格 / 事实 / 现场（规格 §7）。
 *
 * 任务：让观众能复述这三类，并给全片后续颜色一个出处。
 * 注意 §7.3 的边界说明：用户长期记忆虽然跨会话保存，仍属于"面向当前用户的上下文"，
 * 不是灰泽满的人格文件。
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
import {TAXONOMY, TaxonomyMap} from "../components/TaxonomyMap";
import {DomainTag} from "../components/KeywordTag";

/** 三类各自的职责边界一句话（§7 固定说法）。 */
const DUTY_LINE: Record<string, string> = {
  persona: "负责她会怎样反应 —— 只有这两份",
  fact: "负责她知道什么 —— 词、偏好、经历、排班、梗",
  scene: "负责此刻发生了什么 —— 补事实和现场，不另造人格",
};

export const ThreeDomainsScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  // 形态：0~1，明细展开进度（标题先出，然后逐层铺明细）
  const reveal = lin(frame, [sec(3.2), sec(12.0)], [0, 1]);

  return (
    <div style={{position: "absolute", left: 110, right: 110, top: 236, display: "flex", flexDirection: "column", gap: 26}}>
      <div style={{display: "flex", alignItems: "center", gap: 18, opacity: enter(frame, 0)}}>
        <span style={{fontFamily: FONT_MONO, fontSize: 15, color: COLORS.accent, letterSpacing: 1}}>
          全片固定色彩语义
        </span>
        <span style={{fontFamily: FONT_SANS, fontSize: 16, color: COLORS.muted}}>
          后面看到这三种颜色，就对应这三类职责
        </span>
      </div>

      <TaxonomyMap frame={frame} reveal={reveal} columnWidth={520} gap={40} />

      {/* 三条职责边界 */}
      <div style={{display: "flex", gap: 40, marginTop: 6}}>
        {TAXONOMY.map((e, i) => {
          const p = enter(frame, sec(12.4) + i * 6);
          return (
            <div
              key={e.domain}
              style={{
                width: 520,
                opacity: p,
                transform: `translateY(${(1 - p) * 12}px)`,
                display: "flex",
                gap: 10,
                alignItems: "flex-start",
              }}
            >
              <DomainTag domain={e.domain} />
              <span style={{fontFamily: FONT_SANS, fontSize: 16, color: COLORS.paper, lineHeight: 1.4}}>
                {DUTY_LINE[e.domain]}
              </span>
            </div>
          );
        })}
      </div>

      {/* 边界提醒（§7.3 末段） */}
      <div
        style={{
          marginTop: 8,
          padding: "11px 16px",
          borderRadius: 6,
          border: `1px solid ${DOMAIN_COLOR.scene}55`,
          background: `${DOMAIN_COLOR.scene}12`,
          fontFamily: FONT_SANS,
          fontSize: 17,
          color: COLORS.paper,
          opacity: enter(frame, sec(14.6)),
          alignSelf: "flex-start",
        }}
      >
        用户长期记忆虽然跨会话保存，但它属于
        <span style={{color: DOMAIN_COLOR.scene}}>现场</span>
        —— 面向当前用户的上下文，不是灰泽满的人格文件。
      </div>
    </div>
  );
};
