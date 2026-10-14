/**
 * 行内标记渲染：把 persona 数据里的 `**加粗**` 变成真正的粗体。
 *
 * 为什么需要：`behaviors.json` / `core_stories.json` 等数据里的 response 字段
 * 用 `**…**` 标重点（例如"**不郑重道谢、不顺着夸、不表功。**"）。
 * 那是 markdown 约定，直接当纯文本渲染就会在画面上出现裸露的 `**`
 * （本集 03 段实测如此）。这里统一解析，全片数据文本都走它。
 */
import React from "react";
import {COLORS, FONT_SANS} from "../theme";

/** 把 `**…**` 切成普通段与粗体段。 */
export const splitBold = (text: string): {t: string; bold: boolean}[] => {
  const parts: {t: string; bold: boolean}[] = [];
  const re = /\*\*(.+?)\*\*/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) {
      parts.push({t: text.slice(last, m.index), bold: false});
    }
    parts.push({t: m[1], bold: true});
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    parts.push({t: text.slice(last), bold: false});
  }
  return parts.length ? parts : [{t: text, bold: false}];
};

/** 渲染一段带 `**加粗**` 的文本。 */
export const RichText: React.FC<{text: string; boldColor?: string}> = ({
  text,
  boldColor = COLORS.paper,
}) => (
  <>
    {splitBold(text).map((p, i) =>
      p.bold ? (
        <span key={i} style={{fontWeight: 700, color: boldColor}}>
          {p.t}
        </span>
      ) : (
        <span key={i}>{p.t}</span>
      ),
    )}
  </>
);

/** 便捷：把带标记的文本转成纯文本（用于等宽卡片里不适合再加粗的场景）。 */
export const stripBold = (text: string) => text.replace(/\*\*(.+?)\*\*/g, "$1");
