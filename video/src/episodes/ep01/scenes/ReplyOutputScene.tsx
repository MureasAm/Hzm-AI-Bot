/**
 * Scene 09 · reply-output —— 输出侧（规格 §11）。
 *
 * 本集只**预告**，不抢第三集内容：
 *   只展示 ① 一段模型原始回复 ② 清洗后的文本 ③ 文字拆分或语音二选一
 *   ④ 一个"代码工程篇继续"的提示。
 *
 * 不展开 split_reply 正则、TTS HTTP 请求、缓存或重试逻辑。
 * 数据（原始回复 / 清洗结果 / 实际投递）来自真实 trace。
 */
import React from "react";
import {traceResult} from "../ep01-fixtures";
import {ReplyPipeline} from "../components/ReplyPipeline";

export const ReplyOutputScene: React.FC<{frame: number; durationInFrames: number}> = ({frame}) => {
  return (
    <div style={{position: "absolute", left: 90, right: 90, top: 226}}>
      <ReplyPipeline
        frame={frame}
        rawReply={traceResult.rawReply}
        cleanedReply={traceResult.cleanedReply}
        delivery={traceResult.delivery}
        showWriteBack
      />
    </div>
  );
};
