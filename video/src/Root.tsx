import {Composition} from "remotion";
import narrationData from "./narration.json";
import ep01Narration from "./episodes/ep01/ep01-narration.json";
import {ExplainerVideo} from "./ExplainerVideo";
import {Ep01PersonaInjection} from "./episodes/ep01/Ep01PersonaInjection";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/*
        旧样片：含已删除架构（traits/styles/voice_samples/phrases），
        只作历史样片与组件参考。本集不覆盖它，也不复用其内容（规格 §12.1）。
      */}
      <Composition
        id="HazelInjectionExplainer"
        component={ExplainerVideo}
        durationInFrames={Math.ceil((narrationData.durationMs / 1000) * narrationData.fps)}
        fps={narrationData.fps}
        width={narrationData.width}
        height={narrationData.height}
        defaultProps={{withVoiceover: false}}
      />

      {/*
        第一集：人格核心与完整注入链路。
        duration 由 ep01-narration.json 计算（规格 §12.1）。
        画面只读 video/data/ep01/ 与 ep01-narration.json；渲染时不联网、不调模型。
      */}
      <Composition
        id="HazelBotEp01PersonaInjection"
        component={Ep01PersonaInjection}
        durationInFrames={Math.ceil((ep01Narration.durationMs / 1000) * ep01Narration.fps)}
        fps={ep01Narration.fps}
        width={ep01Narration.width}
        height={ep01Narration.height}
        defaultProps={{withVoiceover: false, withSubtitles: true}}
      />
    </>
  );
};
