import {Composition} from "remotion";
import narrationData from "./narration.json";
import {ExplainerVideo} from "./ExplainerVideo";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="HazelInjectionExplainer"
      component={ExplainerVideo}
      durationInFrames={Math.ceil((narrationData.durationMs / 1000) * narrationData.fps)}
      fps={narrationData.fps}
      width={narrationData.width}
      height={narrationData.height}
      defaultProps={{withVoiceover: false}}
    />
  );
};
