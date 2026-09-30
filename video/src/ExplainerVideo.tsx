import React from "react";
import {Audio} from "@remotion/media";
import {
  AbsoluteFill,
  Img,
  Sequence,
  Series,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import narrationData from "./narration.json";

type Scene = (typeof narrationData.scenes)[number];
type NarrationLine = (typeof narrationData.lines)[number];

type VisualProps = {
  scene: Scene;
  sceneFrame: number;
  sceneDuration: number;
};

const colors = {
  ink: "#171816",
  paper: "#f4efe6",
  coral: "#ef6950",
  teal: "#2aa198",
  amber: "#df9f2f",
  blue: "#4d7fd6",
  muted: "#a9a59b",
  panel: "#242622",
  panelSoft: "#30332e",
  line: "#4a4d45",
};

const msToFrames = (ms: number) => Math.round((ms / 1000) * narrationData.fps);

const sceneDurationInFrames = (scene: Scene) =>
  msToFrames(scene.endMs - scene.startMs);

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

const fadeWindow = (frame: number, duration: number, fadeFrames = 12) =>
  Math.min(
    clamp01(frame / fadeFrames),
    clamp01((duration - frame) / fadeFrames),
  );

const enter = (frame: number, fps: number, delayFrames = 0) =>
  spring({
    frame: frame - delayFrames,
    fps,
    config: {damping: 200},
    durationInFrames: Math.round(fps * 0.5),
  });

const CaptionTrack: React.FC = () => {
  const frame = useCurrentFrame();
  const line = narrationData.lines.find(
    (item) => frame >= msToFrames(item.startMs) && frame < msToFrames(item.endMs),
  );

  if (!line) {
    return null;
  }

  const startFrame = msToFrames(line.startMs);
  const endFrame = msToFrames(line.endMs);
  const duration = endFrame - startFrame;
  const local = frame - startFrame;
  const reveal = clamp01(local / 6);
  const exit = clamp01((duration - local) / 6);
  const visible = Math.min(reveal, exit);
  const progress = clamp01(local / Math.max(1, duration));

  return (
    <div
      style={{
        position: "absolute",
        left: 250,
        right: 250,
        bottom: 54,
        height: 112,
        padding: "20px 28px 18px",
        borderRadius: 8,
        background: "rgba(15, 16, 15, 0.86)",
        border: `1px solid ${colors.line}`,
        color: colors.paper,
        opacity: visible,
        transform: `translateY(${interpolate(reveal, [0, 1], [12, 0])}px)`,
        boxShadow: "0 18px 50px rgba(0,0,0,0.32)",
      }}
    >
      <div
        style={{
          fontFamily: '"Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
          fontSize: 31,
          fontWeight: 600,
          lineHeight: 1.35,
          letterSpacing: 0,
        }}
      >
        {line.text}
      </div>
      <div
        style={{
          position: "absolute",
          left: 28,
          right: 28,
          bottom: 12,
          height: 3,
          background: "rgba(244,239,230,0.16)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${progress * 100}%`,
            height: "100%",
            background: colors.coral,
          }}
        />
      </div>
    </div>
  );
};

const VoiceoverTrack: React.FC<{enabled: boolean}> = ({enabled}) => {
  if (!enabled) {
    return null;
  }

  return (
    <>
      {narrationData.lines.map((line) => (
        <Sequence
          key={line.id}
          from={msToFrames(line.startMs)}
          durationInFrames={Math.max(1, msToFrames(line.endMs - line.startMs))}
          premountFor={narrationData.fps}
        >
          <Audio src={staticFile(`audio/${line.id}.wav`)} />
        </Sequence>
      ))}
    </>
  );
};

const GridBackground: React.FC = () => {
  return (
    <AbsoluteFill
      style={{
        background:
          "repeating-linear-gradient(90deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 72px), repeating-linear-gradient(0deg, rgba(255,255,255,0.028) 0 1px, transparent 1px 72px)",
      }}
    />
  );
};

const Panel: React.FC<{
  children: React.ReactNode;
  tone?: string;
  style?: React.CSSProperties;
}> = ({children, tone = colors.panel, style}) => (
  <div
    style={{
      background: tone,
      border: `1px solid ${colors.line}`,
      borderRadius: 8,
      boxShadow: "0 24px 54px rgba(0,0,0,0.22)",
      ...style,
    }}
  >
    {children}
  </div>
);

const Label: React.FC<{children: React.ReactNode; color?: string}> = ({
  children,
  color = colors.paper,
}) => (
  <div
    style={{
      fontFamily: '"Microsoft YaHei", sans-serif',
      fontSize: 20,
      fontWeight: 600,
      color,
      letterSpacing: 0,
    }}
  >
    {children}
  </div>
);

const FileCard: React.FC<{name: string; tone: string; delay: number}> = ({
  name,
  tone,
  delay,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = enter(frame, fps, delay);
  return (
    <Panel
      tone={colors.panelSoft}
      style={{
        width: 360,
        height: 104,
        padding: "19px 20px",
        display: "flex",
        alignItems: "center",
        gap: 14,
        opacity: p,
        transform: `translateX(${(1 - p) * -24}px)`,
      }}
    >
      <div
        style={{
          width: 34,
          height: 42,
          border: `2px solid ${tone}`,
          borderRadius: 3,
          position: "relative",
          flex: "0 0 auto",
        }}
      >
        <div
          style={{
            position: "absolute",
            right: -2,
            top: -2,
            width: 11,
            height: 11,
            borderLeft: `2px solid ${tone}`,
            borderBottom: `2px solid ${tone}`,
            background: colors.panelSoft,
          }}
        />
      </div>
      <div>
        <Label color={colors.muted}>FILE</Label>
        <div
          style={{
            fontFamily: '"Cascadia Mono", "Consolas", monospace',
            fontSize: 21,
            color: colors.paper,
            marginTop: 5,
            whiteSpace: "nowrap",
          }}
        >
          {name}
        </div>
      </div>
    </Panel>
  );
};

const MessageBubble: React.FC<{
  role: string;
  text: string;
  tone: string;
  delay: number;
  align?: "left" | "right";
}> = ({role, text, tone, delay, align = "left"}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = enter(frame, fps, delay);
  return (
    <div
      style={{
        width: 392,
        padding: "16px 18px",
        borderRadius: 8,
        border: `1px solid ${tone}`,
        background: align === "right" ? "rgba(239,105,80,0.12)" : colors.panelSoft,
        opacity: p,
        transform: `translateY(${(1 - p) * 20}px)`,
        alignSelf: align === "right" ? "flex-end" : "flex-start",
      }}
    >
      <div
        style={{
          fontFamily: '"Cascadia Mono", "Consolas", monospace',
          color: tone,
          fontSize: 18,
          marginBottom: 8,
        }}
      >
        {role}
      </div>
      <div
        style={{
          fontFamily: '"Microsoft YaHei", sans-serif',
          color: colors.paper,
          fontSize: 25,
          lineHeight: 1.35,
        }}
      >
        {text}
      </div>
    </div>
  );
};

const FileToMessageVisual: React.FC<VisualProps> = ({sceneFrame}) => {
  const {fps} = useVideoConfig();
  const shift = enter(sceneFrame, fps, 18);
  return (
    <div
      style={{
        position: "absolute",
        left: 112,
        right: 112,
        top: 285,
        height: 410,
        display: "flex",
        alignItems: "center",
        gap: 62,
      }}
    >
      <div style={{display: "flex", flexDirection: "column", gap: 16}}>
        <FileCard name="system_prompt.txt" tone={colors.coral} delay={0} />
        <FileCard name="traits.json" tone={colors.amber} delay={7} />
        <FileCard name="corpus_vectors.json" tone={colors.teal} delay={14} />
      </div>
      <div
        style={{
          flex: "0 0 auto",
          width: 260,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 15,
          opacity: enter(sceneFrame, fps, 25),
        }}
      >
        <div
          style={{
            fontFamily: '"Cascadia Mono", "Consolas", monospace',
            color: colors.muted,
            fontSize: 19,
          }}
        >
          READ + TRANSFORM
        </div>
        <div
          style={{
            width: 230,
            height: 3,
            background: colors.line,
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${interpolate(shift, [0, 1], [0, 100])}%`,
              height: "100%",
              background: colors.coral,
            }}
          />
        </div>
        <div style={{fontSize: 54, color: colors.coral}}>→</div>
      </div>
      <div style={{display: "flex", flexDirection: "column", gap: 14}}>
        <MessageBubble
          role="system"
          text="【灰泽满的周表】周三 19:00"
          tone={colors.amber}
          delay={28}
        />
        <MessageBubble
          role="user"
          text="今天在干嘛呀"
          tone={colors.blue}
          delay={36}
          align="right"
        />
        <MessageBubble
          role="assistant"
          text="没干嘛，刚写了两行作业。"
          tone={colors.coral}
          delay={44}
        />
      </div>
    </div>
  );
};

const TransformVisual: React.FC<VisualProps> = (props) => (
  <FileToMessageVisual {...props} />
);

const JourneyVisual: React.FC<VisualProps> = ({sceneFrame}) => {
  const {fps} = useVideoConfig();
  const steps = [
    "收到",
    "攒批",
    "理解",
    "取素材",
    "组装",
    "生成",
    "后处理",
    "发送",
    "回写",
  ];
  return (
    <div
      style={{
        position: "absolute",
        left: 150,
        right: 150,
        top: 300,
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: 20,
      }}
    >
      {steps.map((step, index) => {
        const p = enter(sceneFrame, fps, index * 5);
        const active = index === 3 || index === 4;
        return (
          <Panel
            key={step}
            tone={active ? "rgba(239,105,80,0.18)" : colors.panelSoft}
            style={{
              height: 112,
              padding: "0 28px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              opacity: p,
              transform: `translateY(${(1 - p) * 18}px)`,
              borderColor: active ? colors.coral : colors.line,
            }}
          >
            <div
              style={{
                fontFamily: '"Cascadia Mono", "Consolas", monospace',
                color: active ? colors.coral : colors.muted,
                fontSize: 22,
              }}
            >
              {String(index + 1).padStart(2, "0")}
            </div>
            <Label color={colors.paper}>{step}</Label>
          </Panel>
        );
      })}
    </div>
  );
};

const AlwaysOnVisual: React.FC<VisualProps> = ({sceneFrame}) => {
  const {fps} = useVideoConfig();
  const cards = [
    ["system_prompt.txt", "身份框架、自称、底线", colors.coral],
    ["traits.json", "性格基底", colors.amber],
    ["styles.json", "语言风格", colors.teal],
  ];
  return (
    <div
      style={{
        position: "absolute",
        left: 150,
        right: 150,
        top: 290,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 70,
      }}
    >
      <div style={{display: "flex", flexDirection: "column", gap: 17}}>
        {cards.map(([name, description, tone], index) => {
          const p = enter(sceneFrame, fps, index * 8);
          return (
            <Panel
              key={name}
              tone={colors.panelSoft}
              style={{
                width: 760,
                height: 104,
                padding: "0 24px",
                display: "flex",
                alignItems: "center",
                gap: 20,
                opacity: p,
                transform: `translateX(${(1 - p) * -24}px)`,
              }}
            >
              <div
                style={{
                  width: 16,
                  height: 16,
                  borderRadius: 4,
                  background: tone,
                }}
              />
              <div
                style={{
                  fontFamily: '"Cascadia Mono", "Consolas", monospace',
                  color: colors.paper,
                  fontSize: 26,
                  width: 280,
                }}
              >
                {name}
              </div>
              <Label color={colors.muted}>{description}</Label>
            </Panel>
          );
        })}
      </div>
      <div
        style={{
          width: 430,
          height: 470,
          position: "relative",
          opacity: enter(sceneFrame, fps, 16),
          transform: `translateX(${(1 - enter(sceneFrame, fps, 16)) * 30}px)`,
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(42,161,152,0.12)",
            border: `1px solid ${colors.teal}`,
            borderRadius: 8,
          }}
        />
        <Img
          src={staticFile("hazel_stand.png")}
          style={{
            position: "absolute",
            width: 340,
            height: 425,
            left: 45,
            top: 28,
            objectFit: "contain",
          }}
        />
      </div>
    </div>
  );
};

const InjectionVisual: React.FC<VisualProps> = ({sceneFrame}) => {
  const {fps} = useVideoConfig();
  const rows = [
    ["周表", "always", colors.amber],
    ["偏好", "关键词命中", colors.coral],
    ["核心记忆", "低阈值向量", colors.teal],
    ["行为", "L3 分类", colors.blue],
    ["经历 / 措辞 / 样本", "检索与判定", colors.coral],
  ];
  return (
    <div
      style={{
        position: "absolute",
        left: 310,
        right: 310,
        top: 285,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      {rows.map(([name, rule, tone], index) => {
        const p = enter(sceneFrame, fps, index * 7);
        return (
          <Panel
            key={name}
            tone={colors.panelSoft}
            style={{
              height: 83,
              padding: "0 26px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              opacity: p,
              transform: `translateY(${(1 - p) * 16}px)`,
            }}
          >
            <Label color={colors.paper}>{name}</Label>
            <div
              style={{
                color: tone,
                fontFamily: '"Cascadia Mono", "Consolas", monospace',
                fontSize: 22,
              }}
            >
              {rule}
            </div>
          </Panel>
        );
      })}
    </div>
  );
};

const RrfVisual: React.FC<VisualProps> = ({sceneFrame}) => {
  const {fps} = useVideoConfig();
  const sources = [
    ["corpus", colors.coral],
    ["voice", colors.teal],
    ["behavior", colors.amber],
    ["phrase", colors.blue],
  ];
  return (
    <div
      style={{
        position: "absolute",
        left: 150,
        right: 150,
        top: 300,
        height: 470,
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 22,
        }}
      >
        {sources.map(([name, tone], index) => {
          const p = enter(sceneFrame, fps, index * 5);
          return (
            <Panel
              key={name}
              tone={colors.panelSoft}
              style={{
                height: 108,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: tone,
                fontFamily: '"Cascadia Mono", "Consolas", monospace',
                fontSize: 28,
                opacity: p,
                transform: `translateY(${(1 - p) * 18}px)`,
              }}
            >
              {name}
            </Panel>
          );
        })}
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: 110,
          marginTop: 26,
          color: colors.muted,
          fontSize: 36,
          opacity: enter(sceneFrame, fps, 20),
        }}
      >
        <span>↘</span>
        <span>↘</span>
        <span>↙</span>
        <span>↙</span>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          marginTop: 2,
          opacity: enter(sceneFrame, fps, 28),
        }}
      >
        <Panel
          tone="rgba(239,105,80,0.18)"
          style={{
            width: 620,
            height: 108,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderColor: colors.coral,
          }}
        >
          <Label color={colors.paper}>RRF 融合 → top 6 → 预算截断</Label>
        </Panel>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: 28,
          marginTop: 26,
          opacity: enter(sceneFrame, fps, 38),
        }}
      >
        <div
          style={{
            border: `1px solid ${colors.teal}`,
            color: colors.teal,
            borderRadius: 8,
            padding: "15px 28px",
            fontFamily: '"Cascadia Mono", "Consolas", monospace',
            fontSize: 23,
          }}
        >
          preference → 直接注入
        </div>
        <div
          style={{
            border: `1px solid ${colors.amber}`,
            color: colors.amber,
            borderRadius: 8,
            padding: "15px 28px",
            fontFamily: '"Cascadia Mono", "Consolas", monospace',
            fontSize: 23,
          }}
        >
          core story → 直接注入
        </div>
      </div>
    </div>
  );
};

const SegmentsVisual: React.FC<VisualProps> = ({sceneFrame}) => {
  const {fps} = useVideoConfig();
  const groups = [
    {label: "基础人格", count: 1, tone: colors.coral},
    {label: "当前事实", count: 2, tone: colors.amber},
    {label: "按需素材", count: 6, tone: colors.teal},
    {label: "对话记忆", count: 5, tone: colors.blue},
    {label: "当前输入", count: 2, tone: colors.coral},
  ];
  const blocks = groups.flatMap((group) =>
    Array.from({length: group.count}, (_, index) => ({
      ...group,
      index,
    })),
  );
  return (
    <div
      style={{
        position: "absolute",
        left: 180,
        right: 180,
        top: 340,
      }}
    >
      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "flex-end",
          justifyContent: "center",
        }}
      >
        {blocks.map((block, index) => {
          const p = enter(sceneFrame, fps, index * 3);
          return (
            <div
              key={`${block.label}-${index}`}
              style={{
                width: 74,
                height: 184,
                borderRadius: 6,
                border: `1px solid ${block.tone}`,
                background: `${block.tone}22`,
                opacity: p,
                transform: `translateY(${(1 - p) * 34}px)`,
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  right: 0,
                  top: 15,
                  textAlign: "center",
                  color: block.tone,
                  fontFamily: '"Cascadia Mono", "Consolas", monospace',
                  fontSize: 19,
                }}
              >
                {String(index + 1).padStart(2, "0")}
              </div>
            </div>
          );
        })}
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 34,
          color: colors.muted,
          fontFamily: '"Microsoft YaHei", sans-serif',
          fontSize: 21,
        }}
      >
        <span>背景 / 人格</span>
        <span>当前输入 / 用户原话</span>
      </div>
    </div>
  );
};

const SummaryVisual: React.FC<VisualProps> = ({sceneFrame}) => {
  const {fps} = useVideoConfig();
  const items = ["文件", "触发判断", "检索筛选", "messages", "生成"];
  return (
    <div
      style={{
        position: "absolute",
        left: 200,
        right: 200,
        top: 345,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 20,
      }}
    >
      {items.map((item, index) => {
        const p = enter(sceneFrame, fps, index * 9);
        return (
          <React.Fragment key={item}>
            <Panel
              tone={index === items.length - 1 ? "rgba(239,105,80,0.20)" : colors.panelSoft}
              style={{
                padding: "27px 28px",
                opacity: p,
                transform: `translateY(${(1 - p) * 20}px)`,
                borderColor: index === items.length - 1 ? colors.coral : colors.line,
              }}
            >
              <Label color={colors.paper}>{item}</Label>
            </Panel>
            {index < items.length - 1 ? (
              <div
                style={{
                  color: colors.muted,
                  fontSize: 35,
                  opacity: enter(sceneFrame, fps, index * 9 + 5),
                }}
              >
                →
              </div>
            ) : null}
          </React.Fragment>
        );
      })}
    </div>
  );
};

const renderVisual = (scene: Scene, sceneFrame: number, sceneDuration: number) => {
  const props = {scene, sceneFrame, sceneDuration};
  switch (scene.visual) {
    case "file-to-message":
      return <FileToMessageVisual {...props} />;
    case "transform":
      return <TransformVisual {...props} />;
    case "journey":
      return <JourneyVisual {...props} />;
    case "always-on":
      return <AlwaysOnVisual {...props} />;
    case "injection":
      return <InjectionVisual {...props} />;
    case "rrf":
      return <RrfVisual {...props} />;
    case "segments":
      return <SegmentsVisual {...props} />;
    case "summary":
      return <SummaryVisual {...props} />;
    default:
      return null;
  }
};

const SceneView: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const sceneDuration = sceneDurationInFrames(scene);
  const fade = fadeWindow(frame, sceneDuration);
  const titleProgress = enter(frame, fps, 8);
  const subtitleProgress = enter(frame, fps, 18);

  return (
    <AbsoluteFill
      style={{
        background: colors.ink,
        color: colors.paper,
        opacity: fade,
        overflow: "hidden",
      }}
    >
      <GridBackground />
      <div
        style={{
          position: "absolute",
          left: 84,
          right: 84,
          top: 62,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <div>
          <div
            style={{
              fontFamily: '"Cascadia Mono", "Consolas", monospace',
              color: colors.coral,
              fontSize: 22,
              marginBottom: 16,
              opacity: titleProgress,
            }}
          >
            {scene.kicker}
          </div>
          <div
            style={{
              fontFamily: '"Microsoft YaHei", sans-serif',
              fontSize: scene.id === "summary" ? 54 : 64,
              fontWeight: 700,
              lineHeight: 1.08,
              letterSpacing: 0,
              opacity: titleProgress,
              transform: `translateY(${(1 - titleProgress) * 20}px)`,
            }}
          >
            {scene.title}
          </div>
        </div>
        <div
          style={{
            width: 470,
            textAlign: "right",
            fontFamily: '"Microsoft YaHei", sans-serif',
            color: colors.muted,
            fontSize: 23,
            lineHeight: 1.45,
            paddingTop: 43,
            opacity: subtitleProgress,
          }}
        >
          {scene.subtitle}
        </div>
      </div>
      {renderVisual(scene, frame, sceneDuration)}
    </AbsoluteFill>
  );
};

export const ExplainerVideo: React.FC<{withVoiceover: boolean}> = ({
  withVoiceover,
}) => {
  return (
    <AbsoluteFill style={{background: colors.ink}}>
      <Series>
        {narrationData.scenes.map((scene) => (
          <Series.Sequence
            key={scene.id}
            durationInFrames={sceneDurationInFrames(scene)}
            premountFor={narrationData.fps}
          >
            <SceneView scene={scene} />
          </Series.Sequence>
        ))}
      </Series>
      <VoiceoverTrack enabled={withVoiceover} />
      <CaptionTrack />
    </AbsoluteFill>
  );
};
