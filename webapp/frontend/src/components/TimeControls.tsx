interface Props {
  currentTime: number;
  maxTime: number;
  playing: boolean;
  speed: number;
  timeLabel: string;
  onTimeChange: (t: number) => void;
  onTogglePlay: () => void;
  onStepForward: () => void;
  onStepBackward: () => void;
  onSpeedChange: (speed: number) => void;
}

const SPEEDS = [1, 2, 5, 10];

export function TimeControls({
  currentTime,
  maxTime,
  playing,
  speed,
  timeLabel,
  onTimeChange,
  onTogglePlay,
  onStepForward,
  onStepBackward,
  onSpeedChange,
}: Props) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        padding: "14px 18px",
        borderRadius: 16,
        background: "rgba(246,244,239,0.06)",
        border: "1px solid rgba(246,244,239,0.15)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <button onClick={onStepBackward} style={btnStyle} title="Step back">
          {"⏮"}
        </button>
        <button onClick={onTogglePlay} style={btnStyle} title={playing ? "Pause" : "Play"}>
          {playing ? "⏸" : "▶"}
        </button>
        <button onClick={onStepForward} style={btnStyle} title="Step forward">
          {"⏭"}
        </button>

        <input
          type="range"
          min={0}
          max={Math.max(maxTime - 1, 0)}
          value={currentTime}
          onChange={(e) => onTimeChange(parseInt(e.target.value))}
          style={{ flex: 1 }}
        />

        <div style={{ display: "flex", gap: 4 }}>
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              style={{
                ...speedBtnStyle,
                background: speed === s ? "rgba(242,107,58,0.2)" : "transparent",
                color: speed === s ? "var(--accent)" : "var(--muted)",
                borderColor: speed === s ? "var(--accent)" : "rgba(246,244,239,0.15)",
              }}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      <div
        className="mono"
        style={{
          fontSize: "0.8rem",
          color: "var(--muted)",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>
          Frame {currentTime + 1}/{maxTime}
        </span>
        <span>{timeLabel}</span>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  width: 36,
  height: 36,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  borderRadius: "50%",
  border: "1px solid rgba(246,244,239,0.2)",
  background: "rgba(246,244,239,0.06)",
  color: "var(--text)",
  fontSize: "0.9rem",
};

const speedBtnStyle: React.CSSProperties = {
  padding: "4px 8px",
  borderRadius: 8,
  border: "1px solid rgba(246,244,239,0.15)",
  fontSize: "0.7rem",
  fontFamily: "IBM Plex Mono, monospace",
};
