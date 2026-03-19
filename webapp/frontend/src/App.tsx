import { useCallback, useEffect, useMemo, useState } from "react";
import { useFireList } from "./hooks/useFireList";
import { useFireData } from "./hooks/useFireData";
import { usePlayback } from "./hooks/usePlayback";
import { FireSelector } from "./components/FireSelector";
import { ModelSelector } from "./components/ModelSelector";
import { TimeControls } from "./components/TimeControls";
import { HeatmapCanvas } from "./components/HeatmapCanvas";
import { OverlayCanvas } from "./components/OverlayCanvas";
import { MetricsPanel } from "./components/MetricsPanel";
import { ColorLegend } from "./components/ColorLegend";
import { HoverTooltip } from "./components/HoverTooltip";
import { computeMetrics, type FrameMetrics } from "./lib/renderFrame";

type ViewMode = "side-by-side" | "overlay";

export default function App() {
  const { fires, loading: firesLoading } = useFireList();
  const [selectedFire, setSelectedFire] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("side-by-side");
  const [hoverInfo, setHoverInfo] = useState<{
    row: number;
    col: number;
    actual: number;
    pred: number | null;
  } | null>(null);

  const { meta, actualFrame, predFrame, loadFrame, loading: metaLoading } =
    useFireData(selectedFire, selectedModel);

  const maxTime = meta?.time_steps.length ?? 0;

  const handleTimeChange = useCallback(
    (t: number) => {
      loadFrame(t);
    },
    [loadFrame]
  );

  const playback = usePlayback(maxTime, handleTimeChange);

  // Reset model selection when fire changes
  useEffect(() => {
    setSelectedModel(null);
    playback.setTime(0);
  }, [selectedFire]);

  // Load first frame when meta loads
  useEffect(() => {
    if (meta && maxTime > 0) {
      loadFrame(0);
    }
  }, [meta]);

  // Reload frame when model changes
  useEffect(() => {
    if (meta && selectedModel) {
      loadFrame(playback.currentTime);
    }
  }, [selectedModel]);

  const availableModels = useMemo(() => {
    if (!meta) return [];
    return meta.available_models.map((m) => m.name);
  }, [meta]);

  // Get available time indices for selected model
  const modelTimeIndices = useMemo(() => {
    if (!meta || !selectedModel) return null;
    const modelInfo = meta.available_models.find((m) => m.name === selectedModel);
    return modelInfo ? new Set(modelInfo.time_indices) : null;
  }, [meta, selectedModel]);

  const metrics: FrameMetrics | null = useMemo(() => {
    if (!actualFrame?.data || !predFrame?.data) return null;
    return computeMetrics(actualFrame.data, predFrame.data);
  }, [actualFrame, predFrame]);

  const timeLabel = meta?.time_steps[playback.currentTime] ?? "";
  const rows = actualFrame?.rows ?? 0;
  const cols = actualFrame?.cols ?? 0;

  const handleHoverSideBySide = useCallback(
    (row: number, col: number, value: number) => {
      setHoverInfo({ row, col, actual: value, pred: null });
    },
    []
  );

  const handleHoverPred = useCallback(
    (row: number, col: number, value: number) => {
      const actualVal =
        actualFrame?.data && rows > 0 && cols > 0
          ? actualFrame.data[row * cols + col]
          : NaN;
      setHoverInfo({ row, col, actual: actualVal, pred: value });
    },
    [actualFrame, rows, cols]
  );

  const handleOverlayHover = useCallback(
    (row: number, col: number, actualVal: number, predVal: number) => {
      setHoverInfo({ row, col, actual: actualVal, pred: predVal });
    },
    []
  );

  if (firesLoading) {
    return (
      <div style={appStyle}>
        <div style={{ color: "var(--muted)", padding: 40 }}>Loading fires...</div>
      </div>
    );
  }

  return (
    <div style={appStyle}>
      {/* Header */}
      <header style={headerStyle}>
        <div>
          <p style={kickerStyle}>Wildfire Prediction</p>
          <h1 style={{ fontSize: "clamp(1.6rem, 3vw, 2.4rem)", margin: 0 }}>
            Model Comparison Viewer
          </h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <FireSelector fires={fires} selected={selectedFire} onSelect={setSelectedFire} />
        </div>
      </header>

      {!selectedFire && (
        <div
          style={{
            padding: "60px 20px",
            textAlign: "center",
            color: "var(--muted)",
          }}
        >
          <h2 style={{ color: "var(--text)", marginBottom: 8 }}>Select a fire to begin</h2>
          <p>Choose one of the 8 held-out test fires to visualize model predictions</p>
        </div>
      )}

      {selectedFire && meta && (
        <>
          {/* Model selection + view toggle */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 12,
            }}
          >
            <ModelSelector
              available={availableModels}
              selected={selectedModel}
              onSelect={setSelectedModel}
            />
            {selectedModel && (
              <div style={{ display: "flex", gap: 6 }}>
                {(["side-by-side", "overlay"] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    style={{
                      padding: "6px 12px",
                      borderRadius: 8,
                      border: viewMode === mode
                        ? "1px solid var(--accent-soft)"
                        : "1px solid rgba(246,244,239,0.15)",
                      background: viewMode === mode
                        ? "rgba(242,193,78,0.1)"
                        : "transparent",
                      color: viewMode === mode ? "var(--accent-soft)" : "var(--muted)",
                      fontSize: "0.75rem",
                      textTransform: "capitalize",
                    }}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Main content */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 240px", gap: 16, alignItems: "start" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {/* Canvas area */}
              {viewMode === "side-by-side" || !selectedModel ? (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: selectedModel ? "1fr 1fr" : "1fr",
                    gap: 12,
                  }}
                >
                  <div style={canvasWrapStyle}>
                    <HeatmapCanvas
                      data={actualFrame?.data ?? null}
                      rows={rows}
                      cols={cols}
                      colorMode="fire"
                      label="Ground Truth"
                      onHover={handleHoverSideBySide}
                    />
                  </div>
                  {selectedModel && (
                    <div style={canvasWrapStyle}>
                      <HeatmapCanvas
                        data={predFrame?.data ?? null}
                        rows={predFrame?.rows ?? 0}
                        cols={predFrame?.cols ?? 0}
                        colorMode="prob"
                        label={selectedModel}
                        onHover={handleHoverPred}
                      />
                    </div>
                  )}
                </div>
              ) : (
                <div style={canvasWrapStyle}>
                  <OverlayCanvas
                    actual={actualFrame?.data ?? null}
                    predicted={predFrame?.data ?? null}
                    rows={rows}
                    cols={cols}
                    onHover={handleOverlayHover}
                  />
                </div>
              )}

              {/* Color legends */}
              <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                {viewMode === "side-by-side" || !selectedModel ? (
                  <>
                    <ColorLegend mode="fire" />
                    {selectedModel && <ColorLegend mode="prob" />}
                  </>
                ) : (
                  <ColorLegend mode="overlay" />
                )}
              </div>

              {/* Timeline */}
              <TimeControls
                currentTime={playback.currentTime}
                maxTime={maxTime}
                playing={playback.playing}
                speed={playback.speed}
                timeLabel={timeLabel}
                onTimeChange={playback.setTime}
                onTogglePlay={playback.togglePlay}
                onStepForward={playback.stepForward}
                onStepBackward={playback.stepBackward}
                onSpeedChange={playback.setSpeed}
              />
            </div>

            {/* Sidebar */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {/* Fire metadata */}
              <div style={metaPanelStyle}>
                <div style={metaHeaderStyle}>Fire Info</div>
                <MetaRow label="Name" value={meta.name.replace(/_/g, " ")} />
                <MetaRow label="Grid" value={`${meta.grid_shape[0]} x ${meta.grid_shape[1]}`} />
                <MetaRow label="Timesteps" value={`${meta.time_steps.length}`} />
                <MetaRow label="CRS" value={meta.crs ?? "—"} />
                <MetaRow label="Models" value={`${meta.available_models.length}`} />
              </div>

              {/* Metrics */}
              {selectedModel && <MetricsPanel metrics={metrics} />}

              {/* Hover tooltip */}
              {hoverInfo && (
                <HoverTooltip
                  row={hoverInfo.row}
                  col={hoverInfo.col}
                  actualValue={hoverInfo.actual}
                  predValue={hoverInfo.pred ?? NaN}
                />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span
        style={{
          display: "block",
          color: "var(--muted)",
          fontSize: "0.7rem",
          marginBottom: 2,
          textTransform: "uppercase",
          letterSpacing: "0.1em",
        }}
      >
        {label}
      </span>
      <strong className="mono" style={{ fontSize: "0.85rem" }}>
        {value}
      </strong>
    </div>
  );
}

const appStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 20,
  padding: "28px clamp(18px, 4vw, 48px) 40px",
  maxWidth: 1400,
  margin: "0 auto",
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  alignItems: "flex-start",
  justifyContent: "space-between",
  gap: 16,
};

const kickerStyle: React.CSSProperties = {
  letterSpacing: "0.16em",
  textTransform: "uppercase",
  color: "var(--accent-soft)",
  fontSize: "0.75rem",
  margin: "0 0 6px",
};

const canvasWrapStyle: React.CSSProperties = {
  borderRadius: 18,
  background: "linear-gradient(135deg, #15292b, #1b3a3b)",
  padding: 12,
  boxShadow: "var(--shadow)",
  minHeight: 300,
};

const metaPanelStyle: React.CSSProperties = {
  display: "grid",
  gap: 10,
  padding: 16,
  borderRadius: 16,
  background: "var(--panel)",
  border: "1px solid rgba(246,244,239,0.12)",
  boxShadow: "var(--shadow)",
  fontSize: "0.9rem",
};

const metaHeaderStyle: React.CSSProperties = {
  fontSize: "0.75rem",
  color: "var(--muted)",
  textTransform: "uppercase",
  letterSpacing: "0.12em",
};
