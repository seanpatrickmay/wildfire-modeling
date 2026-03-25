import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchFires,
  fetchProgression,
  fetchActualFrame,
  fetchPredictedFrame,
  type FireSummary,
  type ProgressionMeta,
  type FrameData,
} from "./api/client";
import { FireSelector } from "./components/FireSelector";
import { HeatmapCanvas } from "./components/HeatmapCanvas";
import { ColorLegend } from "./components/ColorLegend";
import { HoverTooltip } from "./components/HoverTooltip";

interface StepMetrics {
  actualFirePx: number;
  predFirePx: number;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
}

function computeStepMetrics(
  actual: Float32Array | null,
  predicted: Float32Array | null,
): StepMetrics | null {
  if (!actual || !predicted || actual.length !== predicted.length) return null;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let actualFirePx = 0;
  let predFirePx = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i] > 0.5 ? 1 : 0;
    const p = predicted[i] > 0.5 ? 1 : 0;
    if (a) actualFirePx++;
    if (p) predFirePx++;
    if (a && p) tp++;
    else if (!a && p) fp++;
    else if (a && !p) fn++;
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { actualFirePx, predFirePx, tp, fp, fn, precision, recall, f1 };
}

export default function App() {
  const [fires, setFires] = useState<FireSummary[]>([]);
  const [firesLoading, setFiresLoading] = useState(true);
  const [selectedFire, setSelectedFire] = useState<string | null>(null);
  const [fireMeta, setFireMeta] = useState<{
    n_hours: number;
    grid_shape: [number, number];
  } | null>(null);
  const [startHour, setStartHour] = useState(6);
  const [numSteps, setNumSteps] = useState(6);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [actualFrames, setActualFrames] = useState<Map<number, FrameData>>(
    new Map(),
  );
  const [predictedFrames, setPredictedFrames] = useState<
    Map<number, FrameData>
  >(new Map());
  const [progressionMeta, setProgressionMeta] =
    useState<ProgressionMeta | null>(null);
  const [framesLoading, setFramesLoading] = useState(false);
  const [hoverInfo, setHoverInfo] = useState<{
    row: number;
    col: number;
    actual: number;
    pred: number | null;
  } | null>(null);

  const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load fires list
  useEffect(() => {
    let cancelled = false;
    setFiresLoading(true);
    fetchFires()
      .then((data) => {
        if (!cancelled) setFires(data);
      })
      .catch((err) => console.error("Failed to load fires:", err))
      .finally(() => {
        if (!cancelled) setFiresLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // When fire selection changes, derive metadata and reset state
  useEffect(() => {
    if (!selectedFire) {
      setFireMeta(null);
      setProgressionMeta(null);
      setActualFrames(new Map());
      setPredictedFrames(new Map());
      setCurrentStep(0);
      setPlaying(false);
      return;
    }
    const fire = fires.find((f) => f.name === selectedFire);
    if (!fire) return;
    setFireMeta({
      n_hours: fire.n_hours,
      grid_shape: fire.grid_shape,
    });
    setStartHour(6);
    setNumSteps(6);
    setCurrentStep(0);
    setPlaying(false);
    setProgressionMeta(null);
    setActualFrames(new Map());
    setPredictedFrames(new Map());
  }, [selectedFire, fires]);

  // Clamp startHour when fireMeta or numSteps change
  const maxStartHour = fireMeta ? fireMeta.n_hours - 2 : 6;
  const minStartHour = 6;
  const clampedStartHour = Math.max(
    minStartHour,
    Math.min(startHour, maxStartHour),
  );

  // Clamp numSteps so start + numSteps doesn't exceed n_hours
  const effectiveMaxSteps = fireMeta
    ? Math.min(12, fireMeta.n_hours - clampedStartHour)
    : 12;
  const clampedNumSteps = Math.min(numSteps, effectiveMaxSteps);

  // Load progression data when fire/start/steps change
  useEffect(() => {
    if (!selectedFire || !fireMeta) return;

    let cancelled = false;
    setFramesLoading(true);
    setCurrentStep(0);
    setPlaying(false);

    const loadProgression = async () => {
      try {
        const meta = await fetchProgression(
          selectedFire,
          clampedStartHour,
          clampedNumSteps,
        );
        if (cancelled) return;
        setProgressionMeta(meta);

        const actualPromises: Array<Promise<[number, FrameData]>> = [];
        const predPromises: Array<Promise<[number, FrameData]>> = [];

        for (let step = 0; step < clampedNumSteps; step++) {
          actualPromises.push(
            fetchActualFrame(selectedFire, clampedStartHour + step + 1).then(
              (frame) => [step, frame] as [number, FrameData],
            ),
          );
          predPromises.push(
            fetchPredictedFrame(selectedFire, clampedStartHour, step).then(
              (frame) => [step, frame] as [number, FrameData],
            ),
          );
        }

        const [actualResults, predResults] = await Promise.all([
          Promise.all(actualPromises),
          Promise.all(predPromises),
        ]);

        if (cancelled) return;

        const newActual = new Map<number, FrameData>();
        for (const [step, frame] of actualResults) {
          newActual.set(step, frame);
        }
        setActualFrames(newActual);

        const newPred = new Map<number, FrameData>();
        for (const [step, frame] of predResults) {
          newPred.set(step, frame);
        }
        setPredictedFrames(newPred);
      } catch (err) {
        console.error("Failed to load progression:", err);
      } finally {
        if (!cancelled) setFramesLoading(false);
      }
    };

    loadProgression();
    return () => {
      cancelled = true;
    };
  }, [selectedFire, fireMeta, clampedStartHour, clampedNumSteps]);

  // Auto-play
  useEffect(() => {
    if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
    }
    if (playing && clampedNumSteps > 0) {
      playIntervalRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          const next = prev + 1;
          if (next >= clampedNumSteps) {
            setPlaying(false);
            return prev;
          }
          return next;
        });
      }, 800);
    }
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    };
  }, [playing, clampedNumSteps]);

  const currentActual = actualFrames.get(currentStep) ?? null;
  const currentPred = predictedFrames.get(currentStep) ?? null;

  const rows = currentActual?.rows ?? fireMeta?.grid_shape[0] ?? 0;
  const cols = currentActual?.cols ?? fireMeta?.grid_shape[1] ?? 0;

  const currentTimeActual = clampedStartHour + currentStep + 1;
  const stepLabel = `Step ${currentStep + 1}/${clampedNumSteps}: t=${clampedStartHour} -> t=${currentTimeActual}`;

  const metrics = useMemo(
    () =>
      computeStepMetrics(
        currentActual?.data ?? null,
        currentPred?.data ?? null,
      ),
    [currentActual, currentPred],
  );

  const handleHoverActual = useCallback(
    (row: number, col: number, value: number) => {
      const predVal =
        currentPred?.data && cols > 0
          ? currentPred.data[row * cols + col]
          : null;
      setHoverInfo({ row, col, actual: value, pred: predVal ?? null });
    },
    [currentPred, cols],
  );

  const handleHoverPred = useCallback(
    (row: number, col: number, value: number) => {
      const actualVal =
        currentActual?.data && cols > 0
          ? currentActual.data[row * cols + col]
          : NaN;
      setHoverInfo({ row, col, actual: actualVal, pred: value });
    },
    [currentActual, cols],
  );

  const stepBackward = useCallback(() => {
    setPlaying(false);
    setCurrentStep((prev) => Math.max(0, prev - 1));
  }, []);

  const stepForward = useCallback(() => {
    setPlaying(false);
    setCurrentStep((prev) => Math.min(clampedNumSteps - 1, prev + 1));
  }, [clampedNumSteps]);

  const togglePlay = useCallback(() => {
    setPlaying((prev) => {
      if (!prev && currentStep >= clampedNumSteps - 1) {
        setCurrentStep(0);
      }
      return !prev;
    });
  }, [currentStep, clampedNumSteps]);

  if (firesLoading) {
    return (
      <div style={appStyle}>
        <div style={{ color: "var(--muted)", padding: 40 }}>
          Loading fires...
        </div>
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
            Fire Progression Viewer
          </h1>
        </div>
        <FireSelector
          fires={fires}
          selected={selectedFire}
          onSelect={setSelectedFire}
        />
      </header>

      {!selectedFire && (
        <div style={emptyStateStyle}>
          <h2 style={{ color: "var(--text)", marginBottom: 8 }}>
            Select a fire to begin
          </h2>
          <p>
            Choose a fire to compare actual vs predicted fire progression over
            time
          </p>
        </div>
      )}

      {selectedFire && fireMeta && (
        <>
          {/* Controls row: start hour slider + steps dropdown */}
          <div style={controlsRowStyle}>
            <div style={controlGroupStyle}>
              <label style={controlLabelStyle}>
                Start Hour:{" "}
                <span className="mono" style={{ color: "var(--accent-soft)" }}>
                  {clampedStartHour}
                </span>
              </label>
              <input
                type="range"
                min={minStartHour}
                max={maxStartHour}
                value={clampedStartHour}
                onChange={(e) => setStartHour(Number(e.target.value))}
                style={{ width: 220 }}
              />
              <span
                className="mono"
                style={{ fontSize: "0.7rem", color: "var(--muted)" }}
              >
                {minStartHour}..{maxStartHour}
              </span>
            </div>

            <div style={controlGroupStyle}>
              <label style={controlLabelStyle}>Steps:</label>
              <select
                value={clampedNumSteps}
                onChange={(e) => setNumSteps(Number(e.target.value))}
                style={selectStyle}
              >
                {Array.from({ length: effectiveMaxSteps }, (_, i) => i + 1).map(
                  (n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ),
                )}
              </select>
            </div>

            {/* Fire metadata inline */}
            <div
              style={{
                display: "flex",
                gap: 16,
                marginLeft: "auto",
                alignItems: "center",
              }}
            >
              <MetaRow
                label="Grid"
                value={`${fireMeta.grid_shape[0]}x${fireMeta.grid_shape[1]}`}
              />
              <MetaRow label="Hours" value={`${fireMeta.n_hours}`} />
            </div>
          </div>

          {/* Loading overlay */}
          {framesLoading && (
            <div style={loadingBarStyle}>
              <div style={loadingBarFillStyle} />
            </div>
          )}

          {/* Main canvas area */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
              opacity: framesLoading ? 0.4 : 1,
              transition: "opacity 0.3s",
            }}
          >
            <div style={canvasWrapStyle}>
              <HeatmapCanvas
                data={currentActual?.data ?? null}
                rows={rows}
                cols={cols}
                colorMode="fire"
                label={`Actual (t=${currentTimeActual})`}
                onHover={handleHoverActual}
              />
            </div>
            <div style={canvasWrapStyle}>
              <HeatmapCanvas
                data={currentPred?.data ?? null}
                rows={rows}
                cols={cols}
                colorMode="prob"
                label={`Predicted (t=${currentTimeActual})`}
                onHover={handleHoverPred}
              />
            </div>
          </div>

          {/* Step navigation */}
          <div style={stepNavStyle}>
            <button
              onClick={stepBackward}
              disabled={currentStep === 0}
              style={navButtonStyle(currentStep === 0)}
              title="Previous step"
            >
              &#9664;
            </button>

            <div style={stepButtonsContainerStyle}>
              {Array.from({ length: clampedNumSteps }, (_, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setPlaying(false);
                    setCurrentStep(i);
                  }}
                  style={stepButtonStyle(i === currentStep)}
                >
                  {i}
                </button>
              ))}
            </div>

            <button
              onClick={stepForward}
              disabled={currentStep >= clampedNumSteps - 1}
              style={navButtonStyle(currentStep >= clampedNumSteps - 1)}
              title="Next step"
            >
              &#9654;
            </button>

            <button onClick={togglePlay} style={playButtonStyle}>
              {playing ? "Pause" : "Play"}
            </button>
          </div>

          {/* Step label */}
          <div style={stepLabelStyle}>
            <span className="mono">{stepLabel}</span>
          </div>

          {/* Divergence metrics */}
          {metrics && (
            <div style={metricsRowStyle}>
              <MetricBadge
                label="F1"
                value={metrics.f1.toFixed(3)}
                highlight
              />
              <MetricBadge
                label="Precision"
                value={metrics.precision.toFixed(3)}
              />
              <MetricBadge label="Recall" value={metrics.recall.toFixed(3)} />
              <MetricBadge
                label="Fire px actual"
                value={String(metrics.actualFirePx)}
              />
              <MetricBadge
                label="Fire px pred"
                value={String(metrics.predFirePx)}
              />
              <MetricBadge label="TP" value={String(metrics.tp)} />
              <MetricBadge label="FP" value={String(metrics.fp)} />
              <MetricBadge label="FN" value={String(metrics.fn)} />
            </div>
          )}

          {/* Color legends + tooltip */}
          <div
            style={{
              display: "flex",
              gap: 24,
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <ColorLegend mode="fire" />
            <ColorLegend mode="prob" />
            {hoverInfo && (
              <HoverTooltip
                row={hoverInfo.row}
                col={hoverInfo.col}
                actualValue={hoverInfo.actual}
                predValue={hoverInfo.pred}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Helper components                                                   */
/* ------------------------------------------------------------------ */

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

function MetricBadge({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div style={metricBadgeStyle(highlight)}>
      <span style={metricBadgeLabelStyle}>{label}</span>
      <span className="mono" style={{ fontSize: "0.9rem" }}>
        {value}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Styles                                                              */
/* ------------------------------------------------------------------ */

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

const emptyStateStyle: React.CSSProperties = {
  padding: "60px 20px",
  textAlign: "center",
  color: "var(--muted)",
};

const controlsRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 24,
  flexWrap: "wrap",
  padding: "12px 16px",
  borderRadius: 14,
  background: "var(--panel)",
  border: "1px solid rgba(246,244,239,0.12)",
  boxShadow: "var(--shadow)",
};

const controlGroupStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
};

const controlLabelStyle: React.CSSProperties = {
  fontSize: "0.8rem",
  color: "var(--muted)",
  whiteSpace: "nowrap",
};

const selectStyle: React.CSSProperties = {
  padding: "6px 12px",
  borderRadius: 8,
  background: "rgba(246,244,239,0.08)",
  border: "1px solid rgba(246,244,239,0.2)",
  color: "var(--text)",
  fontSize: "0.85rem",
};

const canvasWrapStyle: React.CSSProperties = {
  borderRadius: 18,
  background: "linear-gradient(135deg, #15292b, #1b3a3b)",
  padding: 12,
  boxShadow: "var(--shadow)",
  minHeight: 300,
};

const loadingBarStyle: React.CSSProperties = {
  height: 3,
  borderRadius: 2,
  background: "rgba(246,244,239,0.08)",
  overflow: "hidden",
};

const loadingBarFillStyle: React.CSSProperties = {
  height: "100%",
  width: "40%",
  borderRadius: 2,
  background: "var(--accent)",
  animation: "loadSlide 1.2s ease-in-out infinite",
};

const stepNavStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 8,
};

const stepButtonsContainerStyle: React.CSSProperties = {
  display: "flex",
  gap: 4,
  flexWrap: "wrap",
  justifyContent: "center",
};

function stepButtonStyle(active: boolean): React.CSSProperties {
  return {
    width: 36,
    height: 36,
    borderRadius: 8,
    border: active
      ? "2px solid var(--accent)"
      : "1px solid rgba(246,244,239,0.15)",
    background: active ? "rgba(242,107,58,0.2)" : "transparent",
    color: active ? "var(--accent)" : "var(--muted)",
    fontWeight: active ? 700 : 400,
    fontSize: "0.8rem",
    fontFamily: "inherit",
    cursor: "pointer",
    transition: "all 0.15s",
  };
}

function navButtonStyle(disabled: boolean): React.CSSProperties {
  return {
    width: 36,
    height: 36,
    borderRadius: 8,
    border: "1px solid rgba(246,244,239,0.15)",
    background: "transparent",
    color: disabled ? "rgba(246,244,239,0.2)" : "var(--text)",
    fontSize: "0.9rem",
    cursor: disabled ? "default" : "pointer",
    opacity: disabled ? 0.4 : 1,
    transition: "all 0.15s",
  };
}

const playButtonStyle: React.CSSProperties = {
  padding: "6px 16px",
  borderRadius: 8,
  border: "1px solid var(--accent-soft)",
  background: "rgba(242,193,78,0.1)",
  color: "var(--accent-soft)",
  fontSize: "0.8rem",
  fontWeight: 600,
  marginLeft: 8,
  cursor: "pointer",
  transition: "all 0.15s",
};

const stepLabelStyle: React.CSSProperties = {
  textAlign: "center",
  color: "var(--muted)",
  fontSize: "0.8rem",
};

const metricsRowStyle: React.CSSProperties = {
  display: "flex",
  gap: 10,
  flexWrap: "wrap",
  padding: "12px 16px",
  borderRadius: 14,
  background: "var(--panel)",
  border: "1px solid rgba(246,244,239,0.12)",
  boxShadow: "var(--shadow)",
};

function metricBadgeStyle(highlight: boolean): React.CSSProperties {
  return {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 2,
    padding: "6px 12px",
    borderRadius: 10,
    background: highlight ? "rgba(242,107,58,0.12)" : "transparent",
    border: highlight
      ? "1px solid rgba(242,107,58,0.3)"
      : "1px solid transparent",
    minWidth: 60,
  };
}

const metricBadgeLabelStyle: React.CSSProperties = {
  fontSize: "0.6rem",
  color: "var(--muted)",
  textTransform: "uppercase",
  letterSpacing: "0.1em",
};
