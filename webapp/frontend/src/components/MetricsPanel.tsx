import type { FrameMetrics } from "../lib/renderFrame";

interface Props {
  metrics: FrameMetrics | null;
}

export function MetricsPanel({ metrics }: Props) {
  if (!metrics) {
    return (
      <div style={panelStyle}>
        <div style={headerStyle}>Metrics</div>
        <div style={{ color: "var(--muted)", fontSize: "0.8rem" }}>
          Select a model and frame to see metrics
        </div>
      </div>
    );
  }

  const total = metrics.tp + metrics.fp + metrics.fn + metrics.tn;
  const accuracy = total > 0 ? (metrics.tp + metrics.tn) / total : 0;

  return (
    <div style={panelStyle}>
      <div style={headerStyle}>Frame Metrics</div>
      <div style={gridStyle}>
        <MetricRow label="F1 Score" value={metrics.f1.toFixed(4)} highlight />
        <MetricRow label="Precision" value={metrics.precision.toFixed(4)} />
        <MetricRow label="Recall" value={metrics.recall.toFixed(4)} />
        <MetricRow label="Accuracy" value={accuracy.toFixed(4)} />
        <div style={dividerStyle} />
        <MetricRow label="True Pos" value={metrics.tp.toLocaleString()} color="var(--tp-color)" />
        <MetricRow label="False Pos" value={metrics.fp.toLocaleString()} color="var(--fp-color)" />
        <MetricRow label="False Neg" value={metrics.fn.toLocaleString()} color="var(--fn-color)" />
        <MetricRow label="True Neg" value={metrics.tn.toLocaleString()} />
      </div>
    </div>
  );
}

function MetricRow({
  label,
  value,
  highlight,
  color,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  color?: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <span
        style={{
          fontSize: "0.75rem",
          color: "var(--muted)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
        }}
      >
        {label}
      </span>
      <span
        className="mono"
        style={{
          fontSize: highlight ? "1.1rem" : "0.85rem",
          fontWeight: highlight ? 600 : 400,
          color: color ?? (highlight ? "var(--accent)" : "var(--text)"),
        }}
      >
        {value}
      </span>
    </div>
  );
}

function HoverInfo({ row, col, actual, predicted }: any) {
  return null;
}

const panelStyle: React.CSSProperties = {
  padding: "16px",
  borderRadius: 16,
  background: "var(--panel)",
  border: "1px solid rgba(246,244,239,0.12)",
  boxShadow: "var(--shadow)",
};

const headerStyle: React.CSSProperties = {
  fontSize: "0.75rem",
  color: "var(--muted)",
  textTransform: "uppercase",
  letterSpacing: "0.12em",
  marginBottom: 12,
};

const gridStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
};

const dividerStyle: React.CSSProperties = {
  height: 1,
  background: "rgba(246,244,239,0.1)",
  margin: "4px 0",
};
