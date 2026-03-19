interface Props {
  row: number;
  col: number;
  actualValue: number;
  predValue: number | null;
}

export function HoverTooltip({ row, col, actualValue, predValue }: Props) {
  return (
    <div
      style={{
        padding: "8px 12px",
        borderRadius: 10,
        background: "rgba(0,0,0,0.7)",
        border: "1px solid rgba(246,244,239,0.15)",
        fontSize: "0.75rem",
        display: "flex",
        gap: 12,
      }}
    >
      <span className="mono" style={{ color: "var(--muted)" }}>
        [{row},{col}]
      </span>
      <span className="mono">
        actual: {isNaN(actualValue) ? "NaN" : actualValue.toFixed(3)}
      </span>
      {predValue !== null && (
        <span className="mono">
          pred: {isNaN(predValue) ? "NaN" : predValue.toFixed(3)}
        </span>
      )}
    </div>
  );
}
