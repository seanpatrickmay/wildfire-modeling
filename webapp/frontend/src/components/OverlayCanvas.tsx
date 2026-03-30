import { useEffect, useRef } from "react";
import { renderOverlay } from "../lib/renderFrame";

interface Props {
  actual: Float32Array | null;
  predicted: Float32Array | null;
  rows: number;
  cols: number;
  threshold?: number;
  probThreshold?: number;
  onHover?: (row: number, col: number, actualVal: number, predVal: number) => void;
}

export function OverlayCanvas({
  actual,
  predicted,
  rows,
  cols,
  threshold = 0.1,
  probThreshold = 0.5,
  onHover,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !actual || !predicted || rows === 0 || cols === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    renderOverlay(ctx, actual, predicted, rows, cols, threshold, probThreshold);
  }, [actual, predicted, rows, cols, threshold, probThreshold]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onHover || !actual || !predicted || rows === 0 || cols === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = cols / rect.width;
    const scaleY = rows / rect.height;
    const col = Math.floor((e.clientX - rect.left) * scaleX);
    const row = Math.floor((e.clientY - rect.top) * scaleY);
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      const idx = row * cols + col;
      onHover(row, col, actual[idx], predicted[idx]);
    }
  };

  return (
    <div style={{ position: "relative" }}>
      <div
        style={{
          position: "absolute",
          top: 8,
          left: 10,
          background: "rgba(0,0,0,0.6)",
          padding: "4px 10px",
          borderRadius: 8,
          fontSize: "0.75rem",
          color: "var(--muted)",
          zIndex: 2,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        }}
      >
        Overlay
      </div>
      <div
        style={{
          position: "absolute",
          bottom: 8,
          left: 10,
          background: "rgba(0,0,0,0.6)",
          padding: "6px 10px",
          borderRadius: 8,
          fontSize: "0.65rem",
          zIndex: 2,
          display: "flex",
          gap: 10,
        }}
      >
        <span>
          <span style={{ color: "var(--tp-color)" }}>{"■"}</span> TP
        </span>
        <span>
          <span style={{ color: "var(--fn-color)" }}>{"■"}</span> FN
        </span>
        <span>
          <span style={{ color: "var(--fp-color)" }}>{"■"}</span> FP
        </span>
      </div>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        style={{
          width: "100%",
          height: "100%",
          imageRendering: "pixelated",
          borderRadius: 14,
          background: "#0b1314",
          border: "1px solid rgba(246,244,239,0.08)",
          display: "block",
        }}
      />
    </div>
  );
}
