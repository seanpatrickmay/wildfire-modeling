import { useEffect, useRef } from "react";
import { renderHeatmap } from "../lib/renderFrame";
import { fireColormap, probColormap } from "../lib/colormap";

interface Props {
  data: Float32Array | null;
  rows: number;
  cols: number;
  colorMode: "fire" | "prob";
  label?: string;
  onHover?: (row: number, col: number, value: number) => void;
}

export function HeatmapCanvas({ data, rows, cols, colorMode, label, onHover }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || rows === 0 || cols === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const colorFn = colorMode === "fire" ? fireColormap : probColormap;
    renderHeatmap(ctx, data, rows, cols, colorFn);
  }, [data, rows, cols, colorMode]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onHover || !data || rows === 0 || cols === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = cols / rect.width;
    const scaleY = rows / rect.height;
    const col = Math.floor((e.clientX - rect.left) * scaleX);
    const row = Math.floor((e.clientY - rect.top) * scaleY);
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      onHover(row, col, data[row * cols + col]);
    }
  };

  return (
    <div style={{ position: "relative" }}>
      {label && (
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
          {label}
        </div>
      )}
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
