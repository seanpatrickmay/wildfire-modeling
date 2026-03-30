import { useEffect, useRef } from "react";
import { fireColormap, probColormap } from "../lib/colormap";

interface Props {
  mode: "fire" | "prob" | "overlay";
  style?: React.CSSProperties;
}

export function ColorLegend({ mode, style }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (mode === "overlay") return;

    const w = 200;
    const h = 12;
    canvas.width = w;
    canvas.height = h;

    const imageData = ctx.createImageData(w, h);
    const colorFn = mode === "fire" ? fireColormap : probColormap;

    for (let x = 0; x < w; x++) {
      const t = x / (w - 1);
      const [r, g, b] = colorFn(t);
      for (let y = 0; y < h; y++) {
        const ptr = (y * w + x) * 4;
        imageData.data[ptr] = Math.round(r);
        imageData.data[ptr + 1] = Math.round(g);
        imageData.data[ptr + 2] = Math.round(b);
        imageData.data[ptr + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [mode]);

  if (mode === "overlay") {
    return (
      <div
        style={{
          display: "flex",
          gap: 12,
          fontSize: "0.7rem",
          color: "var(--muted)",
          ...style,
        }}
      >
        <span><span style={{ color: "var(--tp-color)" }}>{"■"}</span> True Positive</span>
        <span><span style={{ color: "var(--fn-color)" }}>{"■"}</span> False Negative</span>
        <span><span style={{ color: "var(--fp-color)" }}>{"■"}</span> False Positive</span>
        <span><span style={{ color: "#555" }}>{"■"}</span> True Negative</span>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, ...style }}>
      <span className="mono" style={{ fontSize: "0.65rem", color: "var(--muted)" }}>
        0
      </span>
      <canvas
        ref={canvasRef}
        style={{
          width: 140,
          height: 10,
          borderRadius: 5,
        }}
      />
      <span className="mono" style={{ fontSize: "0.65rem", color: "var(--muted)" }}>
        1
      </span>
      <span style={{ fontSize: "0.65rem", color: "var(--muted)", marginLeft: 4 }}>
        {mode === "fire" ? "Confidence" : "Probability"}
      </span>
    </div>
  );
}
