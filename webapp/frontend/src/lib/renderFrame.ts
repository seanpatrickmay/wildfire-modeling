import { fireColormap, probColormap } from "./colormap";

export type ViewMode = "side-by-side" | "overlay";

export function renderHeatmap(
  ctx: CanvasRenderingContext2D,
  data: Float32Array,
  rows: number,
  cols: number,
  colorFn: (t: number) => [number, number, number]
): void {
  const canvas = ctx.canvas;
  canvas.width = cols;
  canvas.height = rows;

  const imageData = ctx.createImageData(cols, rows);
  const pixels = imageData.data;

  for (let i = 0; i < rows * cols; i++) {
    const v = data[i];
    const ptr = i * 4;
    if (isNaN(v)) {
      pixels[ptr] = 0;
      pixels[ptr + 1] = 0;
      pixels[ptr + 2] = 0;
      pixels[ptr + 3] = 0;
    } else {
      const [r, g, b] = colorFn(v);
      pixels[ptr] = Math.round(r);
      pixels[ptr + 1] = Math.round(g);
      pixels[ptr + 2] = Math.round(b);
      pixels[ptr + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

export function renderOverlay(
  ctx: CanvasRenderingContext2D,
  actual: Float32Array,
  predicted: Float32Array,
  rows: number,
  cols: number,
  threshold: number = 0.1,
  probThreshold: number = 0.5
): void {
  const canvas = ctx.canvas;
  canvas.width = cols;
  canvas.height = rows;

  const imageData = ctx.createImageData(cols, rows);
  const pixels = imageData.data;

  for (let i = 0; i < rows * cols; i++) {
    const ptr = i * 4;
    const aVal = actual[i];
    const pVal = predicted[i];

    if (isNaN(aVal) || isNaN(pVal)) {
      pixels[ptr] = 8;
      pixels[ptr + 1] = 12;
      pixels[ptr + 2] = 13;
      pixels[ptr + 3] = 255;
      continue;
    }

    const isActual = aVal >= threshold;
    const isPredicted = pVal >= probThreshold;

    if (isActual && isPredicted) {
      // True positive — yellow/white
      pixels[ptr] = 255;
      pixels[ptr + 1] = 224;
      pixels[ptr + 2] = 102;
      pixels[ptr + 3] = 255;
    } else if (isActual && !isPredicted) {
      // False negative — red
      pixels[ptr] = 255;
      pixels[ptr + 1] = 74;
      pixels[ptr + 2] = 74;
      pixels[ptr + 3] = 255;
    } else if (!isActual && isPredicted) {
      // False positive — blue
      pixels[ptr] = 74;
      pixels[ptr + 1] = 158;
      pixels[ptr + 2] = 255;
      pixels[ptr + 3] = 255;
    } else {
      // True negative — dark background
      pixels[ptr] = 8;
      pixels[ptr + 1] = 12;
      pixels[ptr + 2] = 13;
      pixels[ptr + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

export interface FrameMetrics {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
}

export function computeMetrics(
  actual: Float32Array,
  predicted: Float32Array,
  threshold: number = 0.1,
  probThreshold: number = 0.5
): FrameMetrics {
  let tp = 0, fp = 0, fn = 0, tn = 0;

  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const p = predicted[i];
    if (isNaN(a) || isNaN(p)) continue;

    const isActual = a >= threshold;
    const isPredicted = p >= probThreshold;

    if (isActual && isPredicted) tp++;
    else if (!isActual && isPredicted) fp++;
    else if (isActual && !isPredicted) fn++;
    else tn++;
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { tp, fp, fn, tn, precision, recall, f1 };
}
