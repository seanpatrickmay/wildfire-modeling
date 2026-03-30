const API_BASE = "/api";

export interface FireSummary {
  name: string;
  grid_shape: [number, number];
  n_hours: number;
  smoothed_fire_pixels: number[];
}

export interface ModelInfo {
  name: string;
  frame_count: number;
  time_indices: number[];
}

export interface FireMeta {
  name: string;
  grid_shape: [number, number];
  crs: string | null;
  geo_transform: number[] | null;
  time_steps: string[];
  available_models: ModelInfo[];
}

export interface FrameData {
  data: Float32Array;
  rows: number;
  cols: number;
  timeIndex: number;
  timeStep: string;
}

export interface ProgressionMeta {
  actual: Array<{ t: number; fire_pixels: number }>;
  predicted: Array<{ t: number; fire_pixels: number }>;
  grid_shape: [number, number];
  start: number;
  steps: number;
}

export async function fetchFires(): Promise<FireSummary[]> {
  const res = await fetch(`${API_BASE}/fires`);
  if (!res.ok) throw new Error(`Failed to fetch fires: ${res.status}`);
  return res.json();
}

export async function fetchFireMeta(name: string): Promise<FireMeta> {
  const res = await fetch(`${API_BASE}/fires/${name}/meta`);
  if (!res.ok) throw new Error(`Failed to fetch fire meta: ${res.status}`);
  return res.json();
}

async function fetchBinaryFrame(url: string): Promise<FrameData> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch frame: ${res.status}`);
  const buffer = await res.arrayBuffer();
  return {
    data: new Float32Array(buffer),
    rows: parseInt(res.headers.get("X-Rows") || "0"),
    cols: parseInt(res.headers.get("X-Cols") || "0"),
    timeIndex: parseInt(res.headers.get("X-Time-Index") || "0"),
    timeStep: res.headers.get("X-Time-Step") || "",
  };
}

export function fetchGroundTruth(
  fireName: string,
  t: number
): Promise<FrameData> {
  return fetchBinaryFrame(`${API_BASE}/fires/${fireName}/frame/${t}`);
}

export function fetchPrediction(
  fireName: string,
  model: string,
  t: number
): Promise<FrameData> {
  return fetchBinaryFrame(
    `${API_BASE}/fires/${fireName}/prediction/${model}/${t}`
  );
}

export async function fetchProgression(
  fireName: string,
  start: number,
  steps: number
): Promise<ProgressionMeta> {
  const res = await fetch(
    `${API_BASE}/fires/${fireName}/progression?start=${start}&steps=${steps}`
  );
  if (!res.ok) throw new Error(`Failed to fetch progression: ${res.status}`);
  return res.json();
}

export function fetchActualFrame(
  fireName: string,
  t: number
): Promise<FrameData> {
  return fetchBinaryFrame(
    `${API_BASE}/fires/${fireName}/progression/actual/${t}`
  );
}

export function fetchPredictedFrame(
  fireName: string,
  start: number,
  step: number
): Promise<FrameData> {
  return fetchBinaryFrame(
    `${API_BASE}/fires/${fireName}/progression/predicted?start=${start}&step=${step}`
  );
}
