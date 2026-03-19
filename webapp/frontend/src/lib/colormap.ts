function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(Math.max(v, min), max);
}

const FIRE_STOPS = [
  { t: 0.0, color: [8, 12, 13] },
  { t: 0.35, color: [44, 90, 86] },
  { t: 0.6, color: [242, 107, 58] },
  { t: 0.85, color: [242, 193, 78] },
  { t: 1.0, color: [255, 241, 205] },
] as const;

export function fireColormap(t: number): [number, number, number] {
  t = clamp(t, 0, 1);
  for (let i = 0; i < FIRE_STOPS.length - 1; i++) {
    const left = FIRE_STOPS[i];
    const right = FIRE_STOPS[i + 1];
    if (t >= left.t && t <= right.t) {
      const local = (t - left.t) / (right.t - left.t);
      return [
        lerp(left.color[0], right.color[0], local),
        lerp(left.color[1], right.color[1], local),
        lerp(left.color[2], right.color[2], local),
      ];
    }
  }
  const last = FIRE_STOPS[FIRE_STOPS.length - 1];
  return [last.color[0], last.color[1], last.color[2]];
}

const PROB_STOPS = [
  { t: 0.0, color: [8, 12, 30] },
  { t: 0.3, color: [30, 70, 160] },
  { t: 0.5, color: [60, 140, 220] },
  { t: 0.7, color: [100, 200, 240] },
  { t: 0.85, color: [180, 230, 255] },
  { t: 1.0, color: [240, 250, 255] },
] as const;

export function probColormap(t: number): [number, number, number] {
  t = clamp(t, 0, 1);
  for (let i = 0; i < PROB_STOPS.length - 1; i++) {
    const left = PROB_STOPS[i];
    const right = PROB_STOPS[i + 1];
    if (t >= left.t && t <= right.t) {
      const local = (t - left.t) / (right.t - left.t);
      return [
        lerp(left.color[0], right.color[0], local),
        lerp(left.color[1], right.color[1], local),
        lerp(left.color[2], right.color[2], local),
      ];
    }
  }
  const last = PROB_STOPS[PROB_STOPS.length - 1];
  return [last.color[0], last.color[1], last.color[2]];
}
