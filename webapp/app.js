const fileInput = document.getElementById("fileInput");
const clearBtn = document.getElementById("clearBtn");
const canvas = document.getElementById("matrixCanvas");
const ctx = canvas.getContext("2d");
const timeSlider = document.getElementById("timeSlider");
const timeLabel = document.getElementById("timeLabel");
const emptyState = document.getElementById("emptyState");

const metaFire = document.getElementById("metaFire");
const metaSource = document.getElementById("metaSource");
const metaGrid = document.getElementById("metaGrid");
const metaPixel = document.getElementById("metaPixel");
const metaCrs = document.getElementById("metaCrs");
const metaTime = document.getElementById("metaTime");

const state = {
  metadata: null,
  data: null,
  min: 0,
  max: 1,
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorStop(t) {
  const stops = [
    { t: 0.0, color: [8, 12, 13] },
    { t: 0.35, color: [44, 90, 86] },
    { t: 0.6, color: [242, 107, 58] },
    { t: 0.85, color: [242, 193, 78] },
    { t: 1.0, color: [255, 241, 205] },
  ];
  for (let i = 0; i < stops.length - 1; i += 1) {
    const left = stops[i];
    const right = stops[i + 1];
    if (t >= left.t && t <= right.t) {
      const local = (t - left.t) / (right.t - left.t);
      return [
        lerp(left.color[0], right.color[0], local),
        lerp(left.color[1], right.color[1], local),
        lerp(left.color[2], right.color[2], local),
      ];
    }
  }
  return stops[stops.length - 1].color;
}

function computeMinMax(data) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const frame of data) {
    for (const row of frame) {
      for (const value of row) {
        if (value === null || Number.isNaN(value)) continue;
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }
  if (min === max) {
    return { min, max: min + 1 };
  }
  return { min, max };
}

function updateMetadata(meta) {
  metaFire.textContent = `${meta.fire_name} (${meta.year})`;
  metaSource.textContent = meta.source || "—";
  metaGrid.textContent = `${meta.grid_shape[0]} x ${meta.grid_shape[1]}`;
  metaPixel.textContent = meta.pixel_size_m
    ? `${meta.pixel_size_m[0].toFixed(2)}m`
    : "—";
  metaCrs.textContent = meta.crs || "—";
  metaTime.textContent = meta.temporal_resolution || "—";
}

function renderFrame(index) {
  if (!state.data || !state.metadata) return;
  const frame = state.data[index];
  if (!frame) return;

  const rows = frame.length;
  const cols = frame[0].length;
  canvas.width = cols;
  canvas.height = rows;

  const imageData = ctx.createImageData(cols, rows);
  const data = imageData.data;
  const range = state.max - state.min;

  let ptr = 0;
  for (let y = 0; y < rows; y += 1) {
    const row = frame[y];
    for (let x = 0; x < cols; x += 1) {
      const value = row[x];
      if (value === null || Number.isNaN(value)) {
        data[ptr] = 0;
        data[ptr + 1] = 0;
        data[ptr + 2] = 0;
        data[ptr + 3] = 0;
      } else {
        const t = clamp((value - state.min) / range, 0, 1);
        const [r, g, b] = colorStop(t);
        data[ptr] = Math.round(r);
        data[ptr + 1] = Math.round(g);
        data[ptr + 2] = Math.round(b);
        data[ptr + 3] = 255;
      }
      ptr += 4;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  const label = state.metadata.time_steps
    ? state.metadata.time_steps[index]
    : `Timestep ${index + 1}`;
  timeLabel.textContent = label;
}

function loadJson(json) {
  if (!json || !json.data || !json.metadata) {
    alert("Invalid JSON format. Expected { metadata, data }.");
    return;
  }
  state.metadata = json.metadata;
  state.data = json.data;
  const { min, max } = computeMinMax(state.data);
  state.min = min;
  state.max = max;

  updateMetadata(state.metadata);

  timeSlider.min = 0;
  timeSlider.max = Math.max(state.data.length - 1, 0);
  timeSlider.value = 0;

  emptyState.style.display = "none";
  renderFrame(0);
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (loadEvent) => {
    try {
      const json = JSON.parse(loadEvent.target.result);
      loadJson(json);
    } catch (error) {
      alert("Failed to parse JSON. Please verify the file.");
    }
  };
  reader.readAsText(file);
});

clearBtn.addEventListener("click", () => {
  fileInput.value = "";
  state.data = null;
  state.metadata = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  emptyState.style.display = "flex";
  timeLabel.textContent = "Load data to begin";
  metaFire.textContent = "—";
  metaSource.textContent = "—";
  metaGrid.textContent = "—";
  metaPixel.textContent = "—";
  metaCrs.textContent = "—";
  metaTime.textContent = "—";
});

timeSlider.addEventListener("input", (event) => {
  const index = Number(event.target.value);
  renderFrame(index);
});
