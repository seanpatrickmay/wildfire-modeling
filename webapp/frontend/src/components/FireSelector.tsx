import type { FireSummary } from "../api/client";

interface Props {
  fires: FireSummary[];
  selected: string | null;
  onSelect: (name: string) => void;
}

export function FireSelector({ fires, selected, onSelect }: Props) {
  return (
    <select
      value={selected ?? ""}
      onChange={(e) => onSelect(e.target.value)}
      style={{
        padding: "10px 14px",
        borderRadius: "999px",
        background: "rgba(246,244,239,0.08)",
        border: "1px solid rgba(246,244,239,0.2)",
        color: "var(--text)",
        fontSize: "0.85rem",
        minWidth: 200,
      }}
    >
      <option value="" disabled>
        Select a fire...
      </option>
      {fires.map((f) => (
        <option key={f.name} value={f.name}>
          {f.name.replace(/_/g, " ")} ({f.grid_shape[0]}x{f.grid_shape[1]},{" "}
          {f.n_hours} hours)
        </option>
      ))}
    </select>
  );
}
