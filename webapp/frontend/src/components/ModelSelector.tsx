const MODEL_LABELS: Record<string, string> = {
  logreg: "Logistic Regression",
  mlp: "MLP",
  xgboost: "XGBoost",
  rnn: "GRU+Attention",
  convgru: "ConvGRU U-Net",
};

interface Props {
  available: string[];
  selected: string | null;
  onSelect: (model: string) => void;
}

export function ModelSelector({ available, selected, onSelect }: Props) {
  if (available.length === 0) {
    return <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>No models available</span>;
  }

  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {available.map((m) => (
        <button
          key={m}
          onClick={() => onSelect(m)}
          style={{
            padding: "8px 14px",
            borderRadius: "999px",
            border:
              selected === m
                ? "1px solid var(--accent)"
                : "1px solid rgba(246,244,239,0.2)",
            background:
              selected === m
                ? "rgba(242,107,58,0.15)"
                : "rgba(246,244,239,0.06)",
            color: selected === m ? "var(--accent)" : "var(--text)",
            fontSize: "0.8rem",
            fontWeight: selected === m ? 600 : 400,
            transition: "all 0.15s ease",
          }}
        >
          {MODEL_LABELS[m] ?? m}
        </button>
      ))}
    </div>
  );
}
