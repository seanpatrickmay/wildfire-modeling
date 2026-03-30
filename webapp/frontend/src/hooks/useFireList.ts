import { useEffect, useState } from "react";
import { fetchFires, type FireSummary } from "../api/client";

export function useFireList() {
  const [fires, setFires] = useState<FireSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchFires()
      .then(setFires)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { fires, loading, error };
}
