import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchFireMeta,
  fetchGroundTruth,
  fetchPrediction,
  type FireMeta,
  type FrameData,
} from "../api/client";

interface FireDataState {
  meta: FireMeta | null;
  loading: boolean;
  error: string | null;
  actualFrame: FrameData | null;
  predFrame: FrameData | null;
  frameLoading: boolean;
}

export function useFireData(fireName: string | null, model: string | null) {
  const [state, setState] = useState<FireDataState>({
    meta: null,
    loading: false,
    error: null,
    actualFrame: null,
    predFrame: null,
    frameLoading: false,
  });

  const cacheRef = useRef<Map<string, FrameData>>(new Map());

  useEffect(() => {
    if (!fireName) return;
    setState((s) => ({ ...s, loading: true, error: null, meta: null }));
    cacheRef.current.clear();
    fetchFireMeta(fireName)
      .then((meta) => setState((s) => ({ ...s, meta, loading: false })))
      .catch((e) =>
        setState((s) => ({ ...s, error: e.message, loading: false }))
      );
  }, [fireName]);

  const loadFrame = useCallback(
    async (t: number) => {
      if (!fireName || !state.meta) return;

      setState((s) => ({ ...s, frameLoading: true }));

      try {
        const actualKey = `actual:${fireName}:${t}`;
        let actual = cacheRef.current.get(actualKey) ?? null;
        if (!actual) {
          actual = await fetchGroundTruth(fireName, t);
          cacheRef.current.set(actualKey, actual);
        }

        let pred: FrameData | null = null;
        if (model) {
          const predKey = `pred:${fireName}:${model}:${t}`;
          pred = cacheRef.current.get(predKey) ?? null;
          if (!pred) {
            try {
              pred = await fetchPrediction(fireName, model, t);
              cacheRef.current.set(predKey, pred);
            } catch {
              // prediction may not exist for this time index
            }
          }
        }

        setState((s) => ({
          ...s,
          actualFrame: actual,
          predFrame: pred,
          frameLoading: false,
        }));

        // Prefetch next few frames
        for (let ahead = 1; ahead <= 5; ahead++) {
          const futureT = t + ahead;
          if (futureT >= (state.meta?.time_steps.length ?? 0)) break;
          const ak = `actual:${fireName}:${futureT}`;
          if (!cacheRef.current.has(ak)) {
            fetchGroundTruth(fireName, futureT).then((f) =>
              cacheRef.current.set(ak, f)
            ).catch(() => {});
          }
          if (model) {
            const pk = `pred:${fireName}:${model}:${futureT}`;
            if (!cacheRef.current.has(pk)) {
              fetchPrediction(fireName, model, futureT).then((f) =>
                cacheRef.current.set(pk, f)
              ).catch(() => {});
            }
          }
        }
      } catch (e: any) {
        setState((s) => ({ ...s, frameLoading: false }));
      }
    },
    [fireName, model, state.meta]
  );

  return { ...state, loadFrame };
}
