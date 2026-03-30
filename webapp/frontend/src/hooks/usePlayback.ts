import { useCallback, useEffect, useRef, useState } from "react";

interface PlaybackState {
  playing: boolean;
  speed: number;
  currentTime: number;
}

export function usePlayback(
  maxTime: number,
  onTimeChange: (t: number) => void
) {
  const [state, setState] = useState<PlaybackState>({
    playing: false,
    speed: 1,
    currentTime: 0,
  });

  const stateRef = useRef(state);
  stateRef.current = state;

  const rafRef = useRef<number>(0);
  const lastTickRef = useRef<number>(0);

  const tick = useCallback(() => {
    const now = performance.now();
    const elapsed = now - lastTickRef.current;
    const interval = 1000 / stateRef.current.speed;

    if (elapsed >= interval) {
      lastTickRef.current = now;
      setState((s) => {
        const next = s.currentTime + 1;
        if (next >= maxTime) {
          return { ...s, playing: false, currentTime: maxTime - 1 };
        }
        onTimeChange(next);
        return { ...s, currentTime: next };
      });
    }

    if (stateRef.current.playing) {
      rafRef.current = requestAnimationFrame(tick);
    }
  }, [maxTime, onTimeChange]);

  useEffect(() => {
    if (state.playing) {
      lastTickRef.current = performance.now();
      rafRef.current = requestAnimationFrame(tick);
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [state.playing, tick]);

  const play = useCallback(() => setState((s) => ({ ...s, playing: true })), []);
  const pause = useCallback(() => setState((s) => ({ ...s, playing: false })), []);
  const togglePlay = useCallback(() => setState((s) => ({ ...s, playing: !s.playing })), []);

  const setTime = useCallback(
    (t: number) => {
      setState((s) => ({ ...s, currentTime: t }));
      onTimeChange(t);
    },
    [onTimeChange]
  );

  const setSpeed = useCallback(
    (speed: number) => setState((s) => ({ ...s, speed })),
    []
  );

  const stepForward = useCallback(() => {
    setState((s) => {
      const next = Math.min(s.currentTime + 1, maxTime - 1);
      onTimeChange(next);
      return { ...s, currentTime: next, playing: false };
    });
  }, [maxTime, onTimeChange]);

  const stepBackward = useCallback(() => {
    setState((s) => {
      const prev = Math.max(s.currentTime - 1, 0);
      onTimeChange(prev);
      return { ...s, currentTime: prev, playing: false };
    });
  }, [onTimeChange]);

  return {
    ...state,
    play,
    pause,
    togglePlay,
    setTime,
    setSpeed,
    stepForward,
    stepBackward,
  };
}
