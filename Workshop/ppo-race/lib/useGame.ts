"use client";

import { useEffect, useRef, useState } from "react";
import type { StateResponse } from "./types";

/** Poll the game state every second (Vercel-friendly: no websockets needed). */
export function useGame(
  pin: string,
  query: { hostKey?: string; playerId?: string }
) {
  const [state, setState] = useState<StateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** serverNow - clientNow, so timers/race sync to the server clock */
  const offsetRef = useRef(0);
  const { hostKey, playerId } = query;

  useEffect(() => {
    if (!pin) return;
    let alive = true;
    const params = new URLSearchParams();
    if (hostKey) params.set("hostKey", hostKey);
    if (playerId) params.set("playerId", playerId);
    const url = `/api/game/${pin}/state?${params.toString()}`;

    const tick = async () => {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (!alive) return;
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          setError(body.error ?? `Fout ${res.status}`);
          return;
        }
        const data: StateResponse = await res.json();
        offsetRef.current = data.serverNow - Date.now();
        setState(data);
        setError(null);
      } catch {
        // transient network error: keep last state, retry on next tick
      }
    };

    tick();
    const id = setInterval(tick, 1000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [pin, hostKey, playerId]);

  return { state, error, offsetRef };
}

export async function postJson<T = Record<string, unknown>>(
  url: string,
  body: unknown
): Promise<{ ok: boolean; status: number; data: T & { error?: string } }> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}
