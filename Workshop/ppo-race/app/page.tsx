"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { TEAMS, TEAM_CAPACITY } from "@/lib/teams";
import { postJson } from "@/lib/useGame";
import type { PlayerPublic } from "@/lib/types";

export default function JoinPage() {
  const router = useRouter();
  const [pin, setPin] = useState("");
  const [name, setName] = useState("");
  const [team, setTeam] = useState<string | null>(null);
  const [roster, setRoster] = useState<PlayerPublic[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // a scanned QR lands here with ?pin=1234 prefilled
    const p = new URLSearchParams(window.location.search).get("pin");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- URL is client-only
    if (p) setPin(p.replace(/\D/g, "").slice(0, 4));
  }, []);

  // once a full PIN is entered, poll the roster so seat availability stays fresh
  useEffect(() => {
    if (pin.length !== 4) return;
    let alive = true;
    const load = async () => {
      try {
        const res = await fetch(`/api/game/${pin}/state`, { cache: "no-store" });
        if (!alive) return;
        setRoster(res.ok ? (await res.json()).players ?? [] : []);
      } catch {
        // keep last roster on a transient error
      }
    };
    load();
    const id = setInterval(load, 2500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [pin]);

  const seatsByTeam = useMemo(() => {
    const m: Record<string, PlayerPublic[]> = {};
    for (const p of roster ?? []) {
      if (p.team) (m[p.team] ??= []).push(p);
    }
    return m;
  }, [roster]);

  const join = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy || pin.length < 4 || !name.trim() || !team) return;
    setBusy(true);
    setError(null);
    const { ok, data } = await postJson<{ playerId: string }>(
      `/api/game/${pin}/join`,
      { name: name.trim(), team }
    );
    if (!ok || !data.playerId) {
      setError(data.error ?? "Joinen mislukt");
      setBusy(false);
      return;
    }
    localStorage.setItem(`ppo:player:${pin}`, data.playerId);
    router.push(`/play/${pin}`);
  };

  return (
    <main className="flex flex-1 flex-col items-center gap-6 p-6 text-center">
      <h1 className="mt-4 text-5xl font-black tracking-tight">
        🏎️ PPO <span className="text-amber-400">Race</span>
      </h1>
      <p className="max-w-sm text-slate-300">
        Kies je F1-team, tune je agent en race tegen de rest!
      </p>

      <form onSubmit={join} className="flex w-full max-w-xl flex-col items-center gap-4">
        <div className="flex w-full max-w-xs flex-col gap-4">
          <input
            value={pin}
            onChange={(e) => setPin(e.target.value.replace(/\D/g, "").slice(0, 4))}
            inputMode="numeric"
            placeholder="Game-PIN"
            className="rounded-xl bg-slate-800 px-4 py-4 text-center text-2xl font-bold tracking-widest tabular-nums outline-none ring-amber-400 focus:ring-2 placeholder:text-slate-500 placeholder:tracking-normal placeholder:font-normal placeholder:text-lg"
          />
          <input
            value={name}
            onChange={(e) => setName(e.target.value.slice(0, 16))}
            placeholder="Je naam"
            className="rounded-xl bg-slate-800 px-4 py-4 text-center text-xl font-semibold outline-none ring-amber-400 focus:ring-2 placeholder:text-slate-500 placeholder:font-normal placeholder:text-lg"
          />
        </div>

        {pin.length === 4 && (
          <div className="w-full">
            <p className="mb-2 text-sm font-semibold text-slate-300">
              Kies je team (max {TEAM_CAPACITY} coureurs)
            </p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {TEAMS.map((t) => {
                const occ = seatsByTeam[t.id] ?? [];
                const full = occ.length >= TEAM_CAPACITY;
                const selected = team === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    disabled={full}
                    onClick={() => setTeam(t.id)}
                    className={`flex flex-col gap-1 rounded-xl border-l-4 px-3 py-2 text-left transition ${
                      selected ? "ring-2 ring-amber-400" : ""
                    } ${full ? "cursor-not-allowed opacity-40" : "hover:bg-slate-800"}`}
                    style={{
                      borderColor: t.color,
                      background: selected
                        ? `${t.color}26`
                        : "rgba(30,41,59,0.6)",
                    }}
                  >
                    <span className="flex items-center justify-between gap-1">
                      <span className="text-sm font-bold">{t.name}</span>
                      {full && (
                        <span className="text-[10px] font-bold text-red-400">
                          VOL
                        </span>
                      )}
                    </span>
                    <span className="flex flex-col gap-0.5 text-[11px] text-slate-400">
                      {[0, 1].map((i) => (
                        <span key={i} className="flex items-center gap-1">
                          <span
                            className="inline-block h-2 w-2 shrink-0 rounded-full"
                            style={{
                              background: occ[i]
                                ? t.color
                                : "rgba(148,163,184,0.3)",
                            }}
                          />
                          <span className="truncate">
                            {occ[i] ? occ[i].name : "vrij"}
                          </span>
                        </span>
                      ))}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={busy || pin.length < 4 || !name.trim() || !team}
          className="w-full max-w-xs rounded-xl bg-amber-400 px-6 py-4 text-xl font-bold text-slate-900 hover:bg-amber-300 disabled:opacity-40"
        >
          {busy ? "Bezig…" : "Meedoen!"}
        </button>
        {error && <p className="text-red-400">{error}</p>}
      </form>

      <Link href="/host" className="text-sm text-slate-500 underline">
        Workshop-host? Start hier een game
      </Link>
    </main>
  );
}
