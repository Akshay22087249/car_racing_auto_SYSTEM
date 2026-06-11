"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { TOTAL_ROUNDS } from "@/lib/constants";
import { postJson } from "@/lib/useGame";

export default function HostStart() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [password, setPassword] = useState("");

  useEffect(() => {
    // remember the host password locally so you don't retype it each game
    const saved = localStorage.getItem("ppo:hostpw");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- localStorage is client-only
    if (saved) setPassword(saved);
  }, []);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    const { ok, data } = await postJson<{ pin: string; hostKey: string }>(
      "/api/game",
      { password }
    );
    if (!ok || !data.pin) {
      setError(data.error ?? "Game aanmaken mislukt");
      setBusy(false);
      return;
    }
    localStorage.setItem("ppo:hostpw", password);
    localStorage.setItem(`ppo:host:${data.pin}`, data.hostKey);
    router.push(`/host/${data.pin}?key=${data.hostKey}`);
  };

  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-8 p-8 text-center">
      <h1 className="text-5xl font-black tracking-tight">
        🏎️ PPO <span className="text-amber-400">Race</span> Host
      </h1>
      <p className="max-w-xl text-lg text-slate-300">
        Maak een game aan, zet dit scherm op de beamer en laat deelnemers joinen
        met de PIN. {TOTAL_ROUNDS} rondes hyperparameter-tuning, dan de race!
      </p>
      <form onSubmit={create} className="flex w-full max-w-xs flex-col gap-4">
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Host-wachtwoord"
          autoComplete="current-password"
          className="rounded-xl bg-slate-800 px-4 py-4 text-center text-xl font-semibold outline-none ring-amber-400 focus:ring-2 placeholder:text-slate-500 placeholder:font-normal placeholder:text-lg"
        />
        <button
          type="submit"
          disabled={busy}
          className="rounded-2xl bg-amber-400 px-10 py-5 text-2xl font-bold text-slate-900 shadow-lg transition hover:bg-amber-300 disabled:opacity-50"
        >
          {busy ? "Bezig…" : "Nieuwe game starten"}
        </button>
      </form>
      {error && <p className="text-red-400">{error}</p>}
      <p className="max-w-xs text-xs text-slate-500">
        Het wachtwoord is alleen nodig op de online versie. Lokaal mag het leeg
        blijven.
      </p>
    </main>
  );
}
