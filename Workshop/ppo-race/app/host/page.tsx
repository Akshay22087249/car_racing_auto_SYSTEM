"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { postJson } from "@/lib/useGame";

export default function HostStart() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const create = async () => {
    setBusy(true);
    setError(null);
    const { ok, data } = await postJson<{ pin: string; hostKey: string }>(
      "/api/game",
      {}
    );
    if (!ok || !data.pin) {
      setError(data.error ?? "Game aanmaken mislukt");
      setBusy(false);
      return;
    }
    localStorage.setItem(`ppo:host:${data.pin}`, data.hostKey);
    // keep the key in the URL so the host screen can be reopened elsewhere
    router.push(`/host/${data.pin}?key=${data.hostKey}`);
  };

  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-8 p-8 text-center">
      <h1 className="text-5xl font-black tracking-tight">
        🏎️ PPO <span className="text-amber-400">Race</span> — Host
      </h1>
      <p className="max-w-xl text-lg text-slate-300">
        Maak een game aan, zet dit scherm op de beamer en laat deelnemers
        joinen met de PIN. Drie rondes hyperparameter-tuning, dan de race!
      </p>
      <button
        onClick={create}
        disabled={busy}
        className="rounded-2xl bg-amber-400 px-10 py-5 text-2xl font-bold text-slate-900 shadow-lg transition hover:bg-amber-300 disabled:opacity-50"
      >
        {busy ? "Bezig…" : "Nieuwe game starten"}
      </button>
      {error && <p className="text-red-400">{error}</p>}
    </main>
  );
}
