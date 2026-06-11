"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { postJson } from "@/lib/useGame";

export default function JoinPage() {
  const router = useRouter();
  const [pin, setPin] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const join = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    const { ok, data } = await postJson<{ playerId: string }>(
      `/api/game/${pin.trim()}/join`,
      { name: name.trim() }
    );
    if (!ok || !data.playerId) {
      setError(data.error ?? "Joinen mislukt");
      setBusy(false);
      return;
    }
    localStorage.setItem(`ppo:player:${pin.trim()}`, data.playerId);
    router.push(`/play/${pin.trim()}`);
  };

  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-8 p-6 text-center">
      <h1 className="text-5xl font-black tracking-tight">
        🏎️ PPO <span className="text-amber-400">Race</span>
      </h1>
      <p className="text-slate-300 max-w-sm">
        Tune je PPO-agent, ronde voor ronde — en race daarna tegen de rest!
      </p>
      <form onSubmit={join} className="flex w-full max-w-xs flex-col gap-4">
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
        <button
          type="submit"
          disabled={busy || pin.length < 4 || !name.trim()}
          className="rounded-xl bg-amber-400 px-6 py-4 text-xl font-bold text-slate-900 hover:bg-amber-300 disabled:opacity-40"
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
