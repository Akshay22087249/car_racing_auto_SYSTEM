"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Countdown from "@/components/Countdown";
import CurveChart from "@/components/CurveChart";
import { CLIP_OPTIONS } from "@/lib/constants";
import { VERDICT_HINTS, VERDICT_LABELS } from "@/lib/sim";
import { useGame, postJson } from "@/lib/useGame";
import type { RoundRecord, StateResponse } from "@/lib/types";

export default function PlayerScreen() {
  const { pin } = useParams<{ pin: string }>();
  const router = useRouter();
  const [playerId, setPlayerId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [justSubmitted, setJustSubmitted] = useState<RoundRecord | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    // ?pid=… lets a player continue on another device
    const urlId = new URLSearchParams(window.location.search).get("pid");
    if (urlId) localStorage.setItem(`ppo:player:${pin}`, urlId);
    const id = urlId ?? localStorage.getItem(`ppo:player:${pin}`);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- localStorage is client-only
    if (id) setPlayerId(id);
    else router.replace("/");
  }, [pin, router]);

  const { state, error, offsetRef } = useGame(pin ?? "", {
    playerId: playerId ?? undefined,
  });

  // reset per-round local state when a new round starts
  const round = state?.round;
  const [seenRound, setSeenRound] = useState(round);
  if (round !== seenRound) {
    setSeenRound(round);
    setJustSubmitted(null);
    setSubmitError(null);
  }

  const pick = async (clip: number) => {
    if (busy || !playerId) return;
    setBusy(true);
    setSubmitError(null);
    const { ok, data } = await postJson<{ record: RoundRecord }>(
      `/api/game/${pin}/answer`,
      { playerId, clip }
    );
    if (ok && data.record) setJustSubmitted(data.record);
    else setSubmitError(data.error ?? "Versturen mislukt");
    setBusy(false);
  };

  if (error) {
    return (
      <Center>
        <p className="text-red-400">{error}</p>
      </Center>
    );
  }
  if (!state || !state.me) {
    return (
      <Center>
        <p className="text-slate-400 animate-pulse">Laden…</p>
      </Center>
    );
  }

  const me = state.me;
  const record = me.currentRound ?? justSubmitted;

  return (
    <main className="flex flex-1 flex-col gap-5 p-5 max-w-md w-full mx-auto">
      <header className="flex items-center gap-2">
        <span
          className="inline-block h-4 w-4 rounded-full"
          style={{ background: me.color }}
        />
        <span className="font-bold">{me.name}</span>
        <span className="ml-auto text-sm text-slate-400 tabular-nums">
          {me.totalScore} pnt
        </span>
      </header>

      {state.phase === "lobby" && (
        <Center>
          <div className="text-center space-y-3">
            <p className="text-6xl">🏎️</p>
            <p className="text-xl font-bold">Je doet mee!</p>
            <p className="text-slate-400">
              Kijk naar het grote scherm — de host start zo ronde 1.
            </p>
          </div>
        </Center>
      )}

      {state.phase === "round" && !record && (
        <section className="flex flex-1 flex-col gap-4">
          <div className="flex items-baseline justify-between">
            <h1 className="text-xl font-black">
              Ronde {state.round}/{state.totalRounds} — kies je clip ε
            </h1>
            <Countdown
              endsAt={state.roundEndsAt}
              offsetRef={offsetRef}
              className="text-2xl font-black"
            />
          </div>
          <p className="text-sm text-slate-400">
            Hoe groot mogen de policy-updates zijn? Tik om te trainen — je
            traint verder vanaf je vorige curve.
          </p>
          <div className="grid grid-cols-2 gap-3">
            {CLIP_OPTIONS.map((clip) => (
              <button
                key={clip}
                onClick={() => pick(clip)}
                disabled={busy}
                className="rounded-2xl bg-slate-800 py-5 text-2xl font-bold ring-amber-400 transition active:scale-95 hover:bg-slate-700 disabled:opacity-50"
              >
                {clip}
              </button>
            ))}
          </div>
          {submitError && <p className="text-red-400 text-sm">{submitError}</p>}
        </section>
      )}

      {(state.phase === "round" || state.phase === "results") && record && (
        <section className="flex flex-col gap-4">
          <h1 className="text-xl font-black">
            Jouw training — ronde {state.round}/{state.totalRounds}
          </h1>
          <div className="rounded-2xl bg-slate-800/70 p-2">
            <CurveChart
              series={[{ curve: me.fullCurve.length ? me.fullCurve : record.curve, color: me.color, width: 3 }]}
              height={240}
              animate
            />
          </div>
          <div className="rounded-xl bg-slate-800 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold">
                ε = {record.clip} · {VERDICT_LABELS[record.verdict]}
              </span>
              <span className="font-black text-amber-300 tabular-nums">
                +{record.points}
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              {VERDICT_HINTS[record.verdict]}
            </p>
          </div>
          <p className="text-center text-sm text-slate-500 animate-pulse">
            {state.phase === "round"
              ? "Wachten tot iedereen getraind heeft…"
              : state.round < state.totalRounds
                ? "Denk na: welke ε kies je volgende ronde?"
                : "Klaar voor de race! 🏁"}
          </p>
        </section>
      )}

      {state.phase === "results" && !record && (
        <Center>
          <div className="text-center space-y-2">
            <p className="text-4xl">😴</p>
            <p className="text-slate-300">
              Je hebt deze ronde niet getraind — je agent is niets opgeschoten.
            </p>
          </div>
        </Center>
      )}

      {state.phase === "race" && (
        <Center>
          <div className="text-center space-y-3">
            <p className="text-6xl">🏁</p>
            <p className="text-xl font-bold">Daar gaat je agent!</p>
            <p className="text-slate-400">Kijk naar het grote scherm…</p>
          </div>
        </Center>
      )}

      {state.phase === "podium" && (
        <PodiumCard state={state} playerId={playerId} />
      )}
    </main>
  );
}

function Center({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-1 items-center justify-center p-6">{children}</div>
  );
}

function PodiumCard({
  state,
  playerId,
}: {
  state: StateResponse;
  playerId: string | null;
}) {
  const me = state.me!;
  const place = state.players.findIndex((p) => p.id === playerId) + 1;
  const medal =
    place === 1 ? "🥇" : place === 2 ? "🥈" : place === 3 ? "🥉" : "🏎️";
  return (
    <Center>
      <div className="text-center space-y-4">
        <p className="text-7xl">{medal}</p>
        {me.racePosition && (
          <p className="text-lg text-slate-300">
            Je agent werd <b>#{me.racePosition}</b> in de race
            {me.racePoints ? ` (+${me.racePoints} punten)` : ""}.
          </p>
        )}
        <p className="text-2xl font-black">
          Eindstand: #{place} van {state.playerCount}
        </p>
        <p className="text-xl font-bold text-amber-300 tabular-nums">
          {me.totalScore} punten
        </p>
        <p className="text-sm text-slate-500">Bedankt voor het racen! 🏁</p>
      </div>
    </Center>
  );
}
