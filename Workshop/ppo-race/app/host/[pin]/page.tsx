"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import Countdown from "@/components/Countdown";
import CurveChart from "@/components/CurveChart";
import RaceView from "@/components/RaceView";
import { CLIP_OPTIONS, LR_OPTIONS, lrLabel, roundEnv } from "@/lib/constants";
import { PPO_FACTS, RECAP_POINTS, RECAP_SOURCE } from "@/lib/education";
import { VERDICT_LABELS } from "@/lib/sim";
import { teamById, type Team } from "@/lib/teams";
import { useGame, postJson } from "@/lib/useGame";
import type { ChoiceDist, PlayerPublic, StateResponse } from "@/lib/types";

export default function HostScreen() {
  const { pin } = useParams<{ pin: string }>();
  const [hostKey, setHostKey] = useState<string | null>(null);
  const [keyMissing, setKeyMissing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [raceDone, setRaceDone] = useState(false);
  const songRef = useRef<HTMLAudioElement>(null);
  const fadeRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // ?key=… makes the host screen portable (e.g. open it on the beamer pc)
    const urlKey = new URLSearchParams(window.location.search).get("key");
    if (urlKey) localStorage.setItem(`ppo:host:${pin}`, urlKey);
    const key = urlKey ?? localStorage.getItem(`ppo:host:${pin}`);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- localStorage is client-only
    if (key) setHostKey(key);
    else setKeyMissing(true);
  }, [pin]);

  const { state, error, offsetRef } = useGame(pin ?? "", {
    hostKey: hostKey ?? undefined,
  });

  const act = async (action: string) => {
    if (!hostKey || busy) return;
    setBusy(true);
    await postJson(`/api/game/${pin}/host`, { hostKey, action });
    setBusy(false);
  };

  // start the race anthem on the host's click; a user gesture lets it autoplay
  const startRace = async () => {
    const a = songRef.current;
    if (a) {
      a.currentTime = 0;
      a.volume = 0;
      a.play().catch(() => {});
      // fade the volume in over the 3s race countdown
      const FADE_MS = 3000;
      const TARGET_VOL = 0.7;
      const start = Date.now();
      if (fadeRef.current) clearInterval(fadeRef.current);
      fadeRef.current = setInterval(() => {
        const p = Math.min((Date.now() - start) / FADE_MS, 1);
        a.volume = TARGET_VOL * p;
        if (p >= 1 && fadeRef.current) {
          clearInterval(fadeRef.current);
          fadeRef.current = null;
        }
      }, 50);
    }
    await act("startRace");
  };

  const reset = () => {
    if (
      window.confirm(
        "Game resetten naar de lobby? Alle scores en curves gaan verloren."
      )
    ) {
      act("reset");
    }
  };

  // stop the anthem (and any running fade) as soon as we leave the race
  useEffect(() => {
    if (state?.phase !== "race") {
      if (fadeRef.current) {
        clearInterval(fadeRef.current);
        fadeRef.current = null;
      }
      const a = songRef.current;
      if (a && !a.paused) {
        a.pause();
        a.currentTime = 0;
      }
    }
  }, [state?.phase]);

  // clear any running fade on unmount
  useEffect(
    () => () => {
      if (fadeRef.current) clearInterval(fadeRef.current);
    },
    []
  );

  const memoryWarning =
    state?.storage === "memory" &&
    typeof window !== "undefined" &&
    !["localhost", "127.0.0.1"].includes(window.location.hostname);

  if (keyMissing) {
    return (
      <Center>
        <p className="text-xl text-red-300">
          Geen host-sleutel gevonden voor game {pin}. Maak een nieuwe game aan
          via <Link className="underline" href="/host">/host</Link>.
        </p>
      </Center>
    );
  }
  if (error) {
    return (
      <Center>
        <p className="text-xl text-red-300">{error}</p>
      </Center>
    );
  }
  if (!state) {
    return (
      <Center>
        <p className="text-xl text-slate-400 animate-pulse">Laden…</p>
      </Center>
    );
  }

  return (
    <main className="relative flex flex-1 flex-col p-8 lg:p-12 gap-8">
      <audio ref={songRef} src="/race-song.mp3" preload="auto" loop />
      {state.phase !== "lobby" && (
        <button
          onClick={reset}
          className="absolute right-4 top-4 z-30 rounded-lg bg-slate-800/80 px-3 py-1.5 text-xs font-semibold text-slate-400 hover:bg-slate-700 hover:text-slate-200"
        >
          ↺ Reset
        </button>
      )}
      {memoryWarning && (
        <div className="rounded-xl bg-red-900/60 border border-red-500 px-4 py-2 text-sm">
          ⚠️ Geen Redis gekoppeld (UPSTASH_REDIS_REST_URL/TOKEN ontbreekt). Op
          Vercel raakt de gamestate zo tussen requests kwijt. Zie de README.
        </div>
      )}
      {state.phase === "lobby" && (
        <Lobby pin={pin} state={state} busy={busy} onStart={() => act("startRound")} />
      )}
      {state.phase === "round" && (
        <Round state={state} offsetRef={offsetRef} busy={busy} onEnd={() => act("endRound")} />
      )}
      {state.phase === "results" && (
        <Results state={state} busy={busy} onNext={() => act("startRound")} onRace={startRace} />
      )}
      {state.phase === "race" && state.race && (
        <section className="flex flex-1 flex-col gap-6">
          <header className="flex items-center justify-between">
            <h1 className="text-4xl font-black">🏁 De race van de getunede agents!</h1>
            {raceDone && (
              <button
                onClick={() => act("podium")}
                disabled={busy}
                className="rounded-xl bg-amber-400 px-6 py-3 text-xl font-bold text-slate-900 hover:bg-amber-300 animate-bounce"
              >
                Naar het podium 🏆
              </button>
            )}
          </header>
          <RaceView
            cars={state.race.cars}
            seed={state.race.seed}
            startedAt={state.race.startedAt}
            offsetRef={offsetRef}
            onAllFinished={() => setRaceDone(true)}
          />
        </section>
      )}
      {state.phase === "podium" && <Podium state={state} />}
    </main>
  );
}

function Center({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex flex-1 items-center justify-center p-8">{children}</main>
  );
}

function PlayerChips({
  state,
  showAnswered,
}: {
  state: StateResponse;
  showAnswered: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-3 justify-center">
      {state.players.map((p) => {
        const dim = showAnswered && !p.answered;
        return (
          <span
            key={p.id}
            className={`rounded-full px-4 py-1.5 text-lg font-semibold transition ${
              dim ? "opacity-30" : ""
            }`}
            style={{ background: `${p.color}`, color: "#0f172a" }}
          >
            {p.name}
            {showAnswered && p.answered ? " ✓" : ""}
          </span>
        );
      })}
    </div>
  );
}

function Lobby({
  pin,
  state,
  busy,
  onStart,
}: {
  pin: string;
  state: StateResponse;
  busy: boolean;
  onStart: () => void;
}) {
  const [origin, setOrigin] = useState("");
  const [fullOrigin, setFullOrigin] = useState("");
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- window is client-only
    setOrigin(window.location.origin.replace(/^https?:\/\//, ""));
    setFullOrigin(window.location.origin);
  }, []);
  return (
    <section className="flex flex-1 flex-col items-center justify-center gap-10 text-center">
      <h1 className="text-5xl font-black tracking-tight">
        🏎️ PPO <span className="text-amber-400">Race</span>
      </h1>
      <div className="flex flex-col items-center gap-6 lg:flex-row lg:gap-12">
        <div className="flex flex-col items-center gap-2">
          <p className="text-2xl text-slate-300">
            Ga naar <span className="font-bold text-white">{origin}</span> en vul
            in:
          </p>
          <p className="text-9xl font-black tracking-widest text-amber-400 tabular-nums">
            {pin}
          </p>
        </div>
        {fullOrigin && (
          <div className="flex flex-col items-center gap-2">
            <div className="rounded-2xl bg-white p-3">
              <QRCodeSVG value={`${fullOrigin}/?pin=${pin}`} size={200} />
            </div>
            <p className="text-lg text-slate-300">
              of scan om direct mee te doen
            </p>
          </div>
        )}
      </div>
      <div className="min-h-24 w-full max-w-4xl">
        {state.playerCount === 0 ? (
          <p className="text-xl text-slate-400 animate-pulse">
            Wachten op spelers…
          </p>
        ) : (
          <PlayerChips state={state} showAnswered={false} />
        )}
      </div>
      <p className="text-xl text-slate-300">
        {state.playerCount} speler{state.playerCount === 1 ? "" : "s"}
      </p>
      <button
        onClick={onStart}
        disabled={busy || state.playerCount === 0}
        className="rounded-2xl bg-amber-400 px-10 py-5 text-2xl font-bold text-slate-900 shadow-lg hover:bg-amber-300 disabled:opacity-40"
      >
        Start ronde 1 ▶
      </button>
    </section>
  );
}

function Round({
  state,
  offsetRef,
  busy,
  onEnd,
}: {
  state: StateResponse;
  offsetRef: React.RefObject<number>;
  busy: boolean;
  onEnd: () => void;
}) {
  const env = roundEnv(state.round);
  return (
    <section className="flex flex-1 flex-col items-center gap-6 text-center">
      <div className="space-y-1">
        <h1 className="text-4xl font-black">
          Ronde {state.round}/{state.totalRounds}:{" "}
          <span className="text-amber-400">{env.theme}</span>
        </h1>
        <p className="text-xl text-slate-300 max-w-3xl">
          Stem je PPO-agent af op je telefoon. Te groot = instortingsgevaar 💥 ·
          te klein = bijna geen vooruitgang 🐌
          {env.showLr ? " · nu ook de learning rate!" : ""}
        </p>
      </div>
      <div className="max-w-3xl rounded-xl border border-indigo-500/40 bg-indigo-950/70 px-5 py-3 text-lg text-indigo-100">
        💡 {env.why}
      </div>
      <Countdown
        endsAt={state.roundEndsAt}
        offsetRef={offsetRef}
        className="text-8xl font-black"
      />
      <p className="text-2xl font-bold text-slate-200 tabular-nums">
        {state.answeredCount}/{state.playerCount} hebben getraind
      </p>
      <PlayerChips state={state} showAnswered={true} />

      <div className="grid w-full max-w-5xl gap-6 lg:grid-cols-2">
        <ChoiceHistogram dist={state.choiceDist} showLr={env.showLr} />
        <RotatingFact />
      </div>

      <button
        onClick={onEnd}
        disabled={busy}
        className="rounded-xl bg-slate-700 px-6 py-3 text-lg font-semibold hover:bg-slate-600"
      >
        Ronde afsluiten
      </button>
    </section>
  );
}

function ChoiceHistogram({
  dist,
  showLr,
}: {
  dist?: ChoiceDist;
  showLr: boolean;
}) {
  const clipCounts = CLIP_OPTIONS.map((c) => dist?.clip?.[String(c)] ?? 0);
  const lrCounts = LR_OPTIONS.map((l) => dist?.lr?.[String(l)] ?? 0);
  const maxClip = Math.max(1, ...clipCounts);
  const maxLr = Math.max(1, ...lrCounts);
  return (
    <div className="rounded-2xl bg-slate-800/60 p-4 text-left">
      <p className="mb-3 text-sm font-bold text-slate-300">Wat kiest de klas?</p>
      <Bars
        title="clip ε"
        labels={CLIP_OPTIONS.map(String)}
        counts={clipCounts}
        max={maxClip}
      />
      {showLr && (
        <div className="mt-4">
          <Bars
            title="learning rate"
            labels={LR_OPTIONS.map(lrLabel)}
            counts={lrCounts}
            max={maxLr}
          />
        </div>
      )}
    </div>
  );
}

function Bars({
  title,
  labels,
  counts,
  max,
}: {
  title: string;
  labels: string[];
  counts: number[];
  max: number;
}) {
  return (
    <div>
      <p className="mb-1 text-xs uppercase tracking-wide text-slate-500">
        {title}
      </p>
      <div className="space-y-1">
        {labels.map((lab, i) => (
          <div key={lab} className="flex items-center gap-2">
            <span className="w-12 shrink-0 text-right text-xs tabular-nums text-slate-400">
              {lab}
            </span>
            <div className="h-4 flex-1 rounded bg-slate-700/40">
              <div
                className="h-4 rounded bg-amber-400 transition-all"
                style={{ width: `${(counts[i] / max) * 100}%` }}
              />
            </div>
            <span className="w-6 text-xs tabular-nums text-slate-400">
              {counts[i] || ""}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RotatingFact() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(
      () => setI((n) => (n + 1) % PPO_FACTS.length),
      7000
    );
    return () => clearInterval(id);
  }, []);
  return (
    <div className="flex items-center rounded-2xl border border-indigo-500/40 bg-indigo-950/70 p-5 text-left">
      <p className="text-lg leading-relaxed text-indigo-100">
        <span className="mr-2">💡</span>
        {PPO_FACTS[i]}
      </p>
    </div>
  );
}

function Results({
  state,
  busy,
  onNext,
  onRace,
}: {
  state: StateResponse;
  busy: boolean;
  onNext: () => void;
  onRace: () => void;
}) {
  const curves = useMemo(() => state.curves ?? [], [state.curves]);
  const ranked = useMemo(
    () => [...curves].sort((a, b) => b.points - a.points),
    [curves]
  );
  // element-wise class average (over players who reached each point)
  const avgCurve = useMemo(() => {
    if (curves.length < 2) return null;
    const maxLen = Math.max(0, ...curves.map((c) => c.curve.length));
    const out: number[] = [];
    for (let i = 0; i < maxLen; i++) {
      let sum = 0;
      let n = 0;
      for (const c of curves) {
        if (i < c.curve.length) {
          sum += c.curve[i];
          n++;
        }
      }
      if (n) out.push(Math.round(sum / n));
    }
    return out;
  }, [curves]);
  const series = useMemo(() => {
    const base = curves.map((c) => ({ curve: c.curve, color: c.color }));
    return avgCurve
      ? [...base, { curve: avgCurve, color: "rgba(255,255,255,0.92)", width: 4 }]
      : base;
  }, [curves, avgCurve]);
  const env = roundEnv(state.round);
  const lastRound = state.round >= state.totalRounds;
  return (
    <section className="flex flex-1 flex-col gap-6">
      <header className="flex items-center justify-between gap-4">
        <h1 className="text-4xl font-black">
          Learning curves na ronde {state.round}/{state.totalRounds}
        </h1>
        <button
          onClick={lastRound ? onRace : onNext}
          disabled={busy}
          className="rounded-xl bg-amber-400 px-6 py-3 text-xl font-bold text-slate-900 hover:bg-amber-300 shrink-0"
        >
          {lastRound ? "🏁 Start de race!" : `Start ronde ${state.round + 1} ▶`}
        </button>
      </header>
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 flex-1 min-h-0">
        <div className="xl:col-span-2 rounded-2xl bg-slate-800/60 p-4">
          <CurveChart series={series} height={420} annotate />
          {avgCurve && (
            <p className="mt-1 text-center text-xs text-slate-400">
              De dikke witte lijn is het klasgemiddelde.
            </p>
          )}
        </div>
        <div className="flex flex-col gap-3 min-h-0">
          <ol className="space-y-1.5 overflow-y-auto pr-1">
            {ranked.map((c, i) => (
              <li
                key={c.id}
                className="flex items-center gap-2 rounded-lg bg-slate-800/80 px-3 py-1.5"
              >
                <span className="w-6 font-bold text-slate-400 tabular-nums">{i + 1}.</span>
                <span
                  className="inline-block h-3 w-3 rounded-full shrink-0"
                  style={{ background: c.color }}
                />
                <span className="truncate font-medium">{c.name}</span>
                <span className="ml-1 text-xs text-slate-400">
                  {c.clip !== null ? `ε=${c.clip}` : "geen"}
                  {c.lr !== null && roundEnv(state.round).showLr
                    ? ` · lr ${lrLabel(c.lr)}`
                    : ""}{" "}
                  · {VERDICT_LABELS[c.verdict]}
                </span>
                <span className="ml-auto font-bold tabular-nums text-amber-300">
                  +{c.points}
                </span>
              </li>
            ))}
          </ol>
          <div className="rounded-xl bg-indigo-950/70 border border-indigo-500/40 p-4 text-sm leading-relaxed text-indigo-100">
            <p className="font-bold mb-1">📚 Wat zie je?</p>
            <p className="mb-2">
              <b>Sweetspot deze ronde:</b> clip ε rond {env.clipOpt}
              {env.showLr ? `, learning rate rond ${lrLabel(env.lrOpt)}` : ""}.{" "}
              {env.why}
            </p>
            <p>
              Een <b>vlakke</b> curve betekent te kleine updates: clip ε of
              learning rate te laag, dus de agent leert nauwelijks bij. Een{" "}
              <b>grillige of ingestorte</b> curve betekent te grote updates: clip
              ε of learning rate te hoog, waardoor geleerd gedrag sneuvelt.
              {!lastRound &&
                " Stel bij in de volgende ronde, je traint verder vanaf waar je nu bent."}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Podium({ state }: { state: StateResponse }) {
  const [showRecap, setShowRecap] = useState(false);
  const top3 = state.players.slice(0, 3);
  const medals = ["🥇", "🥈", "🥉"];
  const heights = ["h-44", "h-32", "h-24"];
  const order = [1, 0, 2]; // silver, gold, bronze layout

  // constructors' championship: teams ranked by their drivers' combined points
  const constructors = useMemo(() => {
    const m: Record<
      string,
      { team: Team; points: number; drivers: PlayerPublic[] }
    > = {};
    for (const p of state.players) {
      const t = teamById(p.team);
      if (!t) continue;
      (m[t.id] ??= { team: t, points: 0, drivers: [] });
      m[t.id].points += p.score;
      m[t.id].drivers.push(p);
    }
    return Object.values(m).sort((a, b) => b.points - a.points);
  }, [state.players]);

  return (
    <section className="relative flex flex-1 flex-col items-center gap-8 overflow-hidden">
      {!showRecap && <Confetti />}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-4">
        <h1 className="text-5xl font-black">
          {showRecap ? "📚 Wat hebben we geleerd?" : "🏆 Eindstand"}
        </h1>
        <button
          onClick={() => setShowRecap((v) => !v)}
          className="rounded-xl bg-indigo-500 px-5 py-2.5 text-lg font-bold hover:bg-indigo-400"
        >
          {showRecap ? "← Naar de eindstand" : "📚 Wat hebben we geleerd?"}
        </button>
      </div>

      {showRecap ? (
        <Recap />
      ) : (
        <>
          <div className="flex items-end gap-6">
            {order.map((idx) =>
              top3[idx] ? (
                <div
                  key={top3[idx].id}
                  className="flex flex-col items-center gap-2"
                >
                  <span className="text-6xl">{medals[idx]}</span>
                  <span className="text-2xl font-bold">{top3[idx].name}</span>
                  <span className="text-sm text-slate-400">
                    {teamById(top3[idx].team)?.name ?? ""}
                  </span>
                  <span className="text-xl text-amber-300 font-bold tabular-nums">
                    {top3[idx].score} punten
                  </span>
                  <div
                    className={`w-40 ${heights[idx]} rounded-t-xl`}
                    style={{ background: top3[idx].color }}
                  />
                </div>
              ) : null
            )}
          </div>

          <div className="grid w-full max-w-5xl gap-6 lg:grid-cols-2">
            <div className="rounded-2xl bg-slate-800/40 p-4">
              <h2 className="mb-2 text-2xl font-black">🏎️ Coureurs</h2>
              <ol className="space-y-1.5">
                {state.players.map((p, i) => {
                  const t = teamById(p.team);
                  return (
                    <li
                      key={p.id}
                      className="flex items-center gap-3 rounded-lg bg-slate-800/80 px-4 py-2"
                    >
                      <span className="w-7 font-bold text-slate-400 tabular-nums">
                        {i + 1}.
                      </span>
                      <span
                        className="inline-block h-3 w-3 rounded-full shrink-0"
                        style={{ background: p.color }}
                      />
                      <span className="font-medium">{p.name}</span>
                      {t && (
                        <span className="text-xs font-semibold text-slate-500">
                          {t.short}
                        </span>
                      )}
                      <span className="ml-auto font-bold tabular-nums text-amber-300">
                        {p.score}
                      </span>
                    </li>
                  );
                })}
              </ol>
            </div>

            <div className="rounded-2xl bg-slate-800/40 p-4">
              <h2 className="mb-2 text-2xl font-black">🏆 Constructeurs</h2>
              <ol className="space-y-1.5">
                {constructors.map((c, i) => (
                  <li
                    key={c.team.id}
                    className="flex items-center gap-3 rounded-lg bg-slate-800/80 px-4 py-2"
                  >
                    <span className="w-7 font-bold text-slate-400 tabular-nums">
                      {i + 1}.
                    </span>
                    <span
                      className="inline-block h-3 w-3 rounded-full shrink-0"
                      style={{ background: c.team.color }}
                    />
                    <span className="font-medium">{c.team.name}</span>
                    <span className="truncate text-xs text-slate-500">
                      {c.drivers.map((d) => d.name).join(", ")}
                    </span>
                    <span className="ml-auto font-bold tabular-nums text-amber-300">
                      {c.points}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

const CONFETTI_COLORS = [
  "#fbbf24",
  "#22c55e",
  "#6366f1",
  "#ef4444",
  "#ec4899",
  "#06b6d4",
];

// deterministic pseudo-random so confetti positions stay put across re-renders
const cRand = (i: number, salt: number) => {
  const x = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453;
  return x - Math.floor(x);
};

function Confetti() {
  const pieces = useMemo(
    () =>
      Array.from({ length: 60 }, (_, i) => ({
        left: cRand(i, 1) * 100,
        delay: cRand(i, 2) * 3,
        dur: 3 + cRand(i, 3) * 2.5,
        color: CONFETTI_COLORS[i % CONFETTI_COLORS.length],
        rot: cRand(i, 4) * 360,
      })),
    []
  );
  return (
    <div className="pointer-events-none absolute inset-0 z-20 overflow-hidden">
      {pieces.map((p, i) => (
        <span
          key={i}
          className="confetti-piece"
          style={{
            left: `${p.left}%`,
            background: p.color,
            animationDuration: `${p.dur}s`,
            animationDelay: `${p.delay}s`,
            transform: `rotate(${p.rot}deg)`,
          }}
        />
      ))}
    </div>
  );
}

function Recap() {
  return (
    <div className="w-full max-w-4xl space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        {RECAP_POINTS.map((pt) => (
          <div
            key={pt.title}
            className="rounded-2xl border border-indigo-500/40 bg-indigo-950/60 p-5 text-left"
          >
            <p className="mb-1 text-xl font-bold text-amber-300">{pt.title}</p>
            <p className="leading-relaxed text-indigo-100">{pt.body}</p>
          </div>
        ))}
      </div>
      <p className="text-center text-sm text-slate-500">{RECAP_SOURCE}</p>
    </div>
  );
}
