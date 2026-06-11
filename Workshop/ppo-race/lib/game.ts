import { TOTAL_ROUNDS } from "./constants";
import type { GameStore } from "./store";
import type {
  CurvePublic,
  GameMeta,
  PlayerData,
  PlayerPublic,
  RaceCar,
  StateResponse,
} from "./types";

export function randomId(): string {
  return Math.random().toString(36).slice(2, 12);
}

export async function freshPin(store: GameStore): Promise<string> {
  for (let i = 0; i < 20; i++) {
    const pin = String(Math.floor(1000 + Math.random() * 9000));
    if (!(await store.getMeta(pin))) return pin;
  }
  throw new Error("Geen vrije PIN gevonden");
}

export function totalScore(p: PlayerData): number {
  const roundSum = Object.values(p.rounds).reduce(
    (acc, r) => acc + r.points,
    0
  );
  return roundSum + (p.racePoints ?? 0);
}

export function concatCurve(p: PlayerData, upToRound: number): number[] {
  const out: number[] = [];
  for (let r = 1; r <= upToRound; r++) {
    const rec = p.rounds[r];
    if (rec) out.push(...rec.curve);
  }
  return out;
}

/** Round phase ends when the timer expires or everyone has answered. */
export async function effectiveMeta(
  store: GameStore,
  meta: GameMeta,
  players: PlayerData[]
): Promise<GameMeta> {
  if (meta.phase !== "round") return meta;
  const allAnswered =
    players.length > 0 && players.every((p) => p.rounds[meta.round]);
  if (Date.now() >= meta.roundEndsAt || allAnswered) {
    const next: GameMeta = { ...meta, phase: "results" };
    await store.setMeta(meta.pin, next);
    return next;
  }
  return meta;
}

export function buildState(
  meta: GameMeta,
  players: PlayerData[],
  storage: "redis" | "memory",
  opts: { isHost: boolean; playerId?: string; race?: RaceCar[] | null }
): StateResponse {
  const sorted = [...players].sort((a, b) => totalScore(b) - totalScore(a));
  const pub: PlayerPublic[] = sorted.map((p) => ({
    id: p.id,
    name: p.name,
    color: p.color,
    team: p.team,
    driverNumber: p.driverNumber,
    score: totalScore(p),
    answered: Boolean(p.rounds[meta.round]),
  }));

  const state: StateResponse = {
    phase: meta.phase,
    round: meta.round,
    totalRounds: TOTAL_ROUNDS,
    roundEndsAt: meta.roundEndsAt,
    serverNow: Date.now(),
    answeredCount: pub.filter((p) => p.answered).length,
    playerCount: pub.length,
    players: pub,
    storage,
  };

  const showCurves =
    meta.phase === "results" || meta.phase === "race" || meta.phase === "podium";
  if (opts.isHost && showCurves) {
    const curves: CurvePublic[] = sorted.map((p) => {
      const rec = p.rounds[meta.round] ?? null;
      return {
        id: p.id,
        name: p.name,
        color: p.color,
        team: p.team,
        clip: rec?.clip ?? null,
        lr: rec?.lr ?? null,
        points: rec?.points ?? 0,
        verdict: rec?.verdict ?? "none",
        curve: concatCurve(p, meta.round),
      };
    });
    state.curves = curves;
  }

  if (opts.isHost && meta.phase === "round") {
    const clip: Record<string, number> = {};
    const lr: Record<string, number> = {};
    for (const p of players) {
      const rec = p.rounds[meta.round];
      if (!rec) continue;
      clip[rec.clip] = (clip[rec.clip] ?? 0) + 1;
      lr[rec.lr] = (lr[rec.lr] ?? 0) + 1;
    }
    state.choiceDist = { clip, lr };
  }

  if (
    opts.isHost &&
    (meta.phase === "race" || meta.phase === "podium") &&
    opts.race
  ) {
    state.race = {
      seed: meta.raceSeed,
      startedAt: meta.raceStartedAt,
      cars: opts.race,
    };
  }

  if (opts.playerId) {
    const me = players.find((p) => p.id === opts.playerId);
    if (me) {
      state.me = {
        name: me.name,
        color: me.color,
        team: me.team,
        totalScore: totalScore(me),
        currentRound: me.rounds[meta.round] ?? null,
        prevRound: me.rounds[meta.round - 1] ?? null,
        fullCurve: concatCurve(me, meta.round),
        racePosition: me.racePosition,
        racePoints: me.racePoints,
      };
    }
  }

  return state;
}
