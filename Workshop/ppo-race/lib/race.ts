import {
  RACE_LAPS,
  RACE_SPEED_MAX,
  RACE_SPEED_MIN,
  RETURN_MAX,
  RETURN_MIN,
} from "./constants";
import { rngFor } from "./rng";
import { NEUTRAL_ACCENT, teamById } from "./teams";
import type { PlayerData, RaceCar } from "./types";

/**
 * Deterministic race: the server derives finish times for scoring and the
 * host screen replays the identical race from the same seed.
 */

export interface TrackPoint {
  x: number;
  y: number;
  /** cumulative arc length up to this point */
  s: number;
}

export interface Track {
  points: TrackPoint[];
  length: number;
}

/** Closed loop built from a jittered circle, smoothed with Catmull-Rom */
export function buildTrack(seed: number): Track {
  const rng = rngFor("track", seed);
  const N = 12;
  const ctrl: { x: number; y: number }[] = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const r = 120 + rng() * 60;
    ctrl.push({ x: Math.cos(a) * r * 1.35, y: Math.sin(a) * r });
  }
  const samplesPerSeg = 50;
  const pts: TrackPoint[] = [];
  let s = 0;
  for (let i = 0; i < N; i++) {
    const p0 = ctrl[(i - 1 + N) % N];
    const p1 = ctrl[i];
    const p2 = ctrl[(i + 1) % N];
    const p3 = ctrl[(i + 2) % N];
    for (let j = 0; j < samplesPerSeg; j++) {
      const t = j / samplesPerSeg;
      const t2 = t * t;
      const t3 = t2 * t;
      const x =
        0.5 *
        (2 * p1.x +
          (-p0.x + p2.x) * t +
          (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
          (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3);
      const y =
        0.5 *
        (2 * p1.y +
          (-p0.y + p2.y) * t +
          (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
          (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3);
      if (pts.length > 0) {
        const prev = pts[pts.length - 1];
        s += Math.hypot(x - prev.x, y - prev.y);
      }
      pts.push({ x, y, s });
    }
  }
  // close the loop
  const first = pts[0];
  const last = pts[pts.length - 1];
  s += Math.hypot(first.x - last.x, first.y - last.y);
  return { points: pts, length: s };
}

/** Position + heading at arc length `dist` (wraps around laps) */
export function pointAt(track: Track, dist: number) {
  const d = ((dist % track.length) + track.length) % track.length;
  const pts = track.points;
  // binary search on cumulative length
  let lo = 0;
  let hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (pts[mid].s <= d) lo = mid;
    else hi = mid - 1;
  }
  const a = pts[lo];
  const b = pts[(lo + 1) % pts.length];
  const span = Math.max(b.s > a.s ? b.s - a.s : track.length - a.s, 1e-6);
  const t = (d - a.s) / span;
  const x = a.x + (b.x - a.x) * t;
  const y = a.y + (b.y - a.y) * t;
  const heading = Math.atan2(b.y - a.y, b.x - a.x);
  return { x, y, heading };
}

/** Final agent quality 0..1 from the last round's end performance */
export function skillOf(player: PlayerData): number {
  const rounds = Object.keys(player.rounds)
    .map(Number)
    .sort((a, b) => a - b);
  const last = rounds.length ? player.rounds[rounds[rounds.length - 1]] : null;
  const endP = last ? last.endP : RETURN_MIN + 30;
  const skill = (endP - RETURN_MIN) / (RETURN_MAX - RETURN_MIN);
  return Math.max(0.05, Math.min(1, skill));
}

export function buildRaceCars(players: PlayerData[], seed: number): RaceCar[] {
  const track = buildTrack(seed);
  const total = track.length * RACE_LAPS;
  return players.map((p) => {
    const rng = rngFor("car", seed, p.id);
    const skill = skillOf(p);
    const roundNums = Object.keys(p.rounds)
      .map(Number)
      .sort((a, b) => a - b);
    const lastVerdict = roundNums.length
      ? p.rounds[roundNums[roundNums.length - 1]].verdict
      : "none";
    const speed =
      RACE_SPEED_MIN +
      skill * (RACE_SPEED_MAX - RACE_SPEED_MIN) +
      (rng() - 0.5) * 2;
    const expectedSpins = (1 - skill) * 5;
    const nSpins = Math.max(0, Math.round(expectedSpins + (rng() - 0.5) * 2));
    const spins = Array.from({ length: nSpins }, () => ({
      at: total * (0.05 + rng() * 0.9),
      dur: 1 + rng() * 1.6,
    })).sort((a, b) => a.at - b.at);
    const finishTime =
      total / speed + spins.reduce((acc, sp) => acc + sp.dur, 0);
    return {
      id: p.id,
      name: p.name,
      color: p.color,
      skill,
      accent: teamById(p.team)?.accent ?? NEUTRAL_ACCENT,
      driverNumber: p.driverNumber,
      lastVerdict,
      speed,
      spins,
      finishTime,
    };
  });
}

/** Car state at time t (seconds since race start) */
export function carAt(car: RaceCar, totalDist: number, t: number) {
  let dist = 0;
  let remaining = t;
  for (const sp of car.spins) {
    const driveTime = (sp.at - dist) / car.speed;
    if (remaining <= driveTime) {
      dist += remaining * car.speed;
      return {
        dist: Math.min(dist, totalDist),
        spinning: false,
        spinProgress: 0,
        finished: false,
      };
    }
    remaining -= driveTime;
    dist = sp.at;
    if (remaining <= sp.dur) {
      return { dist, spinning: true, spinProgress: remaining / sp.dur, finished: false };
    }
    remaining -= sp.dur;
  }
  dist += remaining * car.speed;
  if (dist >= totalDist) {
    return { dist: totalDist, spinning: false, spinProgress: 0, finished: true };
  }
  return { dist, spinning: false, spinProgress: 0, finished: false };
}
