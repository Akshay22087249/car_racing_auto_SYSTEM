import { CURVE_POINTS, RETURN_MAX, RETURN_MIN, roundEnv } from "./constants";
import type { RoundEnv } from "./constants";
import { randn, rngFor } from "./rng";
import type { RoundRecord, Verdict } from "./types";

/**
 * Parametric model of how PPO training responds to BOTH the clip ε and the
 * learning rate, relative to a per-round optimum:
 *  - either knob too small → tiny, timid updates: stable but very slow (flat)
 *  - both near the optimum → healthy, steady learning curve
 *  - either knob too large → big destructive updates: noisy curve with a high
 *    chance of catastrophic collapse, and the danger compounds when both are hot
 * The optimum shifts each round (see ROUND_ENVS), so a setting that was perfect
 * earlier may need re-tuning.
 */

/** Healthy fraction of the remaining gap to RETURN_MAX closed in one round */
const SWEET_GROWTH = 0.5;

interface Dynamics {
  growth: number;
  instability: number;
}

/** Map (clip, lr) against the round optimum to a growth + instability pair. */
function dynamics(clip: number, lr: number, env: RoundEnv): Dynamics {
  const clipOct = Math.log2(clip / env.clipOpt);
  const lrOct = Math.log2(lr / env.lrOpt);
  // how far below the optimum (too timid) and above it (too aggressive),
  // summed over both knobs and scaled by the round's tolerance
  const timid =
    (Math.max(0, -clipOct) + Math.max(0, -lrOct)) / env.tolerance;
  const aggr = (Math.max(0, clipOct) + Math.max(0, lrOct)) / env.tolerance;

  // timidity throttles learning; mild aggressiveness gives an early burst
  const growth = SWEET_GROWTH * Math.exp(-0.7 * timid) * (1 + 0.2 * Math.min(aggr, 1.5));
  const instability = Math.min(1, 0.12 * aggr + 0.18 * aggr * aggr);
  return { growth, instability };
}

export function simulateRound(
  clip: number,
  lr: number,
  startP: number,
  seed: string,
  round: number
): Omit<RoundRecord, "points" | "bonus"> {
  const env = roundEnv(round);
  const rng = rngFor("curve", seed);
  const { growth, instability } = dynamics(clip, lr, env);
  const stepRate = 1 - Math.pow(1 - Math.min(growth, 0.95), 1 / CURVE_POINTS);
  const noiseSd = 6 + instability * 110;

  const collapses = rng() < Math.min(instability * 1.35, 0.97);
  const collapseAt = collapses
    ? 6 + Math.floor(rng() * (CURVE_POINTS - 10))
    : -1;

  const curve: number[] = [];
  let p = startP;
  let collapsed = false;
  for (let t = 0; t < CURVE_POINTS; t++) {
    if (t === collapseAt) {
      collapsed = true;
      p = RETURN_MIN + 20 + rng() * 40;
    }
    // a collapsed policy barely recovers within the same round
    const rate = collapsed ? stepRate * 0.04 : stepRate;
    const sd = collapsed ? noiseSd * 0.5 : noiseSd;
    p = p + (RETURN_MAX - p) * rate + randn(rng) * sd;
    p = Math.max(RETURN_MIN, Math.min(RETURN_MAX, p));
    curve.push(Math.round(p));
  }

  const tail = curve.slice(-4);
  const endP = Math.round(tail.reduce((a, b) => a + b, 0) / tail.length);

  let verdict: Verdict;
  if (collapsed) verdict = "collapsed";
  else if (instability > 0.15) verdict = "unstable";
  else if (growth < 0.28) verdict = "slow";
  else verdict = "good";

  return { clip, lr, curve, startP, endP, verdict };
}

export function roundPoints(record: { startP: number; endP: number }): number {
  return Math.max(0, Math.round(record.endP - record.startP));
}

/** Coaching line for the start of a round, based on the player's last result. */
export function roundCoach(
  prev: Pick<RoundRecord, "verdict"> | null,
  env: RoundEnv
): string {
  const shift =
    env.round === 3
      ? " Let op: je agent is volwassener, dus de sweetspot is kleiner geworden."
      : env.round === 4
        ? " Het is laat in de training: verlaag nu je learning rate (annealing)."
        : env.round === 5
          ? " Laatste ronde: fijn afstellen, de marges zijn klein."
          : "";
  if (!prev || prev.verdict === "none") {
    return `Ronde ${env.round}: ${env.intro}.${shift}`;
  }
  const base: Record<Exclude<Verdict, "none">, string> = {
    collapsed: "Vorige ronde stortte je in 💥. Kies flink kleiner, zowel clip als learning rate.",
    unstable: "Vorige ronde was grillig ⚡. Iets kleiner is veiliger.",
    slow: "Vorige ronde was traag 🐌. Durf groter.",
    good: "Vorige ronde ging top 🚀. Blijf wel opletten, de omgeving kan schuiven.",
  };
  return `${base[prev.verdict]}${shift}`;
}

export const VERDICT_HINTS: Record<Verdict, string> = {
  collapsed:
    "Je curve is ingestort! Je updates waren te groot: clip ε en/of learning rate te hoog, waardoor de policy zijn geleerde gedrag kapotmaakte. Kies kleiner.",
  unstable:
    "Grillige curve. Je updates staan hoog: grote stappen geven snelle winst, maar je flirt met een totale instorting. Iets kleiner is veiliger.",
  slow: "Vlakke, trage curve. Je updates zijn te klein: clip ε en/of learning rate laag, dus je agent leert bijna niets bij. Durf groter!",
  good: "Mooie, stabiele stijging. Clip ε en learning rate zitten in de sweet spot: groot genoeg om te leren, klein genoeg om stabiel te blijven.",
  none: "Je hebt deze ronde niet getraind, dus je agent is niets opgeschoten.",
};

export const VERDICT_LABELS: Record<Verdict, string> = {
  collapsed: "Ingestort 💥",
  unstable: "Grillig ⚡",
  slow: "Traag 🐌",
  good: "Stabiel 🚀",
  none: "Niet getraind 😴",
};
