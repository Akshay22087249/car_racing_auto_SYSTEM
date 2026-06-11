import { CURVE_POINTS, RETURN_MAX, RETURN_MIN } from "./constants";
import { randn, rngFor } from "./rng";
import type { RoundRecord, Verdict } from "./types";

/**
 * Parametric model of how PPO training responds to the clip ε:
 *  - too small  → tiny trust region, stable but very slow progress (flat curve)
 *  - sweet spot (~0.1–0.3) → healthy, steady learning curve
 *  - too large  → big destructive policy updates: noisy curve with a high
 *    chance of catastrophic collapse mid-training
 */

/** Fraction of the remaining gap to RETURN_MAX closed in one round */
function growthFor(clip: number): number {
  const eff = Math.min(clip, 0.3);
  return 0.5 * Math.pow(eff / 0.2, 0.6);
}

/** 0 = stable, 1 = maximally unstable */
function instabilityFor(clip: number): number {
  return Math.max(0, Math.min(1, (clip - 0.25) / 0.55));
}

export function simulateRound(
  clip: number,
  startP: number,
  seed: string
): Omit<RoundRecord, "points" | "clip"> & { clip: number } {
  const rng = rngFor("curve", seed);
  const g = growthFor(clip);
  const inst = instabilityFor(clip);
  // Large clips push faster at first — that is what makes them tempting.
  const burst = 1 + inst * 0.6;
  const stepRate = 1 - Math.pow(1 - Math.min(g * burst, 0.95), 1 / CURVE_POINTS);
  const noiseSd = 6 + inst * 110;

  const collapses = rng() < Math.min(inst * 1.35, 0.97);
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
  else if (inst > 0.05) verdict = "unstable";
  else if (g < 0.28) verdict = "slow";
  else verdict = "good";

  return { clip, curve, startP, endP, verdict };
}

export function roundPoints(record: { startP: number; endP: number }): number {
  return Math.max(0, Math.round(record.endP - record.startP));
}

export const VERDICT_HINTS: Record<Verdict, string> = {
  collapsed:
    "Je curve is ingestort! Je clip ε is te groot: de policy-updates waren zo groot dat je agent zijn geleerde gedrag kapotmaakte. Probeer een kleinere waarde.",
  unstable:
    "Grillige curve. Je clip ε staat hoog: grote updates geven snelle winst, maar je flirt met een totale instorting. Iets kleiner is veiliger.",
  slow: "Vlakke, trage curve. Je clip ε is klein: elke update is voorzichtig, dus je agent leert bijna niets bij. Durf groter!",
  good: "Mooie, stabiele stijging — deze clip ε zit in de sweet spot. Updates zijn groot genoeg om te leren, klein genoeg om stabiel te blijven.",
  none: "Je hebt deze ronde niet getraind — je agent is niets opgeschoten.",
};

export const VERDICT_LABELS: Record<Verdict, string> = {
  collapsed: "Ingestort 💥",
  unstable: "Grillig ⚡",
  slow: "Traag 🐌",
  good: "Stabiel 🚀",
  none: "Niet getraind 😴",
};
