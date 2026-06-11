export const TOTAL_ROUNDS = 5;
export const ROUND_SECONDS = 60;

/** Choices presented to the players each round */
export const CLIP_OPTIONS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8];

/** Learning-rate choices (shown from round 2 on); sweet spot ≈ 1e-4..3e-4 */
export const LR_OPTIONS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3];

/** Human-friendly label for a learning rate, e.g. 3e-4 → "3e-4" */
export function lrLabel(lr: number): string {
  const exp = Math.round(Math.log10(lr));
  const mant = Math.round(lr / Math.pow(10, exp));
  return `${mant}e${exp}`;
}

/**
 * Per-round environment. The optimum shifts and the tolerance tightens as the
 * agent matures, so a setting that was perfect in round 1 needs re-tuning
 * later (the core "there is no single magic number" lesson).
 */
export interface RoundEnv {
  round: number;
  theme: string;
  intro: string;
  /** one-line reasoning shown on the beamer, reinforces the lesson */
  why: string;
  clipOpt: number;
  lrOpt: number;
  /** octaves of slack around the optimum before growth/instability bite */
  tolerance: number;
  /** whether players tune the learning rate this round */
  showLr: boolean;
}

export const ROUND_ENVS: RoundEnv[] = [
  { round: 1, theme: "Vind de clip-sweetspot", intro: "kies je clip ε", why: "Een te kleine clip ε leert traag, een te grote stort in. Zoek de stabiele middenweg.", clipOpt: 0.2, lrOpt: 3e-4, tolerance: 1.6, showLr: false },
  { round: 2, theme: "Stem ook de learning rate", intro: "kies clip ε én learning rate", why: "De learning rate schaalt elke stap. Samen met de clip ε bepaalt hij hoe groot je updates zijn.", clipOpt: 0.2, lrOpt: 3e-4, tolerance: 1.4, showLr: true },
  { round: 3, theme: "De sweetspot verschuift", intro: "je agent is volwassener, dus stel opnieuw af", why: "Je agents zijn nu verder getraind. Een grote clip ε verandert het beleid te hard, dus de veilige sweetspot is kleiner geworden.", clipOpt: 0.1, lrOpt: 3e-4, tolerance: 1.3, showLr: true },
  { round: 4, theme: "Anneal je learning rate", intro: "het is laat in de training, verlaag je learning rate", why: "Laat in de training wil je kleine, precieze stappen. Verlaag je learning rate (annealing), anders schiet je over het doel heen.", clipOpt: 0.1, lrOpt: 1e-4, tolerance: 1.2, showLr: true },
  { round: 5, theme: "Fine-tunen", intro: "laatste ronde, de marges zijn klein", why: "Finetunen: de marges zijn klein, kleine bijstellingen werken nu het best.", clipOpt: 0.05, lrOpt: 1e-4, tolerance: 0.95, showLr: true },
];

export function roundEnv(round: number): RoundEnv {
  return ROUND_ENVS[Math.max(0, Math.min(ROUND_ENVS.length - 1, round - 1))];
}

/** Bonus points per extra consecutive "stabiel" round (streak). */
export const STREAK_BONUS = 60;

/** Points of the learning curve plotted per round */
export const CURVE_POINTS = 30;
/** Training steps represented by one round (for axis labels) */
export const STEPS_PER_ROUND = 500_000;

/** CarRacing-style return range */
export const RETURN_START = -30;
export const RETURN_MAX = 900;
export const RETURN_MIN = -100;

export const RACE_LAPS = 2;
/** Total race distance is laps * track length (track length ≈ 1000 units) */
export const RACE_SPEED_MIN = 26;
export const RACE_SPEED_MAX = 58;

export function racePointsForPosition(pos: number): number {
  return Math.max(50, Math.round(500 * Math.pow(0.85, pos - 1)));
}

export function playerColor(index: number): string {
  const hue = Math.round((index * 137.508) % 360);
  return `hsl(${hue} 85% 60%)`;
}

export const GAME_TTL_SECONDS = 6 * 60 * 60;
