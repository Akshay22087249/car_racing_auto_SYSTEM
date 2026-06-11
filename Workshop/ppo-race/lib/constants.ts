export const TOTAL_ROUNDS = 3;
export const ROUND_SECONDS = 60;

/** Choices presented to the players each round */
export const CLIP_OPTIONS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8];

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
