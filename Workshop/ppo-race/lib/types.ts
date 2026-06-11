export type Phase = "lobby" | "round" | "results" | "race" | "podium";

export type Verdict = "good" | "slow" | "unstable" | "collapsed" | "none";

export interface RoundRecord {
  clip: number;
  points: number;
  /** Performance (return) samples for this training segment */
  curve: number[];
  startP: number;
  endP: number;
  verdict: Verdict;
}

export interface PlayerData {
  id: string;
  name: string;
  color: string;
  joinedAt: number;
  rounds: Record<number, RoundRecord>;
  racePoints?: number;
  racePosition?: number;
}

export interface GameMeta {
  pin: string;
  hostKey: string;
  phase: Phase;
  round: number;
  roundEndsAt: number;
  raceSeed: number;
  raceStartedAt: number;
  createdAt: number;
}

/** One car in the (deterministic) race, shared by server scoring and client animation */
export interface RaceCar {
  id: string;
  name: string;
  color: string;
  /** 0..1, derived from the final tuned agent */
  skill: number;
  /** track units per second */
  speed: number;
  /** spin events: at distance `at`, car spins for `dur` seconds */
  spins: { at: number; dur: number }[];
  finishTime: number;
}

export interface PlayerPublic {
  id: string;
  name: string;
  color: string;
  score: number;
  answered: boolean;
}

export interface CurvePublic {
  id: string;
  name: string;
  color: string;
  clip: number | null;
  points: number;
  verdict: Verdict;
  /** Full concatenated curve over all rounds so far */
  curve: number[];
}

export interface StateResponse {
  phase: Phase;
  round: number;
  totalRounds: number;
  roundEndsAt: number;
  serverNow: number;
  answeredCount: number;
  playerCount: number;
  players: PlayerPublic[];
  storage: "redis" | "memory";
  /** host only, during results/race/podium */
  curves?: CurvePublic[];
  /** during race/podium */
  race?: { seed: number; startedAt: number; cars: RaceCar[] };
  /** player only */
  me?: {
    name: string;
    color: string;
    totalScore: number;
    currentRound: RoundRecord | null;
    /** concatenated curve over all rounds so far */
    fullCurve: number[];
    racePosition?: number;
    racePoints?: number;
  };
}
