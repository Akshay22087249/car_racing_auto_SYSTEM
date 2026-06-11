/**
 * The eleven Formula 1 teams of the 2026 season (22 seats, 2 drivers each).
 * Colours are each team's signature livery: a primary plus an accent used for
 * the car's wing/stripe.
 */
export interface Team {
  id: string;
  name: string;
  /** 3-letter code shown in classifications */
  short: string;
  color: string;
  accent: string;
}

export const TEAMS: Team[] = [
  { id: "mclaren", name: "McLaren", short: "MCL", color: "#FF8000", accent: "#1E9BD7" },
  { id: "ferrari", name: "Ferrari", short: "FER", color: "#E8002D", accent: "#FFF200" },
  { id: "mercedes", name: "Mercedes", short: "MER", color: "#00D2BE", accent: "#C0C0C0" },
  { id: "redbull", name: "Red Bull Racing", short: "RBR", color: "#3671C6", accent: "#FF1E00" },
  { id: "aston", name: "Aston Martin", short: "AMR", color: "#229971", accent: "#CEDC00" },
  { id: "alpine", name: "Alpine", short: "ALP", color: "#FF87BC", accent: "#0093CC" },
  { id: "williams", name: "Williams", short: "WIL", color: "#37BEDD", accent: "#0B2E6F" },
  { id: "racingbulls", name: "Racing Bulls", short: "RB", color: "#6692FF", accent: "#E50000" },
  { id: "haas", name: "Haas", short: "HAA", color: "#C7C9CB", accent: "#E6002B" },
  { id: "audi", name: "Audi", short: "AUD", color: "#A21E3C", accent: "#D0D0D0" },
  { id: "cadillac", name: "Cadillac", short: "CAD", color: "#D4AF37", accent: "#0B1E3B" },
];

/** Drivers allowed per team, just like the real grid. */
export const TEAM_CAPACITY = 2;

export function teamById(id: string | undefined | null): Team | undefined {
  return TEAMS.find((t) => t.id === id);
}

/** Fallback accent for a player with no team (privateer). */
export const NEUTRAL_ACCENT = "#e2e8f0";
