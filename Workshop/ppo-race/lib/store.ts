import { Redis } from "@upstash/redis";
import { GAME_TTL_SECONDS } from "./constants";
import type { GameMeta, PlayerData, RaceCar } from "./types";

/**
 * Storage layout (Redis):
 *   ppo:{pin}:meta     → JSON GameMeta
 *   ppo:{pin}:players  → hash, field = playerId, value = JSON PlayerData
 *   ppo:{pin}:race     → JSON RaceCar[]
 * Player records are only ever written by that player's own requests (or a
 * single host action), so per-field hash writes avoid cross-player races.
 */

export interface GameStore {
  kind: "redis" | "memory";
  getMeta(pin: string): Promise<GameMeta | null>;
  setMeta(pin: string, meta: GameMeta): Promise<void>;
  getPlayers(pin: string): Promise<Record<string, PlayerData>>;
  getPlayer(pin: string, id: string): Promise<PlayerData | null>;
  setPlayer(pin: string, player: PlayerData): Promise<void>;
  getRace(pin: string): Promise<RaceCar[] | null>;
  setRace(pin: string, cars: RaceCar[]): Promise<void>;
}

function redisUrl(): { url: string; token: string } | null {
  const url =
    process.env.UPSTASH_REDIS_REST_URL ?? process.env.KV_REST_API_URL;
  const token =
    process.env.UPSTASH_REDIS_REST_TOKEN ?? process.env.KV_REST_API_TOKEN;
  if (url && token) return { url, token };
  return null;
}

function makeRedisStore(creds: { url: string; token: string }): GameStore {
  const redis = new Redis({ url: creds.url, token: creds.token });
  return {
    kind: "redis",
    async getMeta(pin) {
      return (await redis.get<GameMeta>(`ppo:${pin}:meta`)) ?? null;
    },
    async setMeta(pin, meta) {
      await redis.set(`ppo:${pin}:meta`, meta, { ex: GAME_TTL_SECONDS });
    },
    async getPlayers(pin) {
      const raw = await redis.hgetall<Record<string, PlayerData>>(
        `ppo:${pin}:players`
      );
      return raw ?? {};
    },
    async getPlayer(pin, id) {
      return (
        (await redis.hget<PlayerData>(`ppo:${pin}:players`, id)) ?? null
      );
    },
    async setPlayer(pin, player) {
      const key = `ppo:${pin}:players`;
      await redis.hset(key, { [player.id]: player });
      await redis.expire(key, GAME_TTL_SECONDS);
    },
    async getRace(pin) {
      return (await redis.get<RaceCar[]>(`ppo:${pin}:race`)) ?? null;
    },
    async setRace(pin, cars) {
      await redis.set(`ppo:${pin}:race`, cars, { ex: GAME_TTL_SECONDS });
    },
  };
}

interface MemGame {
  meta: GameMeta;
  players: Map<string, PlayerData>;
  race: RaceCar[] | null;
}

function makeMemoryStore(): GameStore {
  const g = globalThis as typeof globalThis & {
    __ppoGames?: Map<string, MemGame>;
  };
  g.__ppoGames ??= new Map();
  const games = g.__ppoGames;
  const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v));
  return {
    kind: "memory",
    async getMeta(pin) {
      return games.get(pin)?.meta ? clone(games.get(pin)!.meta) : null;
    },
    async setMeta(pin, meta) {
      const game = games.get(pin);
      if (game) game.meta = clone(meta);
      else games.set(pin, { meta: clone(meta), players: new Map(), race: null });
    },
    async getPlayers(pin) {
      const game = games.get(pin);
      if (!game) return {};
      return clone(Object.fromEntries(game.players));
    },
    async getPlayer(pin, id) {
      const p = games.get(pin)?.players.get(id);
      return p ? clone(p) : null;
    },
    async setPlayer(pin, player) {
      games.get(pin)?.players.set(player.id, clone(player));
    },
    async getRace(pin) {
      const race = games.get(pin)?.race;
      return race ? clone(race) : null;
    },
    async setRace(pin, cars) {
      const game = games.get(pin);
      if (game) game.race = clone(cars);
    },
  };
}

let store: GameStore | null = null;

export function getStore(): GameStore {
  if (!store) {
    const creds = redisUrl();
    if (creds) {
      store = makeRedisStore(creds);
    } else {
      if (process.env.VERCEL) {
        console.warn(
          "[ppo-race] Geen Upstash Redis gevonden (UPSTASH_REDIS_REST_URL/TOKEN). " +
            "Op Vercel werkt de in-memory store NIET betrouwbaar tussen requests!"
        );
      }
      store = makeMemoryStore();
    }
  }
  return store;
}
