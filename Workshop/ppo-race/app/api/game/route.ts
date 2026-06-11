import { NextResponse } from "next/server";
import { freshPin, randomId } from "@/lib/game";
import { getStore } from "@/lib/store";
import { hashSeed } from "@/lib/rng";
import type { GameMeta } from "@/lib/types";

export async function POST() {
  const store = getStore();
  const pin = await freshPin(store);
  const hostKey = randomId() + randomId();
  const meta: GameMeta = {
    pin,
    hostKey,
    phase: "lobby",
    round: 0,
    roundEndsAt: 0,
    raceSeed: hashSeed(pin + Date.now()),
    raceStartedAt: 0,
    createdAt: Date.now(),
  };
  await store.setMeta(pin, meta);
  return NextResponse.json({ pin, hostKey, storage: store.kind });
}
