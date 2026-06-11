import { NextResponse } from "next/server";
import { freshPin, randomId } from "@/lib/game";
import { getStore } from "@/lib/store";
import { hashSeed } from "@/lib/rng";
import type { GameMeta } from "@/lib/types";

export async function POST(req: Request) {
  // gate game creation behind a host password when one is configured.
  // HOST_PASSWORD is server-only (no NEXT_PUBLIC_), so it never reaches the
  // client; leave it unset for open local development.
  const expected = process.env.HOST_PASSWORD;
  if (expected) {
    const body = await req.json().catch(() => ({}));
    if (String(body?.password ?? "") !== expected) {
      return NextResponse.json(
        { error: "Verkeerd host-wachtwoord" },
        { status: 403 }
      );
    }
  }

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
