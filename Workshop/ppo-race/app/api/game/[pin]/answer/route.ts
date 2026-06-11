import { NextResponse } from "next/server";
import { CLIP_OPTIONS, RETURN_START } from "@/lib/constants";
import { roundPoints, simulateRound } from "@/lib/sim";
import { getStore } from "@/lib/store";
import type { RoundRecord } from "@/lib/types";

export async function POST(
  req: Request,
  { params }: { params: Promise<{ pin: string }> }
) {
  const { pin } = await params;
  const store = getStore();
  const meta = await store.getMeta(pin);
  if (!meta) {
    return NextResponse.json({ error: "Game niet gevonden" }, { status: 404 });
  }
  if (meta.phase !== "round" || Date.now() > meta.roundEndsAt + 2000) {
    return NextResponse.json(
      { error: "Deze ronde is niet (meer) open" },
      { status: 409 }
    );
  }
  const body = await req.json().catch(() => ({}));
  const playerId = String(body.playerId ?? "");
  const clip = Number(body.clip);
  if (!CLIP_OPTIONS.includes(clip)) {
    return NextResponse.json({ error: "Ongeldige clip ε" }, { status: 400 });
  }
  const player = await store.getPlayer(pin, playerId);
  if (!player) {
    return NextResponse.json({ error: "Speler niet gevonden" }, { status: 404 });
  }
  if (player.rounds[meta.round]) {
    return NextResponse.json(
      { error: "Je hebt deze ronde al getraind" },
      { status: 409 }
    );
  }

  const prev = player.rounds[meta.round - 1];
  const startP = prev ? prev.endP : RETURN_START;
  const sim = simulateRound(clip, startP, `${pin}:${playerId}:${meta.round}`);
  const record: RoundRecord = { ...sim, points: roundPoints(sim) };
  player.rounds[meta.round] = record;
  await store.setPlayer(pin, player);

  return NextResponse.json({ record });
}
