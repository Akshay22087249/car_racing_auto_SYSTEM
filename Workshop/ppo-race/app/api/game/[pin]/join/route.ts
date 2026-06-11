import { NextResponse } from "next/server";
import { playerColor } from "@/lib/constants";
import { randomId } from "@/lib/game";
import { getStore } from "@/lib/store";
import type { PlayerData } from "@/lib/types";

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
  if (meta.phase !== "lobby") {
    return NextResponse.json(
      { error: "De game is al begonnen" },
      { status: 409 }
    );
  }
  const body = await req.json().catch(() => ({}));
  const name = String(body.name ?? "").trim().slice(0, 16);
  if (!name) {
    return NextResponse.json({ error: "Vul een naam in" }, { status: 400 });
  }
  const players = await store.getPlayers(pin);
  const player: PlayerData = {
    id: randomId(),
    name,
    color: playerColor(Object.keys(players).length),
    joinedAt: Date.now(),
    rounds: {},
  };
  await store.setPlayer(pin, player);
  return NextResponse.json({ playerId: player.id, color: player.color });
}
