import { NextResponse } from "next/server";
import { playerColor } from "@/lib/constants";
import { randomId } from "@/lib/game";
import { getStore } from "@/lib/store";
import { TEAM_CAPACITY, teamById } from "@/lib/teams";
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
  // late joiners may hop in during the tuning rounds (they start fresh);
  // only the race and podium are closed to newcomers
  if (meta.phase === "race" || meta.phase === "podium") {
    return NextResponse.json(
      { error: "De race is al bezig, je kunt niet meer meedoen" },
      { status: 409 }
    );
  }
  const body = await req.json().catch(() => ({}));
  const name = String(body.name ?? "").trim().slice(0, 16);
  if (!name) {
    return NextResponse.json({ error: "Vul een naam in" }, { status: 400 });
  }

  const players = Object.values(await store.getPlayers(pin));

  // a chosen F1 team sets the livery colour and enforces the 2-driver cap
  const team = teamById(String(body.team ?? ""));
  let color: string;
  let teamId: string | undefined;
  let driverNumber: number | undefined;
  if (team) {
    const inTeam = players.filter((p) => p.team === team.id);
    if (inTeam.length >= TEAM_CAPACITY) {
      return NextResponse.json(
        { error: `${team.name} zit al vol (2 coureurs)` },
        { status: 409 }
      );
    }
    teamId = team.id;
    color = team.color;
    driverNumber = inTeam.length + 1;
  } else {
    // privateer fallback when no team is chosen
    color = playerColor(players.length);
  }

  const player: PlayerData = {
    id: randomId(),
    name,
    color,
    team: teamId,
    driverNumber,
    joinedAt: Date.now(),
    rounds: {},
  };
  await store.setPlayer(pin, player);
  return NextResponse.json({ playerId: player.id, color: player.color });
}
