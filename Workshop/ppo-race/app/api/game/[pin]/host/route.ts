import { NextResponse } from "next/server";
import { racePointsForPosition, ROUND_SECONDS, TOTAL_ROUNDS } from "@/lib/constants";
import { buildRaceCars } from "@/lib/race";
import { getStore } from "@/lib/store";

type Action = "startRound" | "endRound" | "startRace" | "podium";

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
  const body = await req.json().catch(() => ({}));
  if (String(body.hostKey ?? "") !== meta.hostKey) {
    return NextResponse.json({ error: "Geen geldige hostKey" }, { status: 403 });
  }
  const action = String(body.action ?? "") as Action;

  switch (action) {
    case "startRound": {
      const fromLobby = meta.phase === "lobby";
      const fromResults =
        meta.phase === "results" && meta.round < TOTAL_ROUNDS;
      if (!fromLobby && !fromResults) {
        return NextResponse.json(
          { error: `Kan geen ronde starten vanuit fase '${meta.phase}'` },
          { status: 409 }
        );
      }
      meta.phase = "round";
      meta.round = fromLobby ? 1 : meta.round + 1;
      meta.roundEndsAt = Date.now() + ROUND_SECONDS * 1000;
      break;
    }
    case "endRound": {
      if (meta.phase !== "round") {
        return NextResponse.json({ error: "Geen ronde bezig" }, { status: 409 });
      }
      meta.phase = "results";
      break;
    }
    case "startRace": {
      if (!(meta.phase === "results" && meta.round >= TOTAL_ROUNDS)) {
        return NextResponse.json(
          { error: "De race kan pas na de laatste ronde starten" },
          { status: 409 }
        );
      }
      const players = Object.values(await store.getPlayers(pin));
      const cars = buildRaceCars(players, meta.raceSeed);
      const order = [...cars].sort((a, b) => a.finishTime - b.finishTime);
      for (const [i, car] of order.entries()) {
        const player = players.find((p) => p.id === car.id);
        if (!player) continue;
        player.racePosition = i + 1;
        player.racePoints = racePointsForPosition(i + 1);
        await store.setPlayer(pin, player);
      }
      await store.setRace(pin, cars);
      meta.phase = "race";
      meta.raceStartedAt = Date.now() + 3000; // 3s countdown on the host screen
      break;
    }
    case "podium": {
      if (meta.phase !== "race") {
        return NextResponse.json({ error: "Er is geen race bezig" }, { status: 409 });
      }
      meta.phase = "podium";
      break;
    }
    default:
      return NextResponse.json({ error: "Onbekende actie" }, { status: 400 });
  }

  await store.setMeta(pin, meta);
  return NextResponse.json({ ok: true, phase: meta.phase, round: meta.round });
}
