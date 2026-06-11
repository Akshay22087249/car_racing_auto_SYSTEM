import { NextResponse } from "next/server";
import {
  CLIP_OPTIONS,
  LR_OPTIONS,
  RETURN_START,
  STREAK_BONUS,
  roundEnv,
} from "@/lib/constants";
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
  const env = roundEnv(meta.round);
  // the lr knob only appears from round 2; otherwise it sits at the optimum
  let lr = env.lrOpt;
  if (env.showLr) {
    lr = Number(body.lr);
    if (!LR_OPTIONS.includes(lr)) {
      return NextResponse.json({ error: "Ongeldige learning rate" }, { status: 400 });
    }
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
  const sim = simulateRound(
    clip,
    lr,
    startP,
    `${pin}:${playerId}:${meta.round}`,
    meta.round
  );

  // streak bonus: count consecutive "stabiel" rounds ending with this one
  let streak = 0;
  for (let r = meta.round; r >= 1; r--) {
    const v = r === meta.round ? sim.verdict : player.rounds[r]?.verdict;
    if (v === "good") streak++;
    else break;
  }
  const bonus = streak >= 2 ? STREAK_BONUS * (streak - 1) : 0;

  const record: RoundRecord = {
    ...sim,
    bonus,
    points: roundPoints(sim) + bonus,
  };
  player.rounds[meta.round] = record;
  await store.setPlayer(pin, player);

  return NextResponse.json({ record });
}
