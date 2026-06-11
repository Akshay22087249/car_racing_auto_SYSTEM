import { NextResponse } from "next/server";
import { buildState, effectiveMeta } from "@/lib/game";
import { getStore } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(
  req: Request,
  { params }: { params: Promise<{ pin: string }> }
) {
  const { pin } = await params;
  const store = getStore();
  let meta = await store.getMeta(pin);
  if (!meta) {
    return NextResponse.json({ error: "Game niet gevonden" }, { status: 404 });
  }
  const url = new URL(req.url);
  const hostKey = url.searchParams.get("hostKey") ?? undefined;
  const playerId = url.searchParams.get("playerId") ?? undefined;
  const isHost = Boolean(hostKey && hostKey === meta.hostKey);

  const players = Object.values(await store.getPlayers(pin));
  meta = await effectiveMeta(store, meta, players);

  const race =
    isHost && (meta.phase === "race" || meta.phase === "podium")
      ? await store.getRace(pin)
      : null;

  const state = buildState(meta, players, store.kind, {
    isHost,
    playerId,
    race,
  });
  return NextResponse.json(state);
}
