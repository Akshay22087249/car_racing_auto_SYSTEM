"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { RACE_LAPS } from "@/lib/constants";
import { buildTrack, carAt, pointAt } from "@/lib/race";
import type { RaceCar, Verdict } from "@/lib/types";

/** Why a car spins, tied back to how its agent was tuned (E3). */
function spinReason(v: Verdict): string {
  switch (v) {
    case "collapsed":
      return "beleid ingestort, clip/lr te groot";
    case "unstable":
      return "grillig getuned, grip kwijt";
    case "slow":
      return "traag beleid, weinig controle";
    default:
      return "kleine misser";
  }
}

interface Standing {
  car: RaceCar;
  dist: number;
  lap: number;
  finished: boolean;
  position: number;
}

/**
 * Replays the deterministic race that the server already scored: same seed,
 * same spin events, same finish times, only the rendering happens here.
 */
export default function RaceView({
  cars,
  seed,
  startedAt,
  offsetRef,
  onAllFinished,
}: {
  cars: RaceCar[];
  seed: number;
  startedAt: number;
  offsetRef: React.RefObject<number>;
  onAllFinished?: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [standings, setStandings] = useState<Standing[]>([]);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [toasts, setToasts] = useState<{ id: number; text: string }[]>([]);
  const idRef = useRef(0);
  const doneRef = useRef(false);
  const spinRef = useRef<Record<string, boolean>>({});
  const finRef = useRef<Record<string, boolean>>({});
  // keep the callback in a ref so re-renders don't restart the animation loop
  const onAllFinishedRef = useRef(onAllFinished);
  useEffect(() => {
    onAllFinishedRef.current = onAllFinished;
  }, [onAllFinished]);

  const track = useMemo(() => buildTrack(seed), [seed]);
  const totalDist = track.length * RACE_LAPS;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = 1280;
    const H = 840;
    canvas.width = W;
    canvas.height = H;

    // fit track into canvas
    const xs = track.points.map((p) => p.x);
    const ys = track.points.map((p) => p.y);
    const minX = Math.min(...xs) - 40;
    const maxX = Math.max(...xs) + 40;
    const minY = Math.min(...ys) - 40;
    const maxY = Math.max(...ys) + 40;
    const scale = Math.min(W / (maxX - minX), H / (maxY - minY));
    const tx = (x: number) => (x - (minX + maxX) / 2) * scale + W / 2;
    const ty = (y: number) => (y - (minY + maxY) / 2) * scale + H / 2;

    const trackPath = new Path2D();
    track.points.forEach((p, i) => {
      if (i === 0) trackPath.moveTo(tx(p.x), ty(p.y));
      else trackPath.lineTo(tx(p.x), ty(p.y));
    });
    trackPath.closePath();

    const maxFinish = Math.max(...cars.map((c) => c.finishTime), 0);
    let raf = 0;
    let lastStandingsUpdate = 0;

    const drawTrack = () => {
      ctx.fillStyle = "#14532d";
      ctx.fillRect(0, 0, W, H);
      // subtle grass texture dots
      ctx.fillStyle = "rgba(255,255,255,0.03)";
      for (let i = 0; i < 60; i++) {
        ctx.fillRect(((i * 211) % W) | 0, ((i * 137) % H) | 0, 3, 3);
      }
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 34 * scale;
      ctx.stroke(trackPath);
      ctx.strokeStyle = "#334155";
      ctx.lineWidth = 30 * scale;
      ctx.stroke(trackPath);
      ctx.strokeStyle = "rgba(226,232,240,0.5)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([12, 14]);
      ctx.stroke(trackPath);
      ctx.setLineDash([]);
      // start/finish line
      const start = pointAt(track, 0);
      const n = { x: -Math.sin(start.heading), y: Math.cos(start.heading) };
      const half = 15 * scale;
      const sq = 5 * scale;
      for (let i = -3; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
          ctx.fillStyle = (i + j) % 2 === 0 ? "#fff" : "#0f172a";
          ctx.fillRect(
            tx(start.x) + n.x * ((i * sq) / scale) * scale + (j - 1) * sq * 0.9,
            ty(start.y) + n.y * ((i * sq) / scale) * scale - half / 5,
            sq * 0.9,
            sq * 0.9
          );
        }
      }
    };

    const drawCar = (car: RaceCar, idx: number, t: number) => {
      const st = carAt(car, totalDist, Math.max(t, 0));
      const pos = pointAt(track, st.dist);
      const wobbleAmp = (1 - car.skill) * 6;
      const wobble = Math.sin(st.dist * 0.09 + idx * 2.1) * wobbleAmp;
      const nx = -Math.sin(pos.heading);
      const ny = Math.cos(pos.heading);
      // spread cars slightly across the road so they don't fully overlap
      const lane = ((idx % 5) - 2) * 4.5;
      const cx = tx(pos.x + (wobble + lane) * nx);
      const cy = ty(pos.y + (wobble + lane) * ny);
      let heading = pos.heading + Math.cos(st.dist * 0.09 + idx * 2.1) * wobbleAmp * 0.02;
      if (st.spinning) heading += st.spinProgress * Math.PI * 4;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(heading);
      const L = 13 * scale;
      const Wc = 7 * scale;
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      ctx.fillRect(-L / 2 + 2, -Wc / 2 + 2, L, Wc);
      ctx.fillStyle = car.color;
      ctx.strokeStyle = "white";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(-L / 2, -Wc / 2, L, Wc, 3);
      ctx.fill();
      ctx.stroke();
      // windshield
      ctx.fillStyle = "rgba(15,23,42,0.7)";
      ctx.fillRect(L * 0.05, -Wc * 0.3, L * 0.25, Wc * 0.6);
      ctx.restore();

      ctx.font = `600 ${Math.max(13, 5 * scale)}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillStyle = "rgba(255,255,255,0.92)";
      ctx.strokeStyle = "rgba(0,0,0,0.6)";
      ctx.lineWidth = 3;
      ctx.strokeText(car.name, cx, cy - 12 * scale);
      ctx.fillText(car.name, cx, cy - 12 * scale);
      if (st.finished) {
        ctx.fillText("🏁", cx, cy - 19 * scale);
      }
    };

    const frame = () => {
      const now = Date.now() + (offsetRef.current ?? 0);
      const t = (now - startedAt) / 1000;
      drawTrack();
      cars.forEach((car, idx) => drawCar(car, idx, t));

      setCountdown(t < 0 ? Math.ceil(-t) : null);

      if (now - lastStandingsUpdate > 300) {
        lastStandingsUpdate = now;
        const rows = cars
          .map((car) => {
            const st = carAt(car, totalDist, Math.max(t, 0));
            return {
              car,
              dist: st.dist,
              lap: Math.min(Math.floor(st.dist / track.length) + 1, RACE_LAPS),
              finished: st.finished,
            };
          })
          .sort((a, b) =>
            a.finished && b.finished
              ? a.car.finishTime - b.car.finishTime
              : b.dist - a.dist
          )
          .map((row, i) => ({ ...row, position: i + 1 }));
        setStandings(rows);

        // commentary: detect new spins / finishes since the last tick
        const events: string[] = [];
        cars.forEach((car) => {
          const st = carAt(car, totalDist, Math.max(t, 0));
          if (st.spinning && !spinRef.current[car.id]) {
            spinRef.current[car.id] = true;
            events.push(`💥 ${car.name} tolt: ${spinReason(car.lastVerdict)}`);
          } else if (!st.spinning && spinRef.current[car.id]) {
            spinRef.current[car.id] = false;
          }
          if (st.finished && !finRef.current[car.id]) {
            finRef.current[car.id] = true;
            events.push(`🏁 ${car.name} over de finish!`);
          }
        });
        if (events.length) {
          const items = events.map((text) => ({ id: idRef.current++, text }));
          setToasts((cur) => [...items, ...cur].slice(0, 4));
          // pop-ups are transient: drop each after a few seconds
          items.forEach((it) =>
            setTimeout(
              () => setToasts((cur) => cur.filter((x) => x.id !== it.id)),
              4500
            )
          );
        }

        if (t > maxFinish + 1.5 && !doneRef.current) {
          doneRef.current = true;
          onAllFinishedRef.current?.();
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [cars, track, totalDist, startedAt, offsetRef]);

  return (
    <div className="flex gap-6 items-start w-full">
      <div className="relative flex-1 min-w-0">
        <canvas
          ref={canvasRef}
          className="w-full h-auto rounded-2xl shadow-2xl"
        />
        {countdown !== null && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-9xl font-black text-white drop-shadow-[0_4px_24px_rgba(0,0,0,0.8)]">
              {countdown}
            </span>
          </div>
        )}
        <div className="pointer-events-none absolute inset-x-0 top-3 flex flex-col items-center gap-2 px-4">
          {toasts.map((t) => (
            <div
              key={t.id}
              className="race-toast max-w-[90%] rounded-full border border-white/15 bg-slate-900/85 px-4 py-2 text-center text-base font-semibold text-white shadow-xl backdrop-blur-sm"
            >
              {t.text}
            </div>
          ))}
        </div>
      </div>
      <ol className="w-72 shrink-0 space-y-1.5 max-h-[70vh] overflow-y-auto pr-1">
        {standings.map((row) => (
          <li
            key={row.car.id}
            className="flex items-center gap-2 rounded-lg bg-slate-800/80 px-3 py-1.5 text-sm"
          >
            <span className="w-7 font-bold text-slate-300 tabular-nums">
              {row.position}.
            </span>
            <span
              className="inline-block h-3 w-3 rounded-full shrink-0"
              style={{ background: row.car.color }}
            />
            <span className="truncate font-medium">{row.car.name}</span>
            <span className="ml-auto text-xs text-slate-400 tabular-nums">
              {row.finished
                ? `🏁 ${row.car.finishTime.toFixed(1)}s`
                : `lap ${row.lap}/${RACE_LAPS}`}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
