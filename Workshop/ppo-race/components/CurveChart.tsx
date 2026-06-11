"use client";

import {
  CURVE_POINTS,
  RETURN_MAX,
  RETURN_MIN,
  STEPS_PER_ROUND,
  TOTAL_ROUNDS,
} from "@/lib/constants";

export interface CurveSeries {
  curve: number[];
  color: string;
  label?: string;
  width?: number;
}

/**
 * SVG learning-curve chart with a fixed x-domain over all rounds, so curves
 * visibly grow to the right as the game progresses.
 */
/** Indices where the curve drops sharply into low return, i.e. a collapse. */
function collapseIdxs(curve: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < curve.length; i++) {
    if (curve[i - 1] - curve[i] > 220 && curve[i] < 250) out.push(i);
  }
  return out;
}

export default function CurveChart({
  series,
  height = 300,
  animate = false,
  highlightRoundBoundaries = true,
  annotate = false,
}: {
  series: CurveSeries[];
  height?: number;
  animate?: boolean;
  highlightRoundBoundaries?: boolean;
  /** shade the "good policy" target band and mark collapse points */
  annotate?: boolean;
}) {
  const W = 800;
  const H = height;
  const PAD = { left: 48, right: 14, top: 14, bottom: 26 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const totalPoints = TOTAL_ROUNDS * CURVE_POINTS;

  const x = (i: number) => PAD.left + (i / (totalPoints - 1)) * innerW;
  const y = (v: number) =>
    PAD.top + (1 - (v - RETURN_MIN) / (RETURN_MAX - RETURN_MIN)) * innerH;

  const yTicks = [0, 300, 600, 900];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full h-auto"
      role="img"
      aria-label="Learning curves"
    >
      {/* "good policy" target band */}
      {annotate && (
        <g>
          <rect
            x={PAD.left}
            y={y(RETURN_MAX)}
            width={innerW}
            height={y(700) - y(RETURN_MAX)}
            fill="rgba(34,197,94,0.10)"
          />
          <text
            x={PAD.left + 6}
            y={y(RETURN_MAX) + 14}
            fontSize={11}
            fill="rgba(134,239,172,0.9)"
          >
            goed beleid
          </text>
        </g>
      )}

      {/* grid + axes */}
      {yTicks.map((v) => (
        <g key={v}>
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={y(v)}
            y2={y(v)}
            stroke="rgba(148,163,184,0.18)"
            strokeWidth={1}
          />
          <text
            x={PAD.left - 8}
            y={y(v) + 4}
            textAnchor="end"
            fontSize={12}
            fill="rgba(148,163,184,0.8)"
          >
            {v}
          </text>
        </g>
      ))}
      {highlightRoundBoundaries &&
        Array.from({ length: TOTAL_ROUNDS - 1 }, (_, r) => (
          <line
            key={r}
            x1={x((r + 1) * CURVE_POINTS - 0.5)}
            x2={x((r + 1) * CURVE_POINTS - 0.5)}
            y1={PAD.top}
            y2={H - PAD.bottom}
            stroke="rgba(148,163,184,0.25)"
            strokeDasharray="4 6"
          />
        ))}
      {Array.from({ length: TOTAL_ROUNDS }, (_, r) => (
        <text
          key={r}
          x={x(r * CURVE_POINTS + CURVE_POINTS / 2)}
          y={H - 8}
          textAnchor="middle"
          fontSize={12}
          fill="rgba(148,163,184,0.7)"
        >
          {`ronde ${r + 1} · ${((r + 1) * STEPS_PER_ROUND) / 1000}k steps`}
        </text>
      ))}
      <text
        x={12}
        y={H / 2}
        fontSize={12}
        fill="rgba(148,163,184,0.8)"
        textAnchor="middle"
        transform={`rotate(-90 12 ${H / 2})`}
      >
        return
      </text>

      {series.map((s, idx) => {
        if (s.curve.length < 2) return null;
        const d = s.curve
          .map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)
          .join(" ");
        return (
          <path
            key={idx}
            d={d}
            fill="none"
            stroke={s.color}
            strokeWidth={s.width ?? 2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
            className={animate ? "curve-draw" : undefined}
            style={
              animate
                ? ({ "--curve-len": 2000 } as React.CSSProperties)
                : undefined
            }
          />
        );
      })}

      {/* collapse markers */}
      {annotate &&
        series.flatMap((s, idx) =>
          collapseIdxs(s.curve).map((i) => (
            <text
              key={`c${idx}-${i}`}
              x={x(i)}
              y={y(s.curve[i]) + 6}
              fontSize={15}
              textAnchor="middle"
            >
              💥
            </text>
          ))
        )}
    </svg>
  );
}
