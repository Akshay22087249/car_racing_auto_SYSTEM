"use client";

import { useEffect, useState } from "react";

export default function Countdown({
  endsAt,
  offsetRef,
  className = "",
}: {
  endsAt: number;
  offsetRef: React.RefObject<number>;
  className?: string;
}) {
  const [left, setLeft] = useState(0);
  useEffect(() => {
    const tick = () =>
      setLeft(
        Math.max(0, Math.ceil((endsAt - Date.now() - (offsetRef.current ?? 0)) / 1000))
      );
    tick();
    const id = setInterval(tick, 250);
    return () => clearInterval(id);
  }, [endsAt, offsetRef]);
  return (
    <span className={`tabular-nums ${className} ${left <= 10 ? "text-red-400" : ""}`}>
      {left}
    </span>
  );
}
