# 🏎️ PPO Race

Kahoot-stijl workshopgame over PPO-hyperparameter-tuning, gebaseerd op
[Emergent Autonomous Racing via Multi-Agent PPO](https://github.com/rmsander/marl_ppo)
(Sander, MIT).

In plaats van "één getal raden + leaderboard" zit het échte PPO-leerproces in
het spel: deelnemers kiezen per ronde een **clip ε**, zien hun **learning
curve** direct reageren en sturen bij in de volgende ronde. Pas na het
"trainen" racen de resulterende agents tegen elkaar.

## Spelverloop

1. **Lobby** — host opent het beamerscherm, deelnemers joinen op hun telefoon
   met een 4-cijferige PIN en een naam.
2. **3 tuning-rondes** (60 s per ronde) — elke speler kiest een clip ε:
   - te klein (≤ 0.05) → vlakke, trage curve 🐌
   - sweet spot (≈ 0.1–0.3) → stabiel stijgende curve 🚀
   - te groot (≥ 0.5) → grillige curve met instortingsgevaar 💥
   Je traint elke ronde **verder vanaf je vorige curve** (cumulatief, zoals
   echte training). Na elke ronde toont de beamer alle curves over elkaar.
   Punten per ronde = vooruitgang van je agent.
3. **De race** 🏁 — een top-down 2D-race op een procedureel circuit. Het
   rijgedrag van elke auto volgt uit de eindprestatie van de getunede agent:
   goed getuned = snel en strak, slecht getuned = traag, slingerend, spins.
   De race levert bonuspunten op (1e = 500, daarna aflopend).
4. **Podium** — eindstand = rondepunten + racebonus.

De ronde sluit automatisch zodra iedereen heeft getraind (of de timer afloopt).

## Lokaal draaien

```bash
npm install
npm run dev
```

Open `http://localhost:3000/host` (beamer) en `http://localhost:3000`
(spelers). Zonder Redis-credentials gebruikt de app automatisch een
**in-memory store** — prima voor lokaal testen, maar niet op Vercel.

## Deployen op Vercel

1. Push deze repo naar GitHub.
2. Vercel → **Add New Project** → importeer de repo →
   zet **Root Directory** op `Workshop/ppo-race` (Framework: Next.js wordt
   automatisch herkend).
3. Koppel Redis (nodig omdat serverless functions geen geheugen delen):
   **Storage** (of Marketplace) → **Upstash for Redis** → gratis database →
   *Connect to project*. Daarmee staan `UPSTASH_REDIS_REST_URL` en
   `UPSTASH_REDIS_REST_TOKEN` (of `KV_REST_API_URL`/`KV_REST_API_TOKEN`)
   automatisch in je environment; beide namen worden ondersteund.
4. Deploy. Host-URL: `https://<jouw-app>.vercel.app/host`.

> Het hostscherm onthoudt de hostKey in localStorage én in de URL
> (`/host/<pin>?key=…`), dus je kunt het beamerscherm openen op een andere
> computer dan waar je de game aanmaakte.

Realtime werkt via **polling (1×/s)** — bewust gekozen omdat Vercel serverless
geen WebSockets ondersteunt. Met ~30 deelnemers blijf je ruim binnen de gratis
tiers van Vercel en Upstash.

## Knoppen om aan te draaien

| Wat | Waar |
| --- | --- |
| Aantal rondes, rondeduur, clip-opties | `lib/constants.ts` |
| Curve-simulator (groei, instabiliteit, collapse-kans) | `lib/sim.ts` |
| Hints/teksten per curve-type | `lib/sim.ts` (`VERDICT_HINTS`) |
| Race (laps, snelheden, spins) | `lib/constants.ts` + `lib/race.ts` |
| Punten (ronde + race) | `lib/sim.ts` (`roundPoints`) + `lib/constants.ts` |

## Architectuur

- **Next.js (App Router)** — API-routes onder `app/api/game/…`, schermen voor
  host (`/host/[pin]`) en speler (`/play/[pin]`).
- **Upstash Redis** — gamestate (meta + spelers als hash + raceresultaat),
  met in-memory fallback voor lokaal ontwikkelen.
- **Deterministische simulatie** — curves en race komen uit een seeded RNG:
  de server berekent de uitslag, het beamerscherm speelt exact dezelfde race
  af. Geen echte training nodig; de curve-vormen zijn gemodelleerd naar
  textbook-PPO-gedrag (clip te groot/te klein).
