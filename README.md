# CarRacing: Rule-Based en Reinforcement Learning Agents

Een universitair portfolioproject voor de vakken **Rule-Based Systems** en **Reinforcement Learning**. Zeven rule-based agents en DQN-agents rijden in de Gymnasium `CarRacing-v3` omgeving. Daarnaast is er een interactief racespel waarbij je zelf kunt racen tegen de AI-agents.

---

## Agents

| Agent | Type | Strategie |
|---|---|---|
| Random baseline | Rule-based | Willekeurige acties (ondergrens) |
| Cautious (My Sister) | Rule-based | Veiligheid eerst, vroeg remmen, zachte correcties |
| Max Verstappen | Rule-based | Langzaam in, snel uit via een toestandsmachine |
| Traction Focused | Rule-based | Frictiecirkel: sturen kost budget |
| MotoGP | Rule-based | Hiërarchisch prioriteitssysteem met adaptief vertrouwen |
| Rain Rider | Rule-based | Adaptieve agressiviteit op basis van recente prestaties |
| Line Hunter | Rule-based | Gewogen trajectplanning via near/mid/far offset |
| DQN-agent | Reinforcement Learning | Zelflerend via Deep Q-Network, getraind op 500 episodes |


---

## Resultaten (10 evaluatie-episodes, seed 42)

| Rang | Agent | Gem. beloning | Voltooiing | Std. afwijking |
|------|-------|--------------|-----------|----------------|
| 1 | DQN-agent | 783.9 | 87.1% | 34.4 |
| 2 | Cautious (My Sister) | 630.8 | 70.7% | 290.6 |
| 3 | Line Hunter | 626.9 | 69.7% | 233.7 |
| 4 | Traction Focused | 626.7 | 70.1% | 324.5 |
| 5 | Rain Rider | 623.3 | 69.8% | 288.8 |
| 6 | MotoGP | 621.5 | 69.1% | 309.8 |
| 7 | Max Verstappen | 578.7 | 64.9% | 328.3 |
| 8 | Random Baseline | -127.8 | 0.0% | 7.1 |

De DQN-agent presteert beter dan alle rule-based agents en is veel consistenter (standaardafwijking van 34 tegenover 290+ bij rule-based agents).

---

## Vergelijkingsvideo

Alle agents naast elkaar in één video:
https://www.youtube.com/watch?v=Mksyq7B8tyU

---

## Installatie

```bash
pip install swig
pip install -r requirements.txt
```

Voor GPU-training (NVIDIA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Let op:** `gymnasium[box2d]` vereist [SWIG](https://www.swig.org/) voor Box2D. Op Windows: `pip install swig`.

---

## Gebruik

### Rule-based agents evalueren
```bash
python main.py --episodes 10
```

### Vergelijkingsvideo genereren
```bash
python compare_video.py
```

### Racespel spelen (jij vs. AI)
```bash
python generate_ghosts_batch.py
python race_game.py
```

### Standaard DQN trainen
```bash
python train_dqn.py --episodes 500 --epsilon-decay 0.99999
```

### DQN met reward shaping trainen
```bash
python train_dqn_shaped.py
```

### Sweep uitvoeren voor shaped versie
```bash
python run_shaped_sweep.py
```

### Trainingsgrafieken bekijken
```bash
python plot_dqn.py
```

### DQN vergelijken met rule-based agents
```bash
python evaluate_dqn.py --episodes 10
```

### Alternatieve evaluatie
```bash
python evaluate_dqn_1.py --episodes 10
```

### Hyperparameter experiment uitvoeren
```bash
python hyperparam_experiment.py --episodes 150
```

---

## Projectstructuur

```
car_racing_auto_SYSTEM/
├── README.md
├── requirements.txt
├── config.py                        # Gedeelde constanten en hyperparameters
├── main.py                          # Evalueer rule-based agents, genereer plots
├── environment.py                   # Observatieverwerking (pixels naar features)
├── train_dqn.py                     # Train de standaard DQN-agent
├── train_dqn_shaped.py              # Train DQN met reward shaping
├── run_shaped_sweep.py              # Sweep voor shaped DQN configuraties
├── evaluate_dqn.py                  # Vergelijk DQN met rule-based agents
├── evaluate_dqn_1.py                # Alternatieve evaluatie
├── plot_dqn.py                      # Trainingsgrafieken genereren
├── hyperparam_experiment.py         # Hyperparameter vergelijking
├── compare_video.py                 # Vergelijkingsvideo alle agents
├── race_game.py                     # Racespel: mens vs. AI
├── generate_ghosts_batch.py         # Ghost data genereren voor racespel
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                # Abstracte basisklasse
│   ├── baseline_random.py           # Random baseline
│   ├── agent_cautious.py            # Cautious (My Sister)
│   ├── agent_apex.py                # Max Verstappen
│   ├── agent_traction.py            # Traction Focused
│   ├── agent_superbike.py           # MotoGP
│   ├── agent_rain.py                # Rain Rider
│   ├── agent_line_hunter.py         # Line Hunter
│   └── dqn_agent.py                 # Deep Q-Network agent
│
├── evaluation/
│   ├── metrics.py                   # Metrics per episode
│   ├── visualize.py                 # Vergelijkingsplots
│   └── compare_video.py             # Vergelijkingsvideo
│
├── game/
│   ├── race.py                      # Racespel loop en HUD
│   └── generate_ghosts.py           # Ghost data genereren
│
└── results/
    ├── results.csv                  # Rule-based resultaten
    ├── dqn/
    │   ├── dqn_training_log.csv     # DQN trainingslog
    │   ├── dqn_comparison.csv       # Vergelijking DQN vs. rule-based
    │   ├── checkpoints/             # Opgeslagen modellen
    │   │   ├── dqn_best.pt
    │   │   └── dqn_latest.pt
    │   └── hyperparams/             # Hyperparameter experiment resultaten
    ├── ghosts/                      # Ghost data voor racespel
    └── videos/                      # Opgenomen videos
```

---

## DQN implementatie

De DQN-agent is volledig zelf geïmplementeerd zonder gebruik van RL-bibliotheken zoals Stable Baselines. De belangrijkste onderdelen zijn:

**Neuraal netwerk:** drie convolutielagen die patronen uit de pixels halen, gevolgd door twee volledig verbonden lagen die Q-waarden berekenen per actie.

**Experience replay:** ervaringen worden opgeslagen in een buffer van 50.000 entries. Per trainingstap wordt een willekeurige batch van 64 ervaringen gesampled om correlatie te doorbreken.

**Doelnetwerk:** een apart netwerk voor stabiele doelwaarden, elke 1.000 stappen bijgewerkt.

**Epsilon-greedy exploratie:** epsilon daalt van 1.0 naar 0.05 over de training zodat de agent eerst verkent en later exploiteert.

**Framestack:** vier opeenvolgende grijswaarden frames worden gestapeld als invoer zodat het netwerk bewegingsinformatie heeft.

**Reward shaping (train_dqn_shaped.py):** een extra beloningssignaal dat de agent straft voor off-track rijden en beloont voor op de baan blijven, zodat hij netter leert rijden.

---

## Observatieverwerking (rule-based agents)

De `environment.py` module zet ruwe 96x96 RGB-frames om naar een feature dictionary:

| Feature | Omschrijving |
|---|---|
| `offset_near` | Baanmidden offset op korte afstand [-1, 1] |
| `offset_mid` | Baanmidden offset op middellange afstand [-1, 1] |
| `offset_far` | Baanmidden offset op verre afstand [-1, 1] |
| `curve_sharpness` | Scherpte van de bocht (0 = recht, 1 = haarspeld) |
| `speed` | Geschatte snelheid [0, 1] |
| `on_track` | Boolean: is de auto op de baan? |
| `warmup` | True tijdens de eerste 50 frames (inzoomanimatie) |

---

## Reproduceerbaarheid

Alle experimenten gebruiken een vaste basis-seed (standaard 42). Elke episode krijgt seed `basis_seed + episode`, zodat alle agents op exact dezelfde banen worden getest.
