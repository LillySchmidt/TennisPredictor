# Tennis ELO Rating System

Standalone Python tools for tennis ELO calculations and analysis.

**Author: Lilly-Sophie Schmidt**

## Quick Start

```bash
# Query a player
python query_elo.py "Jannik Sinner"

# Query historical ELO
python query_elo.py "Novak Djokovic" 2016-06-06

# Interactive mode
python query_elo.py
```

## Files

| File | Description |
|------|-------------|
| `tennis_elo.py` | Core ELO system implementation |
| `query_elo.py` | Interactive command-line query tool |
| `demo_elo_analysis.py` | Full analysis script |
| `player_ratings.csv` | All 1,706 player ratings |
| `elo_history.csv` | Complete match-by-match history |
| `surface_ratings.csv` | Surface-specific ratings |
| `elo_comparison_chart.png` | Big 3 + Sinner comparison chart |
| `elo_history_charts.png` | Individual player history charts |

## Python API

```python
from tennis_elo import TennisEloSystem, EloConfig, load_and_process_matches

# Load system
elo_system = load_and_process_matches('data/matches_with_stats_cleaned.csv')

# Get current ELO
elo = elo_system.get_elo("Novak Djokovic")

# Get historical ELO
from datetime import datetime
peak_elo = elo_system.get_elo("Novak Djokovic", datetime(2016, 6, 6))

# Get surface-specific ELO
clay_elo = elo_system.get_surface_elo("Rafael Nadal", "Clay")

# Calculate win probability
prob = elo_system.expected_score(2300, 2100)  # ~76%

# Get rankings
top_20 = elo_system.get_rankings(20)
```

## ELO Configuration

```python
config = EloConfig(
    initial_elo=1500.0,
    k_factors={
        'G': 48,   # Grand Slams
        'F': 40,   # Tour Finals
        'M': 36,   # Masters 1000
        'A': 32,   # ATP 250/500
        'O': 24,   # Other
    },
    decay_enabled=True,
    decay_days_threshold=60,
    decay_rate_per_day=0.5,
    decay_cap=200.0,
    mov_enabled=True,
)
```

---
© 2024 Lilly-Sophie Schmidt
