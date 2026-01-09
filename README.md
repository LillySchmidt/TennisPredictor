# Tennis Match Predictor Pro

A professional tennis match prediction application using an ELO rating system.

**Author: Lilly-Sophie Schmidt**

## Quick Start

```bash
# Install dependencies
npm install

# Start the frontend development server
npm run dev

# In a separate terminal, start the backend
cd backend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`

## Features

### ELO-Based Predictions
- **1,706 players** with calculated ELO ratings
- **Surface-specific ratings** for Hard, Clay, and Grass courts
- **Win probability** calculated using the standard ELO formula
- **Historical data** from 1991-2024 (43,003 matches)

### Application Features
- Import custom tournaments with player draws
- View match predictions with ELO ratings and win probabilities
- Track prediction accuracy as matches complete
- Dark/light theme support

## How ELO Predictions Work

The win probability is calculated using:
```
P(win) = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
```

For example, **Sinner (2409 Hard ELO) vs Alcaraz (2102 Hard ELO)**:
- Sinner win probability: **87%**
- Alcaraz win probability: **13%**

## Current Top 10 ELO Ratings

| Rank | Player | Overall | Hard | Clay | Grass |
|------|--------|---------|------|------|-------|
| 1 | Jannik Sinner | 2337 | 2409 | 1894 | 1805 |
| 2 | Novak Djokovic | 2224 | 2349 | 2158 | 2072 |
| 3 | Carlos Alcaraz | 2193 | 2102 | 2102 | 1951 |
| 4 | Alexander Zverev | 2125 | 2153 | 2019 | 1782 |
| 5 | Taylor Fritz | 2120 | 2188 | 1819 | 1752 |
| 6 | Daniil Medvedev | 2053 | 2130 | 1796 | 1798 |
| 7 | Casper Ruud | 2032 | 1963 | 2024 | 1581 |
| 8 | Grigor Dimitrov | 1986 | 1996 | 1895 | 1857 |
| 9 | Tommy Paul | 1977 | 2015 | 1759 | 1755 |
| 10 | Hubert Hurkacz | 1973 | 2029 | 1726 | 1842 |

## All-Time Peak ELO

| Player | Peak ELO | Date |
|--------|----------|------|
| Novak Djokovic | 2606 | May 2016 |
| Rafael Nadal | 2547 | Aug 2013 |
| Roger Federer | 2473 | Apr 2007 |
| Andy Murray | 2470 | Jan 2017 |
| Jannik Sinner | 2337 | Nov 2024 |

## Project Structure

```
tennis-predictor-final/
├── src/                    # Frontend React application
│   ├── assets/
│   │   └── player-elo.json # ELO ratings for 1,706 players
│   ├── components/         # React components
│   └── pages/              # Page components
├── backend/                # Express.js backend
│   └── src/
│       ├── routes/         # API routes
│       └── services/       # Business logic
├── elo-system/             # Standalone ELO tools
│   ├── tennis_elo.py       # Core ELO implementation
│   ├── query_elo.py        # Interactive query tool
│   ├── player_ratings.csv  # Full player ratings
│   └── data/               # Source match data
└── package.json
```

## ELO System (Python)

The `elo-system/` folder contains standalone Python tools for working with the ELO ratings:

```bash
cd elo-system

# Query a player's current ELO
python query_elo.py "Novak Djokovic"

# Query historical ELO
python query_elo.py "Novak Djokovic" 2016-06-06

# Interactive mode
python query_elo.py
```

### ELO System Features
- **Dynamic K-factor** based on tournament importance (Grand Slams weighted highest)
- **Inactivity decay** - ratings regress toward mean after 60 days without matches
- **Surface-specific ratings** - separate ELOs for Hard, Clay, and Grass
- **Historical tracking** - query any player's ELO at any point in time

## Tech Stack

### Frontend
- React 18 + TypeScript
- Vite
- Tailwind CSS
- shadcn/ui components

### Backend
- Express.js
- SQLite (better-sqlite3)
- TypeScript

### ELO System
- Python 3
- pandas, numpy

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tournaments` | List all tournaments |
| POST | `/api/tournaments` | Create new tournament |
| GET | `/api/tournaments/:id` | Get tournament details |
| GET | `/api/tournaments/:id/matches` | Get tournament matches |
| POST | `/api/tournaments/:id/matches/:matchId/winner` | Record match winner |

## License

MIT License - Free to use and modify.

---
© 2026 Lilly-Sophie Schmidt
