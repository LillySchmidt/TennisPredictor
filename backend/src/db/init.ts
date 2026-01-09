import { db } from "./database";

const CREATE_TABLES_SQL = `
CREATE TABLE IF NOT EXISTS tournaments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  nation TEXT,
  surface TEXT,
  draw_size INTEGER NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS players (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tournament_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  is_bye INTEGER NOT NULL DEFAULT 0,
  seed_position INTEGER NOT NULL,
  FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tournament_id INTEGER NOT NULL,
  round_number INTEGER NOT NULL,
  round_name TEXT NOT NULL,
  match_order INTEGER NOT NULL,
  player1_id INTEGER,
  player2_id INTEGER,
  winner_id INTEGER,
  FOREIGN KEY (tournament_id) REFERENCES tournaments(id) ON DELETE CASCADE,
  FOREIGN KEY (player1_id) REFERENCES players(id) ON DELETE SET NULL,
  FOREIGN KEY (player2_id) REFERENCES players(id) ON DELETE SET NULL,
  FOREIGN KEY (winner_id) REFERENCES players(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS match_progressions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  next_match_id INTEGER NOT NULL,
  next_player_position INTEGER NOT NULL,
  FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
  FOREIGN KEY (next_match_id) REFERENCES matches(id) ON DELETE CASCADE
);
`;

export const initializeDatabase = () => {
  CREATE_TABLES_SQL.trim()
    .split(";\n\n")
    .filter(Boolean)
    .forEach((statement) => {
      db.prepare(`${statement};`).run();
    });
};
