import { db, runInTransaction } from "../db/database";
import {
  autoAdvanceByes,
  createBracket,
  ensureMatchBelongsToTournament,
  getMatchesWithPlayers,
  MatchDTO,
  clearMatchWinner,
  recordMatchWinner,
} from "./matchService";

export interface TournamentPlayerInput {
  name: string;
  isBye: boolean;
}

export interface TournamentPayload {
  name: string;
  nation: string;
  surface: string;
  drawSize: number;
  players: TournamentPlayerInput[];
}

export interface TournamentRecord {
  id: number;
  name: string;
  nation: string;
  surface: string;
  drawSize: number;
  createdAt: string;
}

export interface PlayerRecord {
  id: number;
  name: string;
  isBye: boolean;
  seedPosition: number;
}

interface TournamentRow {
  id: number;
  name: string;
  nation: string;
  surface: string;
  draw_size: number;
  created_at: string;
}

interface PlayerRow {
  id: number;
  name: string;
  is_bye: number;
  seed_position: number;
}

const isPowerOfTwo = (value: number) => value > 0 && (value & (value - 1)) === 0;

export const listTournaments = (): TournamentRecord[] => {
  const rows = db
    .prepare(`SELECT id, name, nation, surface, draw_size, created_at FROM tournaments ORDER BY created_at DESC`)
    .all() as TournamentRow[];

  return rows.map((row) => ({
    id: row.id,
    name: row.name,
    nation: row.nation,
    surface: row.surface,
    drawSize: row.draw_size,
    createdAt: row.created_at,
  }));
};

export const getTournamentById = (id: number): TournamentRecord | null => {
  const row = db
    .prepare(`SELECT id, name, nation, surface, draw_size, created_at FROM tournaments WHERE id = ?`)
    .get(id) as TournamentRow | undefined;

  if (!row) return null;

  return {
    id: row.id,
    name: row.name,
    nation: row.nation,
    surface: row.surface,
    drawSize: row.draw_size,
    createdAt: row.created_at,
  };
};

const getPlayersForTournament = (tournamentId: number): PlayerRecord[] => {
  const players = db
    .prepare(`
      SELECT id, name, is_bye, seed_position
      FROM players
      WHERE tournament_id = ?
      ORDER BY seed_position ASC
    `)
    .all(tournamentId) as PlayerRow[];

  return players.map((player) => ({
    id: player.id,
    name: player.name,
    isBye: Boolean(player.is_bye),
    seedPosition: player.seed_position,
  }));
};

export const createTournament = (payload: TournamentPayload): { tournamentId: number } => {
  if (!isPowerOfTwo(payload.drawSize)) {
    throw new Error("Draw size must be a power of two");
  }

  if (payload.players.length !== payload.drawSize) {
    throw new Error("Player count must match draw size");
  }

  return runInTransaction(() => {
    const tournamentResult = db
      .prepare(`
        INSERT INTO tournaments (name, nation, surface, draw_size)
        VALUES (?, ?, ?, ?)
      `)
      .run(payload.name, payload.nation, payload.surface, payload.drawSize);

    const tournamentId = Number(tournamentResult.lastInsertRowid);

    const insertPlayer = db.prepare(`
      INSERT INTO players (tournament_id, name, is_bye, seed_position)
      VALUES (?, ?, ?, ?)
    `);

    const playerIds = payload.players.map((player, index) => {
      const name = player.name.trim() || `Player ${index + 1}`;
      const result = insertPlayer.run(tournamentId, name, player.isBye ? 1 : 0, index + 1);
      return Number(result.lastInsertRowid);
    });

    createBracket(tournamentId, playerIds, payload.drawSize);
    autoAdvanceByes(tournamentId);

    return { tournamentId };
  });
};

export const getTournamentDetails = (tournamentId: number) => {
  const tournament = getTournamentById(tournamentId);
  if (!tournament) {
    throw new Error("Tournament not found");
  }

  const players = getPlayersForTournament(tournamentId);
  const matches = getMatchesWithPlayers(tournamentId);
  return { tournament, players, matches };
};

export const getTournamentMatches = (tournamentId: number): MatchDTO[] => {
  return getMatchesWithPlayers(tournamentId);
};

export const assignWinnerBySlot = (
  tournamentId: number,
  matchId: number,
  winnerSlot: 1 | 2
) => {
  return runInTransaction(() => {
    const match = ensureMatchBelongsToTournament(matchId, tournamentId);
    const targetPlayerId = winnerSlot === 1 ? match.player1_id : match.player2_id;

    if (!targetPlayerId) {
      throw new Error("Selected player slot is empty");
    }

    const isBye = winnerSlot === 1 ? Boolean(match.player1_is_bye) : Boolean(match.player2_is_bye);
    if (isBye) {
      throw new Error("Cannot select a bye as winner");
    }

    recordMatchWinner(matchId, targetPlayerId);
    return getMatchesWithPlayers(tournamentId);
  });
};

export const resetMatchWinner = (tournamentId: number, matchId: number) => {
  return runInTransaction(() => {
    ensureMatchBelongsToTournament(matchId, tournamentId);
    clearMatchWinner(matchId);
    return getMatchesWithPlayers(tournamentId);
  });
};
