import { db } from "../db/database";

export interface MatchPlayerDTO {
  id: number;
  name: string;
  isBye: boolean;
  seedPosition?: number;
}

export interface MatchDTO {
  id: number;
  tournamentId: number;
  roundNumber: number;
  roundName: string;
  matchOrder: number;
  winnerSlot: 1 | 2 | null;
  player1: MatchPlayerDTO | null;
  player2: MatchPlayerDTO | null;
}

interface MatchRow {
  id: number;
  tournament_id: number;
  round_number: number;
  round_name: string;
  match_order: number;
  winner_id: number | null;
  player1_id: number | null;
  player1_name: string | null;
  player1_is_bye: number | null;
  player1_seed_position: number | null;
  player2_id: number | null;
  player2_name: string | null;
  player2_is_bye: number | null;
  player2_seed_position: number | null;
}

const getRoundName = (roundNumber: number, totalRounds: number) => {
  if (roundNumber === totalRounds) return "Final";
  if (roundNumber === totalRounds - 1) return "Semi-Final";
  if (roundNumber === totalRounds - 2) return "Quarter-Final";
  return `Round ${roundNumber}`;
};

const mapRowToMatch = (row: MatchRow): MatchDTO => {
  const player1 = row.player1_id
    ? { id: row.player1_id, name: row.player1_name ?? "", isBye: Boolean(row.player1_is_bye), seedPosition: row.player1_seed_position ?? undefined }
    : null;
  const player2 = row.player2_id
    ? { id: row.player2_id, name: row.player2_name ?? "", isBye: Boolean(row.player2_is_bye), seedPosition: row.player2_seed_position ?? undefined }
    : null;

  let winnerSlot: 1 | 2 | null = null;
  if (row.winner_id && player1 && row.winner_id === player1.id) {
    winnerSlot = 1;
  } else if (row.winner_id && player2 && row.winner_id === player2.id) {
    winnerSlot = 2;
  }

  return {
    id: row.id,
    tournamentId: row.tournament_id,
    roundNumber: row.round_number,
    roundName: row.round_name,
    matchOrder: row.match_order,
    player1,
    player2,
    winnerSlot,
  };
};

const matchSelectBase = `
  SELECT 
    m.id,
    m.tournament_id,
    m.round_number,
    m.round_name,
    m.match_order,
    m.winner_id,
    p1.id AS player1_id,
    p1.name AS player1_name,
    p1.is_bye AS player1_is_bye,
    p1.seed_position AS player1_seed_position,
    p2.id AS player2_id,
    p2.name AS player2_name,
    p2.is_bye AS player2_is_bye,
    p2.seed_position AS player2_seed_position
  FROM matches m
  LEFT JOIN players p1 ON p1.id = m.player1_id
  LEFT JOIN players p2 ON p2.id = m.player2_id
`;

export const getMatchesWithPlayers = (tournamentId: number): MatchDTO[] => {
  const rows = db
    .prepare(`${matchSelectBase} WHERE m.tournament_id = ? ORDER BY m.round_number, m.match_order`)
    .all(tournamentId) as MatchRow[];
  return rows.map(mapRowToMatch);
};

const getMatchById = (matchId: number): MatchRow | undefined => {
  const row = db
    .prepare(`${matchSelectBase} WHERE m.id = ?`)
    .get(matchId) as MatchRow | undefined;
  return row;
};

const getProgression = (matchId: number) => {
  return db
    .prepare(`SELECT next_match_id, next_player_position FROM match_progressions WHERE match_id = ?`)
    .get(matchId) as { next_match_id: number; next_player_position: number } | undefined;
};

export const createBracket = (tournamentId: number, playerIds: number[], drawSize: number) => {
  if (!Number.isInteger(Math.log2(drawSize))) {
    throw new Error("Draw size must be a power of two");
  }

  const totalRounds = Math.log2(drawSize);
  const insertMatch = db.prepare(`
    INSERT INTO matches (tournament_id, round_number, round_name, match_order, player1_id, player2_id, winner_id)
    VALUES (?, ?, ?, ?, ?, ?, NULL)
  `);
  const insertProgression = db.prepare(`
    INSERT INTO match_progressions (match_id, next_match_id, next_player_position)
    VALUES (?, ?, ?)
  `);

  const matchesByRound: number[][] = [];
  let playerIndex = 0;

  for (let roundNumber = 1; roundNumber <= totalRounds; roundNumber++) {
    const matchCount = drawSize / Math.pow(2, roundNumber);
    const roundMatches: number[] = [];
    const roundName = getRoundName(roundNumber, totalRounds);

    for (let matchOrder = 0; matchOrder < matchCount; matchOrder++) {
      const player1Id = roundNumber === 1 ? playerIds[playerIndex++] ?? null : null;
      const player2Id = roundNumber === 1 ? playerIds[playerIndex++] ?? null : null;

      const result = insertMatch.run(
        tournamentId,
        roundNumber,
        roundName,
        matchOrder,
        player1Id,
        player2Id
      );

      roundMatches.push(Number(result.lastInsertRowid));
    }

    matchesByRound.push(roundMatches);
  }

  matchesByRound.forEach((currentRound, roundIndex) => {
    const nextRound = matchesByRound[roundIndex + 1];
    if (!nextRound) return;

    currentRound.forEach((matchId, idx) => {
      const nextMatchId = nextRound[Math.floor(idx / 2)];
      const nextPlayerPosition = (idx % 2) + 1;
      insertProgression.run(matchId, nextMatchId, nextPlayerPosition);
    });
  });
};

const tryResolveBye = (matchId: number): void => {
  const match = getMatchById(matchId);
  if (!match || match.winner_id) return;

  const player1Bye = match.player1_id ? Boolean(match.player1_is_bye) : false;
  const player2Bye = match.player2_id ? Boolean(match.player2_is_bye) : false;

  if (player1Bye && !player2Bye && match.player2_id) {
    recordMatchWinner(match.id, match.player2_id);
  } else if (player2Bye && !player1Bye && match.player1_id) {
    recordMatchWinner(match.id, match.player1_id);
  }
};

const advanceWinner = (matchId: number, winnerPlayerId: number) => {
  const progression = getProgression(matchId);

  if (!progression) return;

  const column = progression.next_player_position === 1 ? "player1_id" : "player2_id";
  db.prepare(`UPDATE matches SET ${column} = ? WHERE id = ?`).run(winnerPlayerId, progression.next_match_id);
  tryResolveBye(progression.next_match_id);
};

export const recordMatchWinner = (matchId: number, winnerPlayerId: number) => {
  db.prepare(`UPDATE matches SET winner_id = ? WHERE id = ?`).run(winnerPlayerId, matchId);
  advanceWinner(matchId, winnerPlayerId);
};

const removeAdvancement = (matchId: number, playerId: number) => {
  const progression = getProgression(matchId);
  if (!progression) return;

  const column = progression.next_player_position === 1 ? "player1_id" : "player2_id";
  const nextMatch = getMatchById(progression.next_match_id);

  if (nextMatch && nextMatch[column as "player1_id" | "player2_id"] === playerId) {
    db.prepare(`UPDATE matches SET ${column} = NULL WHERE id = ?`).run(progression.next_match_id);

    // If the downstream match winner was this player, clear it and continue cascading.
    if (nextMatch.winner_id === playerId) {
      db.prepare(`UPDATE matches SET winner_id = NULL WHERE id = ?`).run(progression.next_match_id);
      removeAdvancement(progression.next_match_id, playerId);
    }
  }
};

export const clearMatchWinner = (matchId: number) => {
  const match = getMatchById(matchId);
  if (!match || !match.winner_id) return;

  const winnerId = match.winner_id;
  db.prepare(`UPDATE matches SET winner_id = NULL WHERE id = ?`).run(matchId);
  removeAdvancement(matchId, winnerId);
};

export const autoAdvanceByes = (tournamentId: number) => {
  const matchIds = db
    .prepare(`SELECT id FROM matches WHERE tournament_id = ? ORDER BY round_number, match_order`)
    .all(tournamentId) as { id: number }[];

  matchIds.forEach(({ id }) => tryResolveBye(id));
};

export const ensureMatchBelongsToTournament = (matchId: number, tournamentId: number): MatchRow => {
  const match = db
    .prepare(`${matchSelectBase} WHERE m.id = ? AND m.tournament_id = ?`)
    .get(matchId, tournamentId) as MatchRow | undefined;

  if (!match) {
    throw new Error("Match not found for tournament");
  }

  return match;
};
