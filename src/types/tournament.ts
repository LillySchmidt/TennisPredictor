export interface TournamentPlayer {
  id?: number;
  name: string;
  isBye: boolean;
  seedPosition?: number;
}

export interface TournamentData {
  id: number;
  name: string;
  nation: string;
  surface: string;
  drawSize: number;
  players: TournamentPlayer[];
}

export interface TournamentFormPayload {
  name: string;
  nation: string;
  surface: string;
  drawSize: number;
  players: TournamentPlayer[];
}

export interface MatchData {
  id: number;
  tournamentId: number;
  roundNumber: number;
  roundName: string;
  matchOrder: number;
  player1: TournamentPlayer | null;
  player2: TournamentPlayer | null;
  winnerSlot: 1 | 2 | null;
}

export interface TournamentDetailsResponse {
  tournament: TournamentData;
  players: TournamentPlayer[];
  matches: MatchData[];
}
