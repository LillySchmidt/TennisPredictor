import playerEloData from "@/assets/player-elo.json";
import {
  MatchData,
  TournamentData,
  TournamentFormPayload,
  TournamentPlayer,
} from "@/types/tournament";

interface PlayerElo {
  rank: number;
  name: string;
  elo: number;
}

const ROUND_NAMES = ["Round 1", "Round 2", "Quarter-Final", "Semi-Final", "Final"];

const shuffle = <T,>(items: T[]): T[] => {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
};

const pickRandomTopPlayers = (count: number) => {
  const top100 = (playerEloData as PlayerElo[]).filter(
    (player) => typeof player.rank === "number" && player.rank <= 100
  );

  return shuffle(top100).slice(0, count);
};

export const generateRandomTournamentPayload = (surface: string = "Hard") => {
  const selectedPlayers = pickRandomTopPlayers(32);
  const seededOrder = [...selectedPlayers].sort((a, b) => a.rank - b.rank);

  const payload: TournamentFormPayload = {
    name: "Random Top 100 ELO Draw",
    nation: "INT",
    surface,
    drawSize: 32,
    players: seededOrder.map((player) => ({
      name: player.name,
      isBye: false,
    })),
  };

  return { payload, selectedPlayers };
};

export const generateRandomTournament = (surface: string = "Hard") => {
  const { selectedPlayers } = generateRandomTournamentPayload(surface);

  // Seed the draw using the relative ELO ranking of the selected players
  const seededOrder = [...selectedPlayers].sort((a, b) => a.rank - b.rank);
  const seedMap = new Map<string, number>(
    seededOrder.map((player, index) => [player.name, index + 1])
  );

  const tournamentId = Date.now();

  const players: TournamentPlayer[] = selectedPlayers.map((player, index) => ({
    id: index + 1,
    name: player.name,
    isBye: false,
    seedPosition: seedMap.get(player.name) ?? index + 1,
  }));

  const matches: MatchData[] = [];
  let matchId = 1;

  ROUND_NAMES.forEach((roundName, roundIndex) => {
    const roundNumber = roundIndex + 1;
    const matchesInRound = Math.max(1, Math.floor(players.length / Math.pow(2, roundNumber)));

    for (let i = 0; i < matchesInRound; i += 1) {
      const player1 = roundNumber === 1 ? players[i * 2] ?? null : null;
      const player2 = roundNumber === 1 ? players[i * 2 + 1] ?? null : null;

      matches.push({
        id: matchId,
        tournamentId,
        roundNumber,
        roundName,
        matchOrder: i,
        player1,
        player2,
        winnerSlot: null,
      });

      matchId += 1;
    }
  });

  const tournament: TournamentData = {
    id: tournamentId,
    name: "Random Top 100 ELO Draw",
    nation: "INT",
    surface,
    drawSize: players.length,
    players,
  };

  return { tournament, matches };
};
