import { MatchCard } from "./MatchCard";
import { MatchData, TournamentData } from "@/types/tournament";
import playerEloData from "@/assets/player-elo.json";

interface PlayerElo {
  rank: number;
  name: string;
  nameReversed: string;
  elo: number;
  peakElo: number;
  wins: number;
  losses: number;
  winRate: number;
  hardElo: number;
  clayElo: number;
  grassElo: number;
}

const normalizeName = (name: string): string => name.toLowerCase().replace(/[^a-z\s]/g, "").trim();

const getPlayerElo = (name: string): PlayerElo | null => {
  const normalizedInput = normalizeName(name);
  return (
    (playerEloData as PlayerElo[]).find((p) => {
      const normalizedName = normalizeName(p.name);
      const normalizedReversed = normalizeName(p.nameReversed);
      return normalizedName === normalizedInput || normalizedReversed === normalizedInput;
    }) || null
  );
};

const expectedScore = (playerElo: number, opponentElo: number): number => {
  return 1.0 / (1.0 + Math.pow(10, (opponentElo - playerElo) / 400.0));
};

const predictMatch = (player1Name: string, player2Name: string) => {
  const p1 = getPlayerElo(player1Name);
  const p2 = getPlayerElo(player2Name);

  const baseElo = 1500;
  const p1Elo = p1?.elo ?? baseElo;
  const p2Elo = p2?.elo ?? baseElo;

  const p1Prob = Math.round(expectedScore(p1Elo, p2Elo) * 100);
  const p2Prob = 100 - p1Prob;

  return {
    winner: p1Prob >= p2Prob ? 1 : 2,
    confidence: { player1: p1Prob, player2: p2Prob },
  };
};

interface TournamentBracketProps {
  tournament: TournamentData | null;
  matches: MatchData[];
}

export const TournamentBracket = ({ tournament, matches }: TournamentBracketProps) => {
  // Defensive: API failures or unexpected payloads could leave matches undefined/null.
  const safeMatches = Array.isArray(matches) ? matches : [];

  const displayMatches =
    tournament && safeMatches.length
      ? safeMatches
          .filter((m) => m.roundNumber === 1) // show first round on Upcoming tab
          .filter((m) => m.player1 && m.player2)
          .map((m) => {
            const prediction = predictMatch(m.player1?.name ?? "", m.player2?.name ?? "");
            return {
              player1: {
                name: m.player1?.name ?? "TBD",
                country: m.player1?.seedPosition ? `Seed ${m.player1.seedPosition}` : "N/A",
                seed: m.player1?.seedPosition,
                confidence: prediction.confidence.player1,
              },
              player2: {
                name: m.player2?.name ?? "TBD",
                country: m.player2?.seedPosition ? `Seed ${m.player2.seedPosition}` : "N/A",
                seed: m.player2?.seedPosition,
                confidence: prediction.confidence.player2,
              },
              round: m.roundName,
              status: m.winnerSlot ? "completed" : "upcoming",
              predictedWinner: prediction.winner,
            };
          })
      : [];

  return (
    <div className="space-y-8">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">
          {tournament ? `${tournament.name} — Upcoming` : "Tournament Predictions"}
        </h2>
        <p className="text-muted-foreground">
          {tournament
            ? "Preview the first-round matchups from the active tournament."
            : "ELO-based confidence analysis for upcoming matches"}
        </p>
      </div>

      {displayMatches.length ? (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
          {displayMatches.map((match, idx) => (
            <MatchCard key={idx} {...match} />
          ))}
        </div>
      ) : (
        <div className="text-center text-muted-foreground">
          No tournament loaded yet. Import or select a tournament to see matchups.
        </div>
      )}
    </div>
  );
};
