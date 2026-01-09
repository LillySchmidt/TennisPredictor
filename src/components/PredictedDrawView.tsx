import { useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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

type PredictedMatch = MatchData & {
  predictedWinner?: 1 | 2;
  confidence?: { player1: number; player2: number };
  isCorrect?: boolean;
  player1Elo?: number;
  player2Elo?: number;
};

interface PredictedDrawViewProps {
  tournament: TournamentData | null;
  manualMatches: MatchData[];
}

// Normalize name for matching (handle "FirstName LastName" vs "LastName FirstName")
const normalizeName = (name: string): string => {
  return name.toLowerCase().replace(/[^a-z\s]/g, '').trim();
};

const getPlayerElo = (name: string): PlayerElo | null => {
  const normalizedInput = normalizeName(name);
  
  return (playerEloData as PlayerElo[]).find((p) => {
    const normalizedName = normalizeName(p.name);
    const normalizedReversed = normalizeName(p.nameReversed);
    return normalizedName === normalizedInput || normalizedReversed === normalizedInput;
  }) || null;
};

const getSurfaceElo = (player: PlayerElo | null, surface: string): number => {
  if (!player) return 1500;
  
  const surfaceLower = surface.toLowerCase();
  if (surfaceLower.includes('hard')) return player.hardElo;
  if (surfaceLower.includes('clay')) return player.clayElo;
  if (surfaceLower.includes('grass')) return player.grassElo;
  return player.elo; // Default to overall ELO
};

// ELO-based expected score calculation
const expectedScore = (playerElo: number, opponentElo: number): number => {
  return 1.0 / (1.0 + Math.pow(10, (opponentElo - playerElo) / 400.0));
};

const predictMatch = (player1Name: string, player2Name: string, surface: string = 'Hard') => {
  const p1Data = getPlayerElo(player1Name);
  const p2Data = getPlayerElo(player2Name);

  // Get surface-specific ELO or default to 1500 for unknown players
  const p1Elo = getSurfaceElo(p1Data, surface);
  const p2Elo = getSurfaceElo(p2Data, surface);

  // Calculate expected win probability using ELO formula
  const p1WinProb = expectedScore(p1Elo, p2Elo);
  const p1Confidence = Math.round(p1WinProb * 100);
  const p2Confidence = 100 - p1Confidence;

  return {
    winner: (p1Confidence >= p2Confidence ? 1 : 2) as 1 | 2,
    confidence: { player1: p1Confidence, player2: p2Confidence },
    player1Elo: Math.round(p1Elo),
    player2Elo: Math.round(p2Elo),
  };
};

export const PredictedDrawView = ({ tournament, manualMatches }: PredictedDrawViewProps) => {
  const { predictedMatches, stats } = useMemo(() => {
    if (!tournament || manualMatches.length === 0) {
      return {
        predictedMatches: [] as PredictedMatch[],
        stats: { total: 0, predicted: 0, correct: 0, percentage: 0 },
      };
    }

    const surface = tournament.surface || 'Hard';

    const matches = manualMatches.map<PredictedMatch>((match) => {
      if (!match.player1 || !match.player2) {
        return { ...match };
      }

      const prediction = predictMatch(match.player1.name, match.player2.name, surface);
      const winnerSlot = match.winnerSlot ?? undefined;
      const isCorrect = winnerSlot ? winnerSlot === prediction.winner : undefined;

      return {
        ...match,
        predictedWinner: prediction.winner,
        confidence: prediction.confidence,
        isCorrect,
        player1Elo: prediction.player1Elo,
        player2Elo: prediction.player2Elo,
      };
    });

    const matchesWithWinners = matches.filter((m) => m.winnerSlot !== null && m.winnerSlot !== undefined);
    const correctPredictions = matches.filter((m) => m.isCorrect === true).length;

    return {
      predictedMatches: matches,
      stats: {
        total: matchesWithWinners.length,
        predicted: matchesWithWinners.length,
        correct: correctPredictions,
        percentage:
          matchesWithWinners.length > 0
            ? Math.round((correctPredictions / matchesWithWinners.length) * 100)
            : 0,
      },
    };
  }, [tournament, manualMatches]);

  if (!tournament) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Please import a tournament first</p>
      </div>
    );
  }

  if (manualMatches.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No matches to predict yet. Go to the Draw tab to see matches.</p>
      </div>
    );
  }

  const roundGroups = predictedMatches.reduce<Record<string, PredictedMatch[]>>((acc, match) => {
    if (!acc[match.roundName]) acc[match.roundName] = [];
    acc[match.roundName].push(match);
    return acc;
  }, {});

  return (
    <div className="space-y-8">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">{tournament.name} - Predicted Draw</h2>
        <p className="text-muted-foreground">
          ELO-based predictions using {tournament.surface || 'Hard'} court ratings
        </p>
      </div>

      <Card className="p-6 max-w-md mx-auto">
        <div className="text-center space-y-2">
          <h3 className="text-lg font-semibold">Prediction Statistics</h3>
          <div className="flex items-center justify-center gap-4">
            <div>
              <div className="text-3xl font-bold text-primary">{stats.correct}</div>
              <div className="text-xs text-muted-foreground">Correct</div>
            </div>
            <div className="text-2xl text-muted-foreground">/</div>
            <div>
              <div className="text-3xl font-bold">{stats.total}</div>
              <div className="text-xs text-muted-foreground">Completed</div>
            </div>
            <div className="text-2xl text-muted-foreground">=</div>
            <div>
              <div className="text-3xl font-bold text-primary">{stats.percentage}%</div>
              <div className="text-xs text-muted-foreground">Accuracy</div>
            </div>
          </div>
        </div>
      </Card>

      <div className="flex gap-4 overflow-x-auto pb-4">
        {Object.entries(roundGroups).map(([round, roundMatches]) => (
          <div key={round} className="space-y-4 min-w-[320px]">
            <h3 className="text-lg font-semibold text-center sticky top-0 bg-background py-2">
              {round}
            </h3>
            {roundMatches.map((match) => (
              <Card key={match.id} className="p-4 space-y-3">
                {match.player1 ? (
                  <div
                    className={`p-3 rounded border relative ${
                      match.winnerSlot === 1
                        ? "bg-primary/10 border-primary"
                        : match.predictedWinner === 1
                        ? "bg-muted/50 border-primary/30"
                        : "border-border"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <span className="font-medium block truncate">{match.player1.name}</span>
                        {match.player1Elo && (
                          <span className="text-xs text-muted-foreground">
                            ELO: {match.player1Elo}
                          </span>
                        )}
                      </div>
                      {match.confidence && (
                        <Badge 
                          variant={match.predictedWinner === 1 ? "default" : "outline"}
                          className="shrink-0"
                        >
                          {match.confidence.player1}%
                        </Badge>
                      )}
                    </div>
                    {match.winnerSlot === 1 && match.isCorrect === true && (
                      <div className="absolute -top-2 -right-2 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center text-xs text-white">
                        ✓
                      </div>
                    )}
                    {match.winnerSlot === 1 && match.isCorrect === false && (
                      <div className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-xs text-white">
                        ✗
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-3 rounded border border-border bg-muted text-muted-foreground text-center text-sm">
                    TBD
                  </div>
                )}

                {match.player2 ? (
                  <div
                    className={`p-3 rounded border relative ${
                      match.winnerSlot === 2
                        ? "bg-primary/10 border-primary"
                        : match.predictedWinner === 2
                        ? "bg-muted/50 border-primary/30"
                        : "border-border"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <span className="font-medium block truncate">{match.player2.name}</span>
                        {match.player2Elo && (
                          <span className="text-xs text-muted-foreground">
                            ELO: {match.player2Elo}
                          </span>
                        )}
                      </div>
                      {match.confidence && (
                        <Badge 
                          variant={match.predictedWinner === 2 ? "default" : "outline"}
                          className="shrink-0"
                        >
                          {match.confidence.player2}%
                        </Badge>
                      )}
                    </div>
                    {match.winnerSlot === 2 && match.isCorrect === true && (
                      <div className="absolute -top-2 -right-2 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center text-xs text-white">
                        ✓
                      </div>
                    )}
                    {match.winnerSlot === 2 && match.isCorrect === false && (
                      <div className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-xs text-white">
                        ✗
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-3 rounded border border-border bg-muted text-muted-foreground text-center text-sm">
                    TBD
                  </div>
                )}
              </Card>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
