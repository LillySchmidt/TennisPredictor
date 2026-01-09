import { useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TournamentData } from "@/types/tournament";
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

interface InitialDrawViewProps {
  tournament: TournamentData | null;
}

// Normalize name for matching
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
  return player.elo;
};

// ELO-based expected score calculation
const expectedScore = (playerElo: number, opponentElo: number): number => {
  return 1.0 / (1.0 + Math.pow(10, (opponentElo - playerElo) / 400.0));
};

const predictMatch = (player1Name: string, player2Name: string, surface: string = 'Hard') => {
  const p1Data = getPlayerElo(player1Name);
  const p2Data = getPlayerElo(player2Name);

  const p1Elo = getSurfaceElo(p1Data, surface);
  const p2Elo = getSurfaceElo(p2Data, surface);

  const p1WinProb = expectedScore(p1Elo, p2Elo);
  const p1Confidence = Math.round(p1WinProb * 100);
  const p2Confidence = 100 - p1Confidence;

  return {
    winner: p1Confidence >= p2Confidence ? 1 : 2,
    confidence: { player1: p1Confidence, player2: p2Confidence },
    player1Elo: Math.round(p1Elo),
    player2Elo: Math.round(p2Elo),
  };
};

export const InitialDrawView = ({ tournament }: InitialDrawViewProps) => {
  const surface = tournament?.surface || 'Hard';

  const { matches, stats } = useMemo(() => {
    if (!tournament) return { matches: [], stats: { total: 0, predicted: 0, percentage: 0 } };

    const firstRoundMatches: {
      id: string;
      player1: TournamentData["players"][number] | null;
      player2: TournamentData["players"][number] | null;
      round: string;
      winner: number;
      confidence?: { player1: number; player2: number };
      player1Elo?: number;
      player2Elo?: number;
    }[] = [];
    const { players } = tournament;

    for (let i = 0; i < players.length; i += 2) {
      const p1 = players[i];
      const p2 = players[i + 1];

      if (p1.isBye || p2.isBye) {
        firstRoundMatches.push({
          id: `r1-${i / 2}`,
          player1: p1.isBye ? null : p1,
          player2: p2.isBye ? null : p2,
          round: "Round 1",
          winner: p1.isBye ? 2 : 1,
        });
      } else {
        const prediction = predictMatch(p1.name, p2.name, surface);
        firstRoundMatches.push({
          id: `r1-${i / 2}`,
          player1: p1,
          player2: p2,
          round: "Round 1",
          winner: prediction.winner,
          confidence: prediction.confidence,
          player1Elo: prediction.player1Elo,
          player2Elo: prediction.player2Elo,
        });
      }
    }

    return {
      matches: firstRoundMatches,
      stats: {
        total: firstRoundMatches.length,
        predicted: firstRoundMatches.length,
        percentage: 100,
      },
    };
  }, [tournament, surface]);

  if (!tournament) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Please import a tournament first</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">{tournament.name} - Initial Predictions</h2>
        <p className="text-muted-foreground">
          ELO-based predictions using {surface} court ratings (first round only)
        </p>
      </div>

      <Card className="p-6 max-w-md mx-auto">
        <div className="text-center space-y-2">
          <h3 className="text-lg font-semibold">Prediction Statistics</h3>
          <div className="flex items-center justify-center gap-4">
            <div>
              <div className="text-3xl font-bold text-primary">{stats.predicted}</div>
              <div className="text-xs text-muted-foreground">Predicted</div>
            </div>
            <div className="text-2xl text-muted-foreground">/</div>
            <div>
              <div className="text-3xl font-bold">{stats.total}</div>
              <div className="text-xs text-muted-foreground">Total Matches</div>
            </div>
            <div className="text-2xl text-muted-foreground">=</div>
            <div>
              <div className="text-3xl font-bold text-primary">{stats.percentage}%</div>
              <div className="text-xs text-muted-foreground">Coverage</div>
            </div>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {matches.map((match) => (
          <Card key={match.id} className="p-4 space-y-3">
            <div className="text-xs text-muted-foreground font-medium">{match.round}</div>

            {match.player1 ? (
              <div
                className={`p-3 rounded border ${
                  match.winner === 1 ? "bg-primary/10 border-primary" : "border-border"
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
                    <Badge variant={match.winner === 1 ? "default" : "outline"} className="shrink-0">
                      {match.confidence.player1}%
                    </Badge>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-3 rounded border border-border bg-muted text-muted-foreground text-center">
                BYE
              </div>
            )}

            {match.player2 ? (
              <div
                className={`p-3 rounded border ${
                  match.winner === 2 ? "bg-primary/10 border-primary" : "border-border"
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
                    <Badge variant={match.winner === 2 ? "default" : "outline"} className="shrink-0">
                      {match.confidence.player2}%
                    </Badge>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-3 rounded border border-border bg-muted text-muted-foreground text-center">
                BYE
              </div>
            )}
          </Card>
        ))}
      </div>
    </div>
  );
};
