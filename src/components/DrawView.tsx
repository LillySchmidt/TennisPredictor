import { useEffect, useMemo, useState } from "react";
import { isAxiosError } from "axios";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { clearMatchWinner, fetchTournamentMatches, submitMatchWinner } from "@/lib/api";
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

interface DrawViewProps {
  tournament: TournamentData | null;
  matches: MatchData[];
  setMatches: (matches: MatchData[]) => void;
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

const getErrorMessage = (error: unknown) => {
  if (isAxiosError<{ message?: string }>(error)) {
    return error.response?.data?.message ?? error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong";
};

export const DrawView = ({ tournament, matches, setMatches }: DrawViewProps) => {
  const [activeMatchId, setActiveMatchId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isFetching, setIsFetching] = useState(false);
  const [isClearingId, setIsClearingId] = useState<number | null>(null);

  const surface = tournament?.surface || 'Hard';

  useEffect(() => {
    const loadMatches = async () => {
      if (!tournament?.id || matches.length > 0) return;
      setIsFetching(true);
      setError(null);
      try {
        const data = await fetchTournamentMatches(tournament.id);
        setMatches(data);
      } catch (err) {
        setError(getErrorMessage(err));
      } finally {
        setIsFetching(false);
      }
    };

    loadMatches();
  }, [tournament?.id, matches.length, setMatches]);

  const roundGroups = useMemo(() => {
    const groups = matches.reduce<Record<string, MatchData[]>>((acc, match) => {
      if (!acc[match.roundName]) {
        acc[match.roundName] = [];
      }
      acc[match.roundName].push(match);
      return acc;
    }, {});

    return Object.entries(groups).sort((a, b) => {
      const roundA = a[1][0]?.roundNumber ?? 0;
      const roundB = b[1][0]?.roundNumber ?? 0;
      return roundA - roundB;
    });
  }, [matches]);

  const handleSelectWinner = async (matchId: number, slot: 1 | 2) => {
    if (!tournament?.id) return;
    setError(null);
    setActiveMatchId(matchId);
    try {
      const updatedMatches = await submitMatchWinner(tournament.id, matchId, slot);
      setMatches(updatedMatches);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setActiveMatchId(null);
    }
  };

  const handleClearWinner = async (matchId: number) => {
    if (!tournament?.id) return;
    setError(null);
    setIsClearingId(matchId);
    try {
      const updatedMatches = await clearMatchWinner(tournament.id, matchId);
      setMatches(updatedMatches);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsClearingId(null);
    }
  };

  if (!tournament) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Please import a tournament first</p>
      </div>
    );
  }

  if (isFetching) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Loading matches...</p>
      </div>
    );
  }

  if (!matches.length) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No matches available yet for this tournament.</p>
      </div>
    );
  }

  const renderPlayerButton = (match: MatchData, slot: 1 | 2) => {
    const player = slot === 1 ? match.player1 : match.player2;
    const otherPlayer = slot === 1 ? match.player2 : match.player1;
    
    if (!player) {
      return (
        <div className="w-full p-2 rounded border border-dashed text-center text-sm text-muted-foreground">
          TBD
        </div>
      );
    }

    const isWinner = match.winnerSlot === slot;
    const isDisabled = player.isBye || activeMatchId === match.id || isClearingId === match.id;

    // Get ELO data
    const playerEloData = getPlayerElo(player.name);
    const playerElo = getSurfaceElo(playerEloData, surface);
    
    // Calculate win probability if both players are known
    let winProb: number | null = null;
    if (otherPlayer && !otherPlayer.isBye) {
      const otherEloData = getPlayerElo(otherPlayer.name);
      const otherElo = getSurfaceElo(otherEloData, surface);
      winProb = Math.round(expectedScore(playerElo, otherElo) * 100);
    }

    return (
      <button
        onClick={() => handleSelectWinner(match.id, slot)}
        disabled={isDisabled}
        className={`w-full p-3 rounded border text-left transition-all ${
          isWinner ? "bg-primary/10 border-primary text-primary font-semibold" : "border-border hover:bg-muted"
        } ${isDisabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex-1 min-w-0">
            <span className="block truncate">{player.name}</span>
            <span className="text-xs text-muted-foreground">
              ELO: {Math.round(playerElo)}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {player.isBye && <span className="text-xs text-muted-foreground">BYE</span>}
            {winProb !== null && !player.isBye && (
              <Badge variant={winProb >= 50 ? "default" : "outline"} className="text-xs">
                {winProb}%
              </Badge>
            )}
          </div>
        </div>
      </button>
    );
  };

  return (
    <div className="space-y-8">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">{tournament.name}</h2>
        <p className="text-muted-foreground">
          {surface} Court • Click on a player to select them as the winner
        </p>
      </div>

      {error && <div className="text-sm text-destructive text-center">{error}</div>}

      <div className="flex gap-4 overflow-x-auto pb-4">
        {roundGroups.map(([round, roundMatches]) => (
          <div key={round} className="space-y-4 min-w-[280px]">
            <h3 className="text-lg font-semibold text-center sticky top-0 bg-background py-2">{round}</h3>
            {roundMatches.map((match) => (
              <Card key={match.id} className="p-4 space-y-2">
                <div className="text-xs text-muted-foreground font-medium mb-3">
                  {round} • Match {match.matchOrder + 1}
                </div>
                {renderPlayerButton(match, 1)}
                <div className="text-center text-xs text-muted-foreground">vs</div>
                {renderPlayerButton(match, 2)}
                {match.winnerSlot && (
                  <div className="flex justify-end">
                    <button
                      onClick={() => handleClearWinner(match.id)}
                      disabled={isClearingId === match.id}
                      className="text-xs text-muted-foreground hover:text-primary disabled:opacity-50"
                    >
                      Undo winner
                    </button>
                  </div>
                )}
                {!match.player1 && !match.player2 && (
                  <div className="text-center text-muted-foreground text-sm py-4">Awaiting players</div>
                )}
              </Card>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
