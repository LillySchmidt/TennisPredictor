import { useMemo, useState } from "react";
import { isAxiosError } from "axios";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import playerRankings from "@/assets/player-rankings.json";
import { TournamentDetailsResponse, TournamentPlayer } from "@/types/tournament";
import { createTournamentRequest } from "@/lib/api";

interface PlayerRanking {
  rank: number;
  name: string;
  country: string;
  points: number;
}

interface ImportTournamentProps {
  onTournamentCreate: (details: TournamentDetailsResponse) => void;
}

const getErrorMessage = (error: unknown) => {
  if (isAxiosError<{ message?: string }>(error)) {
    return error.response?.data?.message ?? error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Failed to create tournament";
};

export const ImportTournament = ({ onTournamentCreate }: ImportTournamentProps) => {
  const [tournamentName, setTournamentName] = useState("");
  const [nation, setNation] = useState("");
  const [surface, setSurface] = useState("");
  const [drawSize, setDrawSize] = useState<number>(32);
  const [players, setPlayers] = useState<TournamentPlayer[]>(
    Array(32)
      .fill(null)
      .map(() => ({ name: "", isBye: false }))
  );
  const [suggestions, setSuggestions] = useState<PlayerRanking[]>([]);
  const [activeInput, setActiveInput] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = useMemo(() => {
    return Boolean(tournamentName.trim() && nation.trim() && surface);
  }, [tournamentName, nation, surface]);

  const handleDrawSizeChange = (value: string) => {
    const size = parseInt(value, 10);
    setDrawSize(size);
    setPlayers(
      Array(size)
        .fill(null)
        .map(() => ({ name: "", isBye: false }))
    );
  };

  const handlePlayerInput = (index: number, value: string) => {
    const newPlayers = [...players];
    newPlayers[index] = { ...newPlayers[index], name: value };
    setPlayers(newPlayers);

    if (value.length > 1) {
      const filtered = (playerRankings as PlayerRanking[])
        .filter((player) => player.name.toLowerCase().includes(value.toLowerCase()))
        .slice(0, 5);
      setSuggestions(filtered);
      setActiveInput(index);
    } else {
      setSuggestions([]);
      setActiveInput(null);
    }
  };

  const selectPlayer = (index: number, player: PlayerRanking) => {
    const newPlayers = [...players];
    newPlayers[index] = { name: player.name, isBye: false };
    setPlayers(newPlayers);
    setSuggestions([]);
    setActiveInput(null);
  };

  const toggleBye = (index: number) => {
    const newPlayers = [...players];
    newPlayers[index] = {
      name: newPlayers[index].isBye ? "" : "BYE",
      isBye: !newPlayers[index].isBye,
    };
    setPlayers(newPlayers);
  };

  const handleSubmit = async () => {
    if (!canSubmit) {
      setError("Please fill out tournament name, nation, and surface.");
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      const sanitizedPlayers = players.map((player, index) => ({
        name: player.isBye ? player.name || "BYE" : player.name.trim() || `Player ${index + 1}`,
        isBye: player.isBye,
      }));

      const details = await createTournamentRequest({
        name: tournamentName.trim(),
        nation: nation.trim(),
        surface,
        drawSize,
        players: sanitizedPlayers,
      });

      onTournamentCreate(details);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Import Tournament</h2>
        <p className="text-muted-foreground">Set up your tournament and fill in player details</p>
      </div>

      <Card className="p-6 space-y-6">
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Tournament Details</h3>

          <div className="grid gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="tournament-name">Tournament Name</Label>
              <Input
                id="tournament-name"
                value={tournamentName}
                onChange={(e) => setTournamentName(e.target.value)}
                placeholder="e.g., ATP Masters 1000"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="nation">Nation</Label>
              <Input
                id="nation"
                value={nation}
                onChange={(e) => setNation(e.target.value)}
                placeholder="e.g., USA"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="surface">Surface</Label>
              <Select value={surface} onValueChange={setSurface}>
                <SelectTrigger>
                  <SelectValue placeholder="Select surface" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="clay">Clay</SelectItem>
                  <SelectItem value="hard">Hard</SelectItem>
                  <SelectItem value="grass">Grass</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="draw-size">Draw Size</Label>
              <Select value={drawSize.toString()} onValueChange={handleDrawSizeChange}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="16">16</SelectItem>
                  <SelectItem value="32">32</SelectItem>
                  <SelectItem value="64">64</SelectItem>
                  <SelectItem value="128">128</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-xl font-semibold">Players ({drawSize} Draw)</h3>
          <p className="text-sm text-muted-foreground">Spot 1 plays Spot 2, Spot 3 plays Spot 4, etc.</p>

          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            {players.map((player, index) => (
              <div key={index} className="relative space-y-1">
                <Label htmlFor={`player-${index}`} className="text-xs">
                  Spot {index + 1}
                </Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Input
                      id={`player-${index}`}
                      value={player.isBye ? "BYE" : player.name}
                      onChange={(e) => handlePlayerInput(index, e.target.value)}
                      placeholder="Start typing name..."
                      className={`text-sm ${player.isBye ? "bg-muted text-muted-foreground" : ""}`}
                      disabled={player.isBye}
                    />

                    {activeInput === index && suggestions.length > 0 && !player.isBye && (
                      <Card className="absolute z-50 w-full mt-1 max-h-48 overflow-auto">
                        {suggestions.map((suggestion) => (
                          <button
                            key={suggestion.rank}
                            onClick={() => selectPlayer(index, suggestion)}
                            className="w-full p-2 text-left text-sm hover:bg-muted transition-colors border-b border-border last:border-0"
                          >
                            <div className="font-medium">{suggestion.name}</div>
                            <div className="text-xs text-muted-foreground">
                              #{suggestion.rank} • {suggestion.country} • {suggestion.points} pts
                            </div>
                          </button>
                        ))}
                      </Card>
                    )}
                  </div>

                  <Button
                    type="button"
                    variant={player.isBye ? "default" : "outline"}
                    size="icon"
                    onClick={() => toggleBye(index)}
                    className="shrink-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {error && <div className="text-sm text-destructive text-center">{error}</div>}

        <Button onClick={handleSubmit} className="w-full" size="lg" disabled={isSubmitting}>
          {isSubmitting ? "Generating..." : "Generate Tournament Draw"}
        </Button>
      </Card>
    </div>
  );
};
