import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { TrendingUp } from "lucide-react";

interface Player {
  name: string;
  country: string;
  seed?: number;
  confidence: number;
  elo?: number;
}

interface MatchCardProps {
  player1: Player;
  player2: Player;
  round: string;
  status: "upcoming" | "in_progress" | "completed";
  predictedWinner?: 1 | 2;
}

export const MatchCard = ({
  player1,
  player2,
  round,
  status,
  predictedWinner,
}: MatchCardProps) => {
  const getStatusColor = () => {
    switch (status) {
      case "completed":
        return "bg-success";
      case "in_progress":
        return "bg-accent";
      default:
        return "bg-muted";
    }
  };

  const getStatusText = () => {
    switch (status) {
      case "completed":
        return "Completed";
      case "in_progress":
        return "Live";
      default:
        return "Upcoming";
    }
  };

  return (
    <Card className="overflow-hidden transition-all hover:shadow-lg">
      <div className="flex items-center justify-between border-b border-border bg-muted/30 px-4 py-2">
        <span className="text-sm font-medium text-muted-foreground">{round}</span>
        <Badge variant="secondary" className={getStatusColor()}>
          {getStatusText()}
        </Badge>
      </div>

      <div className="p-4 space-y-3">
        {/* Player 1 */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {player1.seed && (
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                  {player1.seed}
                </span>
              )}
              <div>
                <p className="font-semibold">{player1.name}</p>
                <div className="flex items-center gap-2">
                  <p className="text-xs text-muted-foreground">{player1.country}</p>
                  {player1.elo && (
                    <span className="text-xs text-muted-foreground">
                      • ELO: {player1.elo}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {predictedWinner === 1 && (
                <TrendingUp className="h-4 w-4 text-success" />
              )}
              <span className="text-lg font-bold text-primary">
                {player1.confidence}%
              </span>
            </div>
          </div>
          <Progress value={player1.confidence} className="h-2" />
        </div>

        <div className="flex justify-center">
          <span className="text-xs font-medium text-muted-foreground">VS</span>
        </div>

        {/* Player 2 */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {player2.seed && (
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                  {player2.seed}
                </span>
              )}
              <div>
                <p className="font-semibold">{player2.name}</p>
                <div className="flex items-center gap-2">
                  <p className="text-xs text-muted-foreground">{player2.country}</p>
                  {player2.elo && (
                    <span className="text-xs text-muted-foreground">
                      • ELO: {player2.elo}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {predictedWinner === 2 && (
                <TrendingUp className="h-4 w-4 text-success" />
              )}
              <span className="text-lg font-bold text-primary">
                {player2.confidence}%
              </span>
            </div>
          </div>
          <Progress value={player2.confidence} className="h-2" />
        </div>
      </div>
    </Card>
  );
};
