import { useCallback, useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { TournamentBracket } from "@/components/TournamentBracket";
import { DrawView } from "@/components/DrawView";
import { InitialDrawView } from "@/components/InitialDrawView";
import { PredictedDrawView } from "@/components/PredictedDrawView";
import { ImportTournament } from "@/components/ImportTournament";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  MatchData,
  TournamentData,
  TournamentDetailsResponse,
} from "@/types/tournament";
import { generateRandomTournament, generateRandomTournamentPayload } from "@/lib/randomTournament";
import { Shuffle } from "lucide-react";
import { createTournamentRequest } from "@/lib/api";

const Index = () => {
  const [tournament, setTournament] = useState<TournamentData | null>(null);
  const [matches, setMatches] = useState<MatchData[]>([]);
  const [hasLoadedDefault, setHasLoadedDefault] = useState(false);

  const handleTournamentCreate = (details: TournamentDetailsResponse) => {
    setTournament({ ...details.tournament, players: details.players });
    setMatches(details.matches);
  };

  const handleGenerateRandom = useCallback(async () => {
    try {
      const { payload } = generateRandomTournamentPayload();
      const details = await createTournamentRequest(payload);
      setTournament({ ...details.tournament, players: details.players });
      setMatches(details.matches);
    } catch (err) {
      console.error("Falling back to local random tournament; API create failed", err);
      const { tournament: randomTournament, matches: randomMatches } = generateRandomTournament();
      setTournament(randomTournament);
      setMatches(randomMatches);
    } finally {
      setHasLoadedDefault(true);
    }
  }, []);

  useEffect(() => {
    if (tournament || hasLoadedDefault) return;
    handleGenerateRandom();
  }, [tournament, hasLoadedDefault, handleGenerateRandom]);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container py-8">
        <div className="flex justify-end mb-6">
          <Button variant="outline" onClick={handleGenerateRandom} className="gap-2">
            <Shuffle className="h-4 w-4" />
            Random top 100 ELO
          </Button>
        </div>
        <Tabs defaultValue="upcoming" className="w-full">
          <TabsList className="grid w-full max-w-4xl mx-auto grid-cols-5 mb-8">
            <TabsTrigger value="upcoming">Upcoming</TabsTrigger>
            <TabsTrigger value="initial-draw">Initial Draw</TabsTrigger>
            <TabsTrigger value="draw">Draw</TabsTrigger>
            <TabsTrigger value="predicted-draw">Predicted Draw</TabsTrigger>
            <TabsTrigger value="import">Import Tournament</TabsTrigger>
          </TabsList>

          <TabsContent value="upcoming">
            <TournamentBracket tournament={tournament} matches={matches} />
          </TabsContent>

          <TabsContent value="initial-draw">
            <InitialDrawView tournament={tournament} />
          </TabsContent>

          <TabsContent value="draw">
            <DrawView tournament={tournament} matches={matches} setMatches={setMatches} />
          </TabsContent>

          <TabsContent value="predicted-draw">
            <PredictedDrawView tournament={tournament} manualMatches={matches} />
          </TabsContent>

          <TabsContent value="import">
            <ImportTournament onTournamentCreate={handleTournamentCreate} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Index;
