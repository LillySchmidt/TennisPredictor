import { Router } from "express";
import {
  assignWinnerBySlot,
  createTournament,
  getTournamentById,
  getTournamentDetails,
  getTournamentMatches,
  listTournaments,
  TournamentPayload,
  resetMatchWinner,
} from "../services/tournamentService";

export const tournamentsRouter = Router();

const parseId = (value: string, label: string) => {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    throw new Error(`${label} must be a number`);
  }
  return parsed;
};

const buildDrawResponse = (tournamentId: number) => {
  const matches = getTournamentMatches(tournamentId);
  return matches.reduce<Record<string, typeof matches>>((acc, match) => {
    if (!acc[match.roundName]) {
      acc[match.roundName] = [];
    }
    acc[match.roundName].push(match);
    return acc;
  }, {});
};

tournamentsRouter.get("/", (_req, res) => {
  const tournaments = listTournaments();
  res.json(tournaments);
});

tournamentsRouter.post("/", (req, res, next) => {
  try {
    const payload = req.body as TournamentPayload;
    const { tournamentId } = createTournament(payload);
    const details = getTournamentDetails(tournamentId);
    res.status(201).json(details);
  } catch (error) {
    next(error);
  }
});

tournamentsRouter.get("/:id", (req, res, next) => {
  try {
    const tournamentId = parseId(req.params.id, "Tournament id");
    const tournament = getTournamentById(tournamentId);
    if (!tournament) {
      res.status(404).json({ message: "Tournament not found" });
      return;
    }
    const playersMatches = getTournamentDetails(tournamentId);
    res.json(playersMatches);
  } catch (error) {
    next(error);
  }
});

tournamentsRouter.get("/:id/matches", (req, res, next) => {
  try {
    const tournamentId = parseId(req.params.id, "Tournament id");
    const matches = getTournamentMatches(tournamentId);
    res.json(matches);
  } catch (error) {
    next(error);
  }
});

tournamentsRouter.get("/:id/draw", (req, res, next) => {
  try {
    const tournamentId = parseId(req.params.id, "Tournament id");
    const rounds = buildDrawResponse(tournamentId);
    res.json({ rounds });
  } catch (error) {
    next(error);
  }
});

tournamentsRouter.post("/:id/matches/:matchId/winner", (req, res, next) => {
  try {
    const tournamentId = parseId(req.params.id, "Tournament id");
    const matchId = parseId(req.params.matchId, "Match id");
    const winnerSlot = Number(req.body?.winnerSlot);

    if (winnerSlot !== 1 && winnerSlot !== 2) {
      res.status(400).json({ message: "winnerSlot must be 1 or 2" });
      return;
    }

    const matches = assignWinnerBySlot(tournamentId, matchId, winnerSlot);
    res.json(matches);
  } catch (error) {
    next(error);
  }
});

tournamentsRouter.post("/:id/matches/:matchId/reset", (req, res, next) => {
  try {
    const tournamentId = parseId(req.params.id, "Tournament id");
    const matchId = parseId(req.params.matchId, "Match id");
    const matches = resetMatchWinner(tournamentId, matchId);
    res.json(matches);
  } catch (error) {
    next(error);
  }
});
