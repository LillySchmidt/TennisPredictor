import axios from "axios";
import {
  MatchData,
  TournamentDetailsResponse,
  TournamentFormPayload,
} from "@/types/tournament";

const resolveFallbackBaseUrl = () => {
  if (typeof window !== "undefined") {
    const { origin, protocol, hostname, port } = window.location;

    // In production (HTTPS), use same-origin to avoid mixed content.
    if (protocol === "https:") {
      return `${origin}/api`;
    }

    // In local dev (vite on 5173/4173 or docker on 15001), talk to 15002.
    if (port === "15001" || port === "5173" || port === "4173") {
      return `${protocol}//${hostname}:15002/api`;
    }

    // Fallback: same host, default port.
    return `${origin}/api`;
  }

  // Non-browser fallback.
  return "http://localhost:15002/api";
};

const API_BASE_URL = import.meta.env.VITE_API_URL ?? resolveFallbackBaseUrl();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

export const createTournamentRequest = async (
  payload: TournamentFormPayload
): Promise<TournamentDetailsResponse> => {
  const { data } = await apiClient.post<TournamentDetailsResponse>("/tournaments", payload);
  return data;
};

export const fetchTournamentDetails = async (
  tournamentId: number
): Promise<TournamentDetailsResponse> => {
  const { data } = await apiClient.get<TournamentDetailsResponse>(`/tournaments/${tournamentId}`);
  return data;
};

export const fetchTournamentMatches = async (tournamentId: number): Promise<MatchData[]> => {
  const { data } = await apiClient.get<MatchData[]>(`/tournaments/${tournamentId}/matches`);
  return data;
};

export const submitMatchWinner = async (
  tournamentId: number,
  matchId: number,
  winnerSlot: 1 | 2
): Promise<MatchData[]> => {
  const { data } = await apiClient.post<MatchData[]>(
    `/tournaments/${tournamentId}/matches/${matchId}/winner`,
    { winnerSlot }
  );
  return data;
};

export const clearMatchWinner = async (tournamentId: number, matchId: number): Promise<MatchData[]> => {
  const { data } = await apiClient.post<MatchData[]>(
    `/tournaments/${tournamentId}/matches/${matchId}/reset`
  );
  return data;
};

export default apiClient;
