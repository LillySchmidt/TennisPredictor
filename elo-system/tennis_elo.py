"""
Tennis ELO Rating System
========================
A comprehensive ELO system for tennis with:
- Dynamic K-factor based on tournament importance and round
- Inactivity decay - ratings regress toward the mean when players don't compete
- Historical tracking - query any player's ELO at any point in time
- Surface-specific ELO (optional)
- Margin of victory adjustments

Author: Lilly-Sophie Schmidt
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class EloConfig:
    """Configuration for the ELO system"""
    # Base settings
    initial_elo: float = 1500.0
    mean_elo: float = 1500.0
    
    # K-factor settings by tournament level
    k_factors: Dict[str, float] = field(default_factory=lambda: {
        'G': 48,   # Grand Slams (most important)
        'F': 40,   # Tour Finals / ATP Finals
        'M': 36,   # Masters 1000
        'A': 32,   # ATP 250/500
        'O': 24,   # Other events (Challengers, etc.)
    })
    
    # Round multipliers (later rounds worth more)
    round_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'F': 1.25,    # Final
        'SF': 1.15,   # Semi-final
        'QF': 1.10,   # Quarter-final
        'R16': 1.05,  # Round of 16
        'R32': 1.00,  # Round of 32
        'R64': 0.95,  # Round of 64
        'R128': 0.90, # Round of 128
        'RR': 1.10,   # Round Robin (Tour Finals)
        'BR': 1.00,   # Bronze medal match
    })
    
    # Decay settings
    decay_enabled: bool = True
    decay_days_threshold: int = 60      # Start decay after 60 days of inactivity
    decay_rate_per_day: float = 0.5     # ELO points lost per day after threshold
    decay_cap: float = 200.0            # Maximum total decay
    
    # Margin of victory settings
    mov_enabled: bool = True
    mov_multiplier: float = 0.1  # How much margin affects K-factor


@dataclass
class PlayerSnapshot:
    """A snapshot of a player's ELO at a specific point in time"""
    date: datetime
    elo: float
    match_id: Optional[int] = None
    opponent: Optional[str] = None
    result: Optional[str] = None  # 'W' or 'L'
    elo_change: float = 0.0
    decay_applied: float = 0.0


class TennisEloSystem:
    """
    A comprehensive ELO rating system for tennis players.
    
    Features:
    - Dynamic K-factors based on tournament importance
    - Inactivity decay
    - Full historical tracking
    - Surface-specific ratings (optional)
    """
    
    def __init__(self, config: Optional[EloConfig] = None):
        self.config = config or EloConfig()
        
        # Current ELO ratings (using regular dict for pickle compatibility)
        self.ratings: Dict[str, float] = {}
        
        # Last match date for each player (for decay calculation)
        self.last_match_date: Dict[str, datetime] = {}
        
        # Full history: player -> list of snapshots
        self.history: Dict[str, List[PlayerSnapshot]] = {}
        
        # Surface-specific ratings (optional)
        self.surface_ratings: Dict[str, Dict[str, float]] = {
            'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}
        }
        self.surface_history: Dict[str, Dict[str, List[PlayerSnapshot]]] = {
            'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}
        }
        
        # Match counter for IDs
        self.match_counter = 0
    
    def _get_rating(self, player: str) -> float:
        """Get player rating, initializing if needed"""
        if player not in self.ratings:
            self.ratings[player] = self.config.initial_elo
        return self.ratings[player]
    
    def _get_surface_rating(self, surface: str, player: str) -> float:
        """Get surface-specific rating, initializing if needed"""
        if surface not in self.surface_ratings:
            self.surface_ratings[surface] = {}
        if player not in self.surface_ratings[surface]:
            self.surface_ratings[surface][player] = self.config.initial_elo
        return self.surface_ratings[surface][player]
    
    def _init_history(self, player: str):
        """Initialize history list for player if needed"""
        if player not in self.history:
            self.history[player] = []
    
    def _init_surface_history(self, surface: str, player: str):
        """Initialize surface history for player if needed"""
        if surface not in self.surface_history:
            self.surface_history[surface] = {}
        if player not in self.surface_history[surface]:
            self.surface_history[surface][player] = []
        
    def expected_score(self, player_elo: float, opponent_elo: float) -> float:
        """Calculate expected score (probability of winning) using ELO formula"""
        return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400.0))
    
    def get_k_factor(self, tourney_level: str, round_name: str, 
                     elo_diff: float = 0, sets_won: int = 0, sets_lost: int = 0) -> float:
        """
        Calculate dynamic K-factor based on:
        - Tournament importance
        - Round in tournament
        - Margin of victory (optional)
        """
        base_k = self.config.k_factors.get(tourney_level, 32)
        round_mult = self.config.round_multipliers.get(round_name, 1.0)
        
        k = base_k * round_mult
        
        # Margin of victory adjustment
        if self.config.mov_enabled and sets_won > 0:
            set_margin = sets_won - sets_lost
            # Bigger wins = slightly higher K
            mov_adjustment = 1 + (set_margin * self.config.mov_multiplier * 0.5)
            k *= mov_adjustment
        
        return k
    
    def apply_decay(self, player: str, current_date: datetime) -> float:
        """
        Apply inactivity decay to a player's rating.
        Returns the amount of decay applied.
        """
        if not self.config.decay_enabled:
            return 0.0
            
        if player not in self.last_match_date:
            return 0.0
            
        days_inactive = (current_date - self.last_match_date[player]).days
        
        if days_inactive <= self.config.decay_days_threshold:
            return 0.0
        
        # Calculate decay
        decay_days = days_inactive - self.config.decay_days_threshold
        raw_decay = decay_days * self.config.decay_rate_per_day
        
        # Cap the decay
        decay = min(raw_decay, self.config.decay_cap)
        
        # Only decay if above mean
        current_elo = self._get_rating(player)
        if current_elo > self.config.mean_elo:
            max_decay = current_elo - self.config.mean_elo
            decay = min(decay, max_decay)
            self.ratings[player] = current_elo - decay
            return decay
        
        return 0.0
    
    def parse_score(self, score: str) -> Tuple[int, int]:
        """Parse a tennis score string to get sets won by winner and loser"""
        if pd.isna(score) or score == '':
            return 2, 0  # Default assumption
            
        sets_winner = 0
        sets_loser = 0
        
        try:
            # Handle various score formats
            score = str(score).replace('[', '').replace(']', '')
            sets = score.split()
            
            for s in sets:
                if 'RET' in s.upper() or 'W/O' in s.upper() or 'DEF' in s.upper():
                    continue
                    
                # Parse individual set score like "6-4" or "7-6(5)"
                s = s.split('(')[0]  # Remove tiebreak score
                if '-' in s:
                    parts = s.split('-')
                    if len(parts) == 2:
                        try:
                            w_games = int(parts[0])
                            l_games = int(parts[1])
                            if w_games > l_games:
                                sets_winner += 1
                            elif l_games > w_games:
                                sets_loser += 1
                        except ValueError:
                            continue
        except Exception:
            return 2, 0
            
        return max(sets_winner, 1), sets_loser  # Winner must have won at least 1 set
    
    def process_match(self, winner: str, loser: str, match_date: datetime,
                      tourney_level: str, round_name: str, score: str = '',
                      surface: Optional[str] = None) -> Tuple[float, float]:
        """
        Process a single match and update ratings.
        Returns (winner_elo_change, loser_elo_change)
        """
        self.match_counter += 1
        
        # Initialize histories
        self._init_history(winner)
        self._init_history(loser)
        
        # Apply decay to both players before the match
        winner_decay = self.apply_decay(winner, match_date)
        loser_decay = self.apply_decay(loser, match_date)
        
        # Get current ratings
        winner_elo = self._get_rating(winner)
        loser_elo = self._get_rating(loser)
        
        # Calculate expected scores
        winner_expected = self.expected_score(winner_elo, loser_elo)
        loser_expected = 1 - winner_expected
        
        # Parse score for margin of victory
        sets_won, sets_lost = self.parse_score(score)
        
        # Get K-factor
        k = self.get_k_factor(tourney_level, round_name, 
                             winner_elo - loser_elo, sets_won, sets_lost)
        
        # Calculate ELO changes
        winner_change = k * (1 - winner_expected)
        loser_change = k * (0 - loser_expected)
        
        # Update ratings
        self.ratings[winner] = winner_elo + winner_change
        self.ratings[loser] = loser_elo + loser_change
        
        # Update last match dates
        self.last_match_date[winner] = match_date
        self.last_match_date[loser] = match_date
        
        # Record history
        self.history[winner].append(PlayerSnapshot(
            date=match_date,
            elo=self.ratings[winner],
            match_id=self.match_counter,
            opponent=loser,
            result='W',
            elo_change=winner_change,
            decay_applied=winner_decay
        ))
        
        self.history[loser].append(PlayerSnapshot(
            date=match_date,
            elo=self.ratings[loser],
            match_id=self.match_counter,
            opponent=winner,
            result='L',
            elo_change=loser_change,
            decay_applied=loser_decay
        ))
        
        # Update surface-specific ratings if surface provided
        if surface:
            self._update_surface_ratings(winner, loser, match_date, 
                                        tourney_level, round_name, score, surface)
        
        return winner_change, loser_change
    
    def _update_surface_ratings(self, winner: str, loser: str, match_date: datetime,
                                tourney_level: str, round_name: str, score: str,
                                surface: str):
        """Update surface-specific ratings"""
        # Initialize surface histories
        self._init_surface_history(surface, winner)
        self._init_surface_history(surface, loser)
        
        winner_elo = self._get_surface_rating(surface, winner)
        loser_elo = self._get_surface_rating(surface, loser)
        
        winner_expected = self.expected_score(winner_elo, loser_elo)
        
        sets_won, sets_lost = self.parse_score(score)
        k = self.get_k_factor(tourney_level, round_name, 
                             winner_elo - loser_elo, sets_won, sets_lost)
        
        winner_change = k * (1 - winner_expected)
        loser_change = k * (0 - (1 - winner_expected))
        
        self.surface_ratings[surface][winner] = winner_elo + winner_change
        self.surface_ratings[surface][loser] = loser_elo + loser_change
        
        # Record surface history
        self.surface_history[surface][winner].append(PlayerSnapshot(
            date=match_date,
            elo=self.surface_ratings[surface][winner],
            opponent=loser,
            result='W',
            elo_change=winner_change
        ))
        
        self.surface_history[surface][loser].append(PlayerSnapshot(
            date=match_date,
            elo=self.surface_ratings[surface][loser],
            opponent=winner,
            result='L',
            elo_change=loser_change
        ))
    
    def get_elo(self, player: str, as_of_date: Optional[datetime] = None) -> float:
        """
        Get a player's ELO rating at a specific point in time.
        If no date provided, returns current rating.
        """
        if as_of_date is None:
            return self._get_rating(player)
        
        # Search history for the rating at that date
        if player not in self.history:
            return self.config.initial_elo
        
        player_history = self.history[player]
        
        # Find the most recent snapshot before or on the given date
        relevant_snapshot = None
        for snapshot in player_history:
            if snapshot.date <= as_of_date:
                relevant_snapshot = snapshot
            else:
                break
        
        if relevant_snapshot is None:
            return self.config.initial_elo
        
        return relevant_snapshot.elo
    
    def get_elo_history(self, player: str) -> List[PlayerSnapshot]:
        """Get the complete ELO history for a player"""
        return self.history.get(player, [])
    
    def get_surface_elo(self, player: str, surface: str, 
                        as_of_date: Optional[datetime] = None) -> float:
        """Get a player's surface-specific ELO"""
        if as_of_date is None:
            return self._get_surface_rating(surface, player)
        
        if surface not in self.surface_history:
            return self.config.initial_elo
            
        history = self.surface_history[surface].get(player, [])
        
        relevant_snapshot = None
        for snapshot in history:
            if snapshot.date <= as_of_date:
                relevant_snapshot = snapshot
            else:
                break
        
        if relevant_snapshot is None:
            return self.config.initial_elo
        
        return relevant_snapshot.elo
    
    def get_rankings(self, top_n: int = 50, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get top N players by ELO rating"""
        if as_of_date is None:
            rankings = [(player, elo) for player, elo in self.ratings.items()]
        else:
            rankings = [(player, self.get_elo(player, as_of_date)) 
                       for player in self.ratings.keys()]
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        df = pd.DataFrame(rankings[:top_n], columns=['Player', 'ELO'])
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'Player', 'ELO']]
        df['ELO'] = df['ELO'].round(1)
        
        return df
    
    def get_player_stats(self, player: str) -> dict:
        """Get comprehensive stats for a player"""
        history = self.history.get(player, [])
        
        if not history:
            return {
                'player': player,
                'current_elo': self.config.initial_elo,
                'matches_played': 0,
                'wins': 0,
                'losses': 0,
                'peak_elo': self.config.initial_elo,
                'peak_date': None,
                'lowest_elo': self.config.initial_elo,
                'first_match': None,
                'last_match': None
            }
        
        wins = sum(1 for s in history if s.result == 'W')
        losses = sum(1 for s in history if s.result == 'L')
        peak = max(history, key=lambda s: s.elo)
        lowest = min(history, key=lambda s: s.elo)
        
        return {
            'player': player,
            'current_elo': round(self._get_rating(player), 1),
            'matches_played': len(history),
            'wins': wins,
            'losses': losses,
            'win_rate': round(wins / len(history) * 100, 1),
            'peak_elo': round(peak.elo, 1),
            'peak_date': peak.date,
            'lowest_elo': round(lowest.elo, 1),
            'first_match': history[0].date,
            'last_match': history[-1].date,
            'avg_elo_change_win': round(np.mean([s.elo_change for s in history if s.result == 'W']), 2) if wins > 0 else 0,
            'avg_elo_change_loss': round(np.mean([s.elo_change for s in history if s.result == 'L']), 2) if losses > 0 else 0
        }


def load_and_process_matches(csv_path: str, config: Optional[EloConfig] = None) -> TennisEloSystem:
    """
    Load matches from CSV and process them through the ELO system.
    """
    print("Loading match data...")
    df = pd.read_csv(csv_path)
    
    # Parse dates from the weird format
    def extract_date(date_str):
        try:
            frac = date_str.split('.')[1].split('+')[0]
            year = int(frac[1:5])
            month = int(frac[5:7])
            day = int(frac[7:9])
            return datetime(year=year, month=month, day=day)
        except:
            return None
    
    df['match_date'] = df['tourney_date'].apply(extract_date)
    df = df.dropna(subset=['match_date'])
    
    # Sort by date (important for chronological processing)
    df = df.sort_values('match_date').reset_index(drop=True)
    
    print(f"Processing {len(df)} matches from {df['match_date'].min().date()} to {df['match_date'].max().date()}...")
    
    # Initialize ELO system
    elo_system = TennisEloSystem(config)
    
    # Process each match
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"  Processed {idx}/{len(df)} matches...")
        
        elo_system.process_match(
            winner=row['winner_name'],
            loser=row['loser_name'],
            match_date=row['match_date'],
            tourney_level=row['tourney_level'],
            round_name=row['round'],
            score=row.get('score', ''),
            surface=row.get('surface', None)
        )
    
    print(f"Done! Processed {len(df)} matches for {len(elo_system.ratings)} players.")
    
    return elo_system


def create_elo_dataframe(elo_system: TennisEloSystem) -> pd.DataFrame:
    """Create a DataFrame with all ELO history for analysis"""
    records = []
    
    for player, snapshots in elo_system.history.items():
        for snap in snapshots:
            records.append({
                'player': player,
                'date': snap.date,
                'elo': round(snap.elo, 1),
                'opponent': snap.opponent,
                'result': snap.result,
                'elo_change': round(snap.elo_change, 2),
                'decay_applied': round(snap.decay_applied, 2)
            })
    
    return pd.DataFrame(records).sort_values(['player', 'date']).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    config = EloConfig(
        decay_enabled=True,
        decay_days_threshold=60,
        decay_rate_per_day=0.5,
        decay_cap=200,
        mov_enabled=True
    )
    
    elo_system = load_and_process_matches('data/matches_with_stats_cleaned.csv', config)
    
    print("\n" + "="*60)
    print("TOP 20 CURRENT ELO RATINGS")
    print("="*60)
    print(elo_system.get_rankings(20).to_string(index=False))
    
    # Show some player stats
    print("\n" + "="*60)
    print("SAMPLE PLAYER STATISTICS")
    print("="*60)
    
    top_players = elo_system.get_rankings(5)['Player'].tolist()
    for player in top_players:
        stats = elo_system.get_player_stats(player)
        print(f"\n{player}:")
        print(f"  Current ELO: {stats['current_elo']}")
        print(f"  Peak ELO: {stats['peak_elo']} ({stats['peak_date'].date() if stats['peak_date'] else 'N/A'})")
        print(f"  Record: {stats['wins']}-{stats['losses']} ({stats['win_rate']}%)")
