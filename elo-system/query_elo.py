"""
Tennis ELO Query Tool
=====================
Interactive command-line tool to query player ELO ratings at any point in time.

Usage:
    python query_elo.py                     # Interactive mode
    python query_elo.py "Player Name"       # Current ELO
    python query_elo.py "Player Name" YYYY-MM-DD  # ELO at date
"""

import sys
import pickle
from datetime import datetime
from tennis_elo import TennisEloSystem, EloConfig, load_and_process_matches


def load_or_build_system():
    """Load cached system or build from scratch"""
    cache_file = 'elo_system.pkl'
    
    try:
        with open(cache_file, 'rb') as f:
            print("Loading cached ELO system...")
            return pickle.load(f)
    except FileNotFoundError:
        print("Building ELO system from match data...")
        config = EloConfig(
            decay_enabled=True,
            decay_days_threshold=60,
            decay_rate_per_day=0.5,
            decay_cap=200,
            mov_enabled=True
        )
        system = load_and_process_matches('data/matches_with_stats_cleaned.csv', config)
        
        # Cache for future use
        with open(cache_file, 'wb') as f:
            pickle.dump(system, f)
        print("Cached for faster future loading.")
        
        return system


def find_player(elo_system: TennisEloSystem, query: str):
    """Find player by partial name match"""
    query_lower = query.lower()
    matches = [p for p in elo_system.ratings.keys() 
               if query_lower in p.lower()]
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"\nMultiple matches found for '{query}':")
        for i, m in enumerate(matches[:10], 1):
            print(f"  {i}. {m}")
        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")
        return None
    else:
        print(f"No player found matching '{query}'")
        return None


def display_player_info(elo_system: TennisEloSystem, player: str, 
                        as_of_date: datetime = None):
    """Display comprehensive player info"""
    
    if as_of_date:
        elo = elo_system.get_elo(player, as_of_date)
        print(f"\n{'='*60}")
        print(f"{player}")
        print(f"{'='*60}")
        print(f"ELO as of {as_of_date.strftime('%Y-%m-%d')}: {elo:.1f}")
    else:
        stats = elo_system.get_player_stats(player)
        
        print(f"\n{'='*60}")
        print(f"{player}")
        print(f"{'='*60}")
        print(f"Current ELO:    {stats['current_elo']}")
        print(f"Peak ELO:       {stats['peak_elo']} ({stats['peak_date'].strftime('%Y-%m-%d') if stats['peak_date'] else 'N/A'})")
        print(f"Lowest ELO:     {stats['lowest_elo']}")
        print(f"Record:         {stats['wins']}-{stats['losses']} ({stats['win_rate']}%)")
        print(f"Matches:        {stats['matches_played']}")
        print(f"First Match:    {stats['first_match'].strftime('%Y-%m-%d') if stats['first_match'] else 'N/A'}")
        print(f"Last Match:     {stats['last_match'].strftime('%Y-%m-%d') if stats['last_match'] else 'N/A'}")
        
        # Show surface ratings
        print(f"\nSurface Ratings:")
        for surface in ['Hard', 'Clay', 'Grass']:
            surf_elo = elo_system.surface_ratings[surface].get(player, 1500)
            if surf_elo != 1500:
                print(f"  {surface:10}: {surf_elo:.1f}")
        
        # Show recent form (last 10 matches)
        history = elo_system.get_elo_history(player)
        if history:
            recent = history[-10:]
            print(f"\nRecent Form (last {len(recent)} matches):")
            for snap in recent:
                change_str = f"+{snap.elo_change:.1f}" if snap.elo_change > 0 else f"{snap.elo_change:.1f}"
                print(f"  {snap.date.strftime('%Y-%m-%d')} {snap.result} vs {snap.opponent[:20]:<20} ELO: {snap.elo:.0f} ({change_str})")


def interactive_mode(elo_system: TennisEloSystem):
    """Run interactive query mode"""
    print("\n" + "="*60)
    print("TENNIS ELO QUERY TOOL - Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  <player name>              - Show current stats")
    print("  <player name> YYYY-MM-DD   - ELO at specific date")
    print("  top [N]                    - Show top N rankings")
    print("  compare <p1> vs <p2>       - Win probability")
    print("  quit                       - Exit")
    print("-"*60)
    
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if query.lower().startswith('top'):
            parts = query.split()
            n = int(parts[1]) if len(parts) > 1 else 20
            print(elo_system.get_rankings(n).to_string(index=False))
            continue
        
        if ' vs ' in query.lower():
            parts = query.lower().split(' vs ')
            p1 = find_player(elo_system, parts[0].strip())
            p2 = find_player(elo_system, parts[1].strip())
            if p1 and p2:
                elo1 = elo_system.ratings[p1]
                elo2 = elo_system.ratings[p2]
                prob = elo_system.expected_score(elo1, elo2) * 100
                print(f"\n{p1} ({elo1:.0f}) vs {p2} ({elo2:.0f})")
                print(f"Win probability: {p1}: {prob:.1f}% | {p2}: {100-prob:.1f}%")
            continue
        
        # Check for date
        parts = query.rsplit(' ', 1)
        player_query = parts[0]
        date = None
        
        if len(parts) == 2:
            try:
                date = datetime.strptime(parts[1], '%Y-%m-%d')
                player_query = parts[0]
            except ValueError:
                pass
        
        player = find_player(elo_system, player_query)
        if player:
            display_player_info(elo_system, player, date)


def main():
    elo_system = load_or_build_system()
    
    if len(sys.argv) == 1:
        interactive_mode(elo_system)
    elif len(sys.argv) == 2:
        player = find_player(elo_system, sys.argv[1])
        if player:
            display_player_info(elo_system, player)
    elif len(sys.argv) == 3:
        player = find_player(elo_system, sys.argv[1])
        try:
            date = datetime.strptime(sys.argv[2], '%Y-%m-%d')
            if player:
                display_player_info(elo_system, player, date)
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
