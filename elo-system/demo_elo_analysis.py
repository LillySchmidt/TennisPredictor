"""
Tennis ELO System - Demonstration & Analysis
=============================================
Shows all features of the ELO system including:
- Current rankings
- Historical ELO queries
- Surface-specific ratings
- Player comparisons over time
"""

import pandas as pd
import numpy as np
from datetime import datetime
from tennis_elo import TennisEloSystem, EloConfig, load_and_process_matches, create_elo_dataframe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def main():
    # Initialize with optimized configuration
    config = EloConfig(
        initial_elo=1500.0,
        
        # K-factors by tournament level
        k_factors={
            'G': 48,   # Grand Slams
            'F': 40,   # ATP Finals
            'M': 36,   # Masters 1000
            'A': 32,   # ATP 250/500
            'O': 24,   # Other
        },
        
        # Decay settings
        decay_enabled=True,
        decay_days_threshold=60,    # Start decay after 60 days inactive
        decay_rate_per_day=0.5,     # Points per day
        decay_cap=200,              # Maximum decay
        
        # Margin of victory
        mov_enabled=True,
        mov_multiplier=0.1
    )
    
    # Load and process all matches
    elo_system = load_and_process_matches('data/matches_with_stats_cleaned.csv', config)
    
    print("\n" + "="*80)
    print("TENNIS ELO RATING SYSTEM - ANALYSIS REPORT")
    print("="*80)
    
    # ============================================================
    # 1. CURRENT TOP 30 RANKINGS (with active player filter)
    # ============================================================
    print("\n" + "-"*80)
    print("SECTION 1: CURRENT TOP 30 ELO RATINGS")
    print("-"*80)
    
    rankings = elo_system.get_rankings(50)
    
    # Add player stats
    rankings_with_stats = []
    for _, row in rankings.iterrows():
        stats = elo_system.get_player_stats(row['Player'])
        last_match = stats['last_match']
        is_active = last_match and last_match.year >= 2024
        rankings_with_stats.append({
            'Rank': row['Rank'],
            'Player': row['Player'],
            'ELO': row['ELO'],
            'Peak': stats['peak_elo'],
            'Record': f"{stats['wins']}-{stats['losses']}",
            'Win%': stats['win_rate'],
            'Last Match': last_match.strftime('%Y-%m-%d') if last_match else 'N/A',
            'Active': '✓' if is_active else ''
        })
    
    df_rankings = pd.DataFrame(rankings_with_stats)
    print(df_rankings.head(30).to_string(index=False))
    
    # ============================================================
    # 2. ACTIVE PLAYERS ONLY
    # ============================================================
    print("\n" + "-"*80)
    print("SECTION 2: TOP 20 ACTIVE PLAYERS (played in 2024)")
    print("-"*80)
    
    active_only = df_rankings[df_rankings['Active'] == '✓'].head(20)
    active_only = active_only.reset_index(drop=True)
    active_only['Rank'] = range(1, len(active_only) + 1)
    print(active_only[['Rank', 'Player', 'ELO', 'Peak', 'Record', 'Win%']].to_string(index=False))
    
    # ============================================================
    # 3. HISTORICAL ELO QUERIES - Point-in-time lookups
    # ============================================================
    print("\n" + "-"*80)
    print("SECTION 3: HISTORICAL ELO QUERIES")
    print("-"*80)
    
    # Key dates in tennis history
    historical_queries = [
        ("Novak Djokovic", datetime(2011, 7, 1), "Peak Djokovic era"),
        ("Novak Djokovic", datetime(2016, 6, 6), "After 2016 French Open (Career Slam)"),
        ("Novak Djokovic", datetime(2024, 1, 1), "Start of 2024"),
        ("Roger Federer", datetime(2007, 1, 1), "Peak Federer era"),
        ("Roger Federer", datetime(2017, 2, 1), "After 2017 AO comeback"),
        ("Rafael Nadal", datetime(2010, 6, 15), "Peak Nadal era"),
        ("Rafael Nadal", datetime(2024, 1, 1), "Nadal in 2024"),
        ("Carlos Alcaraz", datetime(2022, 9, 15), "After first US Open win"),
        ("Carlos Alcaraz", datetime(2024, 7, 15), "After 2024 Wimbledon"),
        ("Jannik Sinner", datetime(2024, 1, 30), "After 2024 AO win"),
        ("Jannik Sinner", datetime(2024, 11, 1), "Current Sinner"),
    ]
    
    print(f"\n{'Player':<20} {'Date':<15} {'ELO':>8}  Context")
    print("-"*70)
    
    for player, date, context in historical_queries:
        elo = elo_system.get_elo(player, date)
        print(f"{player:<20} {date.strftime('%Y-%m-%d'):<15} {elo:>8.1f}  {context}")
    
    # ============================================================
    # 4. PEAK ELO ANALYSIS - All-time greats
    # ============================================================
    print("\n" + "-"*80)
    print("SECTION 4: HIGHEST PEAK ELO RATINGS (All-Time)")
    print("-"*80)
    
    all_peaks = []
    for player in elo_system.history.keys():
        stats = elo_system.get_player_stats(player)
        if stats['matches_played'] >= 50:  # Minimum matches for validity
            all_peaks.append({
                'Player': player,
                'Peak ELO': stats['peak_elo'],
                'Peak Date': stats['peak_date'].strftime('%Y-%m-%d') if stats['peak_date'] else 'N/A',
                'Matches': stats['matches_played'],
                'Win%': stats['win_rate']
            })
    
    peaks_df = pd.DataFrame(all_peaks)
    peaks_df = peaks_df.sort_values('Peak ELO', ascending=False).head(25)
    peaks_df['Rank'] = range(1, len(peaks_df) + 1)
    peaks_df = peaks_df[['Rank', 'Player', 'Peak ELO', 'Peak Date', 'Win%', 'Matches']]
    print(peaks_df.to_string(index=False))
    
    # ============================================================
    # 5. SURFACE-SPECIFIC RATINGS
    # ============================================================
    print("\n" + "-"*80)
    print("SECTION 5: SURFACE-SPECIFIC RATINGS (Current)")
    print("-"*80)
    
    surfaces = ['Hard', 'Clay', 'Grass']
    
    for surface in surfaces:
        print(f"\n{surface.upper()} COURT - Top 10:")
        surface_rankings = []
        for player in elo_system.surface_ratings[surface].keys():
            elo = elo_system.surface_ratings[surface][player]
            overall = elo_system.ratings[player]
            if elo != 1500:  # Has played on this surface
                surface_rankings.append((player, elo, overall))
        
        surface_rankings.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  {'Rank':<5} {'Player':<25} {'Surface ELO':>12} {'Overall ELO':>12}")
        for i, (player, surf_elo, overall_elo) in enumerate(surface_rankings[:10], 1):
            print(f"  {i:<5} {player:<25} {surf_elo:>12.1f} {overall_elo:>12.1f}")
    
    # ============================================================
    # 6. HEAD-TO-HEAD ELO COMPARISON
    # ============================================================
    print("\n" + "-"*80)
    print("SECTION 6: HEAD-TO-HEAD WIN PROBABILITY (Current ELO)")
    print("-"*80)
    
    players = ["Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", 
               "Alexander Zverev", "Daniil Medvedev", "Taylor Fritz"]
    
    # Create probability matrix
    print(f"\nExpected Win Probability (row player vs column player):")
    print(f"\n{'':20}", end='')
    for p in players:
        print(f"{p[:10]:>12}", end='')
    print()
    
    for p1 in players:
        elo1 = elo_system.ratings[p1]
        print(f"{p1:<20}", end='')
        for p2 in players:
            if p1 == p2:
                print(f"{'---':>12}", end='')
            else:
                elo2 = elo_system.ratings[p2]
                prob = elo_system.expected_score(elo1, elo2)
                print(f"{prob*100:>11.1f}%", end='')
        print()
    
    # ============================================================
    # 7. PLAYER ELO HISTORY VISUALIZATION
    # ============================================================
    print("\n" + "-"*80)
    print("SECTION 7: CREATING ELO HISTORY CHARTS")
    print("-"*80)
    
    # Create visualization for top players
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    players_to_plot = [
        ("Novak Djokovic", "blue"),
        ("Roger Federer", "red"),
        ("Rafael Nadal", "orange"),
        ("Jannik Sinner", "green")
    ]
    
    # Individual player charts
    for idx, (player, color) in enumerate(players_to_plot):
        ax = axes[idx // 2, idx % 2]
        history = elo_system.get_elo_history(player)
        
        if history:
            dates = [s.date for s in history]
            elos = [s.elo for s in history]
            
            ax.plot(dates, elos, color=color, linewidth=1.5, alpha=0.8)
            ax.fill_between(dates, 1500, elos, alpha=0.3, color=color)
            
            # Mark peak
            peak_idx = np.argmax(elos)
            ax.scatter([dates[peak_idx]], [elos[peak_idx]], 
                      color='gold', s=100, zorder=5, edgecolors='black')
            ax.annotate(f'Peak: {elos[peak_idx]:.0f}', 
                       xy=(dates[peak_idx], elos[peak_idx]),
                       xytext=(10, 10), textcoords='offset points', fontsize=9)
            
            ax.set_title(f"{player} ELO History", fontsize=14, fontweight='bold')
            ax.set_ylabel("ELO Rating")
            ax.set_xlabel("Date")
            ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='Average')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('elo_history_charts.png', dpi=150, bbox_inches='tight')
    print("Saved: elo_history_charts.png")
    
    # Combined comparison chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for player, color in players_to_plot:
        history = elo_system.get_elo_history(player)
        if history:
            dates = [s.date for s in history]
            elos = [s.elo for s in history]
            ax.plot(dates, elos, color=color, linewidth=2, label=player, alpha=0.8)
    
    ax.set_title("ELO Rating Comparison - Tennis Legends", fontsize=16, fontweight='bold')
    ax.set_ylabel("ELO Rating", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('elo_comparison_chart.png', dpi=150, bbox_inches='tight')
    print("Saved: elo_comparison_chart.png")
    
    # ============================================================
    # 8. EXPORT FULL ELO HISTORY TO CSV
    # ============================================================
    print("\n" + "-"*80)
    print("SECTION 8: EXPORTING DATA")
    print("-"*80)
    
    # Export full history
    history_df = create_elo_dataframe(elo_system)
    history_df.to_csv('elo_history.csv', index=False)
    print(f"Saved: elo_history.csv ({len(history_df)} records)")
    
    # Export current rankings
    final_rankings = []
    for player in elo_system.ratings.keys():
        stats = elo_system.get_player_stats(player)
        last_match = stats['last_match']
        final_rankings.append({
            'player': player,
            'current_elo': stats['current_elo'],
            'peak_elo': stats['peak_elo'],
            'peak_date': stats['peak_date'].strftime('%Y-%m-%d') if stats['peak_date'] else None,
            'matches': stats['matches_played'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'win_rate': stats['win_rate'],
            'first_match': stats['first_match'].strftime('%Y-%m-%d') if stats['first_match'] else None,
            'last_match': last_match.strftime('%Y-%m-%d') if last_match else None,
            'active_2024': last_match and last_match.year >= 2024 if last_match else False
        })
    
    rankings_df = pd.DataFrame(final_rankings)
    rankings_df = rankings_df.sort_values('current_elo', ascending=False)
    rankings_df.to_csv('player_ratings.csv', index=False)
    print(f"Saved: player_ratings.csv ({len(rankings_df)} players)")
    
    # Surface-specific export
    surface_data = []
    for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
        for player, elo in elo_system.surface_ratings[surface].items():
            if elo != 1500:
                surface_data.append({
                    'player': player,
                    'surface': surface,
                    'surface_elo': round(elo, 1)
                })
    
    surface_df = pd.DataFrame(surface_data)
    surface_df.to_csv('surface_ratings.csv', index=False)
    print(f"Saved: surface_ratings.csv ({len(surface_df)} records)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return elo_system


if __name__ == "__main__":
    elo = main()
