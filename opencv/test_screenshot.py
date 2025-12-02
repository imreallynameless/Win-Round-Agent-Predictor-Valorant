#!/usr/bin/env python3
"""
Test a single screenshot with the prediction model.

Usage:
    python test_screenshot.py screenshots/val4_operator.png
    python test_screenshot.py screenshots/val4_operator.png --map haven
"""

import cv2
import os
import sys
import argparse
import pandas as pd

# Add parent directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.naive_bayes import GaussianNaiveBayes

# Import functions from opencv.py
from opencv import (
    load_agent_icons, load_gun_icons, 
    get_time_remaining, get_players_alive, get_alive_player_agents,
    get_roles_alive, get_team_credits, get_atk_team
)


def test_screenshot(img_path: str, model: GaussianNaiveBayes) -> dict:
    """Test a single screenshot and return prediction results."""
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error: Could not load image: {img_path}")
        return None
    
    print(f"\n🎮 Analyzing: {img_path}")
    print("="*60)
    
    # Extract game state
    time_remaining, spike_planted = get_time_remaining(img)
    print(f"⏱️  Time remaining: {time_remaining}s" + (" 💣 SPIKE PLANTED" if spike_planted else ""))
    
    t1_players_alive = get_players_alive(img, 1)
    t2_players_alive = get_players_alive(img, 2)
    
    agent_templates1, agent_templates2 = load_agent_icons()
    t1_agents_alive = get_alive_player_agents(img, 1, t1_players_alive, agent_templates1)
    t2_agents_alive = get_alive_player_agents(img, 2, t2_players_alive, agent_templates2)
    
    t1_duelists, t1_initiators, t1_sentinels, t1_controllers = get_roles_alive(t1_agents_alive)
    t2_duelists, t2_initiators, t2_sentinels, t2_controllers = get_roles_alive(t2_agents_alive)
    
    gun_templates1, gun_templates2 = load_gun_icons()
    _, _, _, t1_credits = get_team_credits(img, 1, t1_players_alive, t1_agents_alive, gun_templates1)
    _, _, _, t2_credits = get_team_credits(img, 2, t2_players_alive, t2_agents_alive, gun_templates2)
    
    # Determine ATK/DEF
    atk_team = get_atk_team(img)
    
    if atk_team == 1:
        atk_duelists, atk_initiators, atk_sentinels, atk_controllers = t1_duelists, t1_initiators, t1_sentinels, t1_controllers
        def_duelists, def_initiators, def_sentinels, def_controllers = t2_duelists, t2_initiators, t2_sentinels, t2_controllers
        atk_credits, def_credits = t1_credits, t2_credits
        atk_agents, def_agents = t1_agents_alive, t2_agents_alive
    else:
        atk_duelists, atk_initiators, atk_sentinels, atk_controllers = t2_duelists, t2_initiators, t2_sentinels, t2_controllers
        def_duelists, def_initiators, def_sentinels, def_controllers = t1_duelists, t1_initiators, t1_sentinels, t1_controllers
        atk_credits, def_credits = t2_credits, t1_credits
        atk_agents, def_agents = t2_agents_alive, t1_agents_alive
    
    print(f"\n👥 Players Alive:")
    print(f"   ATK: {len(atk_agents)} ({', '.join(atk_agents) if atk_agents else 'none'})")
    print(f"   DEF: {len(def_agents)} ({', '.join(def_agents) if def_agents else 'none'})")
    
    print(f"\n💰 Team Credits:")
    print(f"   ATK: ${atk_credits:,}")
    print(f"   DEF: ${def_credits:,}")
    
    print(f"\n🎭 Roles Alive:")
    print(f"   ATK: {atk_duelists}D {atk_initiators}I {atk_sentinels}S {atk_controllers}C")
    print(f"   DEF: {def_duelists}D {def_initiators}I {def_sentinels}S {def_controllers}C")
    
    # Prepare features for model
    features = pd.DataFrame([{
        "time_remaining_s": time_remaining if time_remaining else 0,
        "spike_planted": 1 if spike_planted else 0,
        "atk_loadout_value": atk_credits,
        "def_loadout_value": def_credits,
        "atk_duelists_alive": atk_duelists,
        "atk_controllers_alive": atk_controllers,
        "atk_initiators_alive": atk_initiators,
        "atk_sentinels_alive": atk_sentinels,
        "def_duelists_alive": def_duelists,
        "def_controllers_alive": def_controllers,
        "def_initiators_alive": def_initiators,
        "def_sentinels_alive": def_sentinels,
    }])
    
    # Get prediction
    probabilities = model.predict_proba(features).iloc[0]
    predicted_winner = probabilities.idxmax()
    atk_pct = float(probabilities.get("ATK", 0)) * 100
    def_pct = float(probabilities.get("DEF", 0)) * 100
    
    print(f"\n" + "="*60)
    print(f"🔮 PREDICTION")
    print("="*60)
    print(f"\n   ATK Win Probability: {atk_pct:.1f}%")
    print(f"   DEF Win Probability: {def_pct:.1f}%")
    print(f"\n   {'🟢 ATK WINS' if predicted_winner == 'ATK' else '🔵 DEF WINS'} ({max(atk_pct, def_pct):.1f}% confidence)")
    print("="*60 + "\n")
    
    return {
        "time_remaining": time_remaining if time_remaining else 0,
        "spike_planted": spike_planted,
        "atk_alive": len(atk_agents),
        "def_alive": len(def_agents),
        "atk_credits": atk_credits,
        "def_credits": def_credits,
        "predicted_winner": predicted_winner,
        "atk_pct": atk_pct,
        "def_pct": def_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Test a screenshot with the prediction model")
    parser.add_argument("screenshot", help="Path to the screenshot file")
    parser.add_argument("--map", default="haven", help="Map name for model selection (default: haven)")
    args = parser.parse_args()
    
    # Load model for the specified map
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', args.map.lower(), 'naive_bayes_stats.json')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print(f"   Available maps: ", end="")
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        maps = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        print(", ".join(maps))
        sys.exit(1)
    
    print(f"📦 Loading model: {args.map}")
    model = GaussianNaiveBayes.load(model_path)
    
    # Test the screenshot
    result = test_screenshot(args.screenshot, model)
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

