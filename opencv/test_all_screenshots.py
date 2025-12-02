# Test script to run predictions on all screenshots in the screenshots folder

import cv2
import os
import sys
import numpy as np
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

def test_screenshot(img_path, model):
    """Test a single screenshot and return prediction results."""
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Extract game state
    time_remaining, spike_planted = get_time_remaining(img)
    
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
    predicted_confidence = float(probabilities[predicted_winner])
    atk_pct = float(probabilities.get("ATK", 0)) * 100
    def_pct = float(probabilities.get("DEF", 0)) * 100
    
    return {
        "screenshot": os.path.basename(img_path),
        "time_remaining": time_remaining if time_remaining else 0,
        "spike_planted": spike_planted,
        "atk_alive": len(atk_agents),
        "def_alive": len(def_agents),
        "atk_credits": atk_credits,
        "def_credits": def_credits,
        "predicted_winner": predicted_winner,
        "confidence": predicted_confidence,
        "atk_pct": atk_pct,
        "def_pct": def_pct,
    }


def main():
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'naive_bayes_stats.json')
    model = GaussianNaiveBayes.load(model_path)
    
    # Get all screenshots
    screenshots_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
    screenshots = sorted([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])
    
    print("=" * 100)
    print(f"{'Screenshot':<30} {'Time':>6} {'Spike':>6} {'ATK':>4} {'DEF':>4} {'ATK$':>8} {'DEF$':>8} {'Winner':>8} {'Conf':>8} {'ATK%':>8} {'DEF%':>8}")
    print("=" * 100)
    
    results = []
    for screenshot in screenshots:
        img_path = os.path.join(screenshots_dir, screenshot)
        result = test_screenshot(img_path, model)
        
        if result:
            results.append(result)
            print(f"{result['screenshot']:<30} {result['time_remaining']:>6} {str(result['spike_planted']):>6} {result['atk_alive']:>4} {result['def_alive']:>4} {result['atk_credits']:>8} {result['def_credits']:>8} {result['predicted_winner']:>8} {result['confidence']:>7.2%} {result['atk_pct']:>7.1f}% {result['def_pct']:>7.1f}%")
    
    print("=" * 100)
    
    # Summary
    atk_wins = sum(1 for r in results if r['predicted_winner'] == 'ATK')
    def_wins = sum(1 for r in results if r['predicted_winner'] == 'DEF')
    avg_conf = sum(r['confidence'] for r in results) / len(results) if results else 0
    
    print(f"\nSummary: {len(results)} screenshots tested")
    print(f"  ATK predicted wins: {atk_wins}")
    print(f"  DEF predicted wins: {def_wins}")
    print(f"  Average confidence: {avg_conf:.2%}")


if __name__ == "__main__":
    main()

