import json
import requests
import pandas as pd
import os
import argparse

# --- Agent Role Map ---
AGENT_ROLES = {
    # Duelists
    14: "Duelist", 17: "Duelist", 15: "Duelist", 16: "Duelist", 
    18: "Duelist", 19: "Duelist", 24: "Duelist",
    # Controllers
    4: "Controller", 2: "Controller", 22: "Controller", 
    20: "Controller", 21: "Controller", 25: "Controller", 8: "Controller",
    # Initiators
    11: "Initiator", 6: "Initiator", 13: "Initiator", 
    21: "Initiator", 12: "Initiator", 23: "Initiator",
    # Sentinels
    1: "Sentinel", 5: "Sentinel", 7: "Sentinel", 
    10: "Sentinel", 26: "Sentinel", 9: "Sentinel", 27: "Sentinel"
}

def get_role(agent_id):
    return AGENT_ROLES.get(agent_id, "Unknown")

def get_loadout_bucket(value):
    return value

def infer_teams(data):
    """
    Infer Team 1 and Team 2 based on Kill/Plant/Defuse events.
    Returns a dict: {playerId: team_num (1 or 2)}
    """
    if 'playerStats' not in data:
        return {}
    
    players = [p['playerId'] for p in data['playerStats']]
    if not players: return {}
    
    # Adjacency list for 'Enemies'
    adj = {p: set() for p in players}
    
    # 1. Build Graph from Events
    if 'events' in data:
        for ev in data['events']:
            # Kill: Killer vs Victim -> Enemies
            if 'killId' in ev:
                k = ev.get('killerId') or ev.get('playerId')
                v = ev.get('referencePlayerId')
                if k and v and k in adj and v in adj:
                    adj[k].add(v)
                    adj[v].add(k)
            
    # 2. Color the Graph (BFS)
    team1_set = set()
    team2_set = set()
    processed = set()
    
    # Iterate all components (in case of disconnects)
    for start_node in players:
        if start_node in processed: continue
        
        # Start BFS
        queue = [(start_node, 1)] # Node, Team (1 or 2 relative to component)
        team1_set.add(start_node)
        processed.add(start_node)
        
        while queue:
            curr, t = queue.pop(0)
            other_t = 2 if t == 1 else 1
            
            for enemy in adj[curr]:
                if enemy not in processed:
                    processed.add(enemy)
                    if other_t == 1: team1_set.add(enemy)
                    else: team2_set.add(enemy)
                    queue.append((enemy, other_t))
    
    # 3. Map to Real Team Numbers (1 vs 2) using metadata
    real_team1 = team1_set
    real_team2 = team2_set
    
    mapping_found = False
    if 'events' in data:
        for ev in data['events']:
            if ev.get('eventType') == 'plant' and 'attackingTeamNumber' in ev:
                pid = ev.get('plantedBy', {}).get('playerId') or ev.get('playerId')
                atk_num = ev['attackingTeamNumber']
                
                if pid in team1_set:
                    if atk_num == 2:
                        real_team1 = team2_set
                        real_team2 = team1_set
                    mapping_found = True
                    break
                elif pid in team2_set:
                    if atk_num == 1:
                        real_team1 = team2_set
                        real_team2 = team1_set
                    mapping_found = True
                    break
        if mapping_found: pass

    # Construct Result
    res = {}
    for p in real_team1: res[p] = 1
    for p in real_team2: res[p] = 2
    
    # Fill any missing players
    for p in players:
        if p not in res:
            if len(real_team1) < 5: res[p] = 1; real_team1.add(p)
            else: res[p] = 2; real_team2.add(p)
            
    return res

def process_matches(mode):
    base_dir = f"{mode}_data"
    match_list_file = os.path.join(base_dir, f"{mode}_matches.txt")
    output_csv = os.path.join(base_dir, f"{mode}_haven_round_data.csv")
    data_dir = os.path.join(base_dir, f"{mode}_data_json")

    if not os.path.exists(match_list_file):
        print(f"No {match_list_file} found.")
        return

    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(match_list_file, "r") as f:
        match_ids = [line.strip() for line in f if line.strip()]

    all_rows = []

    for mid in match_ids:
        json_path = None
        # Find the file for this match ID (either {mid}.json or {slug}-{mid}.json)
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith(f"-{mid}.json") or filename == f"{mid}.json":
                    json_path = os.path.join(data_dir, filename)
                    break
        
        if not json_path:
            # Fallback to fetch if not found locally
            json_path = os.path.join(data_dir, f"{mid}.json")
            print(f"Fetching {mid}...")
            try:
                r = requests.get(f"https://be-prod.rib.gg/v1/matches/{mid}/details")
                if r.status_code == 200:
                    data = r.json()
                    with open(json_path, "w") as f:
                        json.dump(data, f)
                else:
                    print(f"Failed to fetch {mid}: Status {r.status_code}")
                    continue
            except Exception as e:
                print(f"Error fetching {mid}: {e}")
                continue
        else:
            with open(json_path, "r") as f:
                data = json.load(f)

        # 1. Map Players using Inference
        team_map = infer_teams(data) # pid -> 1 or 2
        if not team_map:
            print(f"Skipping {mid}: Could not infer teams")
            continue
            
        # Build full player info
        pid_to_agent = {}
        if 'economies' in data:
            for eco in data['economies']:
                if eco['roundNumber'] == 1:
                    pid_to_agent[eco['playerId']] = eco['agentId']
                    
        player_map = {}
        for pid, t_num in team_map.items():
            aid = pid_to_agent.get(pid, 0)
            role = get_role(aid)
            player_map[pid] = {"team": t_num, "role": role, "agentId": aid}

        print(f"Mapped {len(player_map)} players for {mid}")

        # 2. Process Rounds
        events_by_round = {}
        if 'events' in data:
            for ev in data['events']:
                rid = ev['roundId'] 
                if rid not in events_by_round:
                    events_by_round[rid] = []
                events_by_round[rid].append(ev)

        round_ids = sorted(list(set(e['roundId'] for e in data['economies'])))
        
        # Running Score
        t1_score = 0
        t2_score = 0
        
        for rid in round_ids:
            round_ecos = [e for e in data['economies'] if e['roundId'] == rid]
            if not round_ecos: continue
            
            r_num = round_ecos[0]['roundNumber']
            r_events = sorted(events_by_round.get(rid, []), key=lambda x: x['roundTimeMillis'])
            
            # --- Determine Winner ---
            winner = "Unknown"
            winning_team = 0 
            
            has_defuse = any(ev.get('eventType') == 'defuse' for ev in r_events)
            plant_event = next((ev for ev in r_events if ev.get('eventType') == 'plant'), None)
            
            attacker_team = 0
            start_ev = next((ev for ev in r_events if ev.get('eventType') == 'start'), None)
            if start_ev and 'attackingTeamNumber' in start_ev:
                attacker_team = start_ev['attackingTeamNumber']
            
            if not attacker_team and plant_event and 'attackingTeamNumber' in plant_event:
                attacker_team = plant_event['attackingTeamNumber']

            t1_survived = any(e['survived'] for e in round_ecos if player_map.get(e['playerId'],{}).get('team') == 1)
            t2_survived = any(e['survived'] for e in round_ecos if player_map.get(e['playerId'],{}).get('team') == 2)

            if has_defuse:
                winning_team = 3 - attacker_team if attacker_team else 0 
                winner = "DEF"
            elif plant_event:
                winning_team = attacker_team
                winner = "ATK"
            else:
                if not t1_survived and t2_survived:
                    winning_team = 2
                elif t1_survived and not t2_survived:
                    winning_team = 1
                else:
                    winning_team = 3 - attacker_team if attacker_team else 0
                    winner = "DEF"

            if winner == "Unknown" and winning_team:
                if winning_team == attacker_team: winner = "ATK"
                else: winner = "DEF"

            # --- Calculate Events ---
            t1_val = sum(e['loadoutValue'] for e in round_ecos if player_map.get(e['playerId'],{}).get('team') == 1)
            t2_val = sum(e['loadoutValue'] for e in round_ecos if player_map.get(e['playerId'],{}).get('team') == 2)
            
            t1_loadout = get_loadout_bucket(t1_val)
            t2_loadout = get_loadout_bucket(t2_val)

            alive = {
                1: {"Duelist": 0, "Controller": 0, "Initiator": 0, "Sentinel": 0, "Unknown": 0},
                2: {"Duelist": 0, "Controller": 0, "Initiator": 0, "Sentinel": 0, "Unknown": 0}
            }
            for pid, info in player_map.items():
                alive[info['team']][info['role']] += 1

            spike_planted = 0
            plant_time = 0
            
            for ev in r_events:
                current_time = ev['roundTimeMillis']
                
                if ev.get('killId') is not None:
                    victim = ev.get('referencePlayerId')
                    if not victim: victim = ev.get('playerId') 
                    
                    if victim in player_map:
                        v_info = player_map[victim]
                        if alive[v_info['team']][v_info['role']] > 0:
                            alive[v_info['team']][v_info['role']] -= 1
                
                if ev.get('eventType') == 'plant':
                    spike_planted = 1
                    plant_time = current_time
                
                if spike_planted:
                    time_left = 45000 - (current_time - plant_time)
                else:
                    time_left = 100000 - current_time
                if time_left < 0: time_left = 0

                row = {
                    "match_id": mid,
                    "round_number": r_num,
                    "t1_score": t1_score,
                    "t2_score": t2_score,
                    "time_remaining_s": int(time_left / 1000),
                    "spike_planted": spike_planted,
                    "t1_loadout_value": t1_loadout,
                    "t2_loadout_value": t2_loadout,
                    
                    "t1_duelists_alive": alive[1]["Duelist"],
                    "t1_controllers_alive": alive[1]["Controller"],
                    "t1_initiators_alive": alive[1]["Initiator"],
                    "t1_sentinels_alive": alive[1]["Sentinel"],
                    
                    "t2_duelists_alive": alive[2]["Duelist"],
                    "t2_controllers_alive": alive[2]["Controller"],
                    "t2_initiators_alive": alive[2]["Initiator"],
                    "t2_sentinels_alive": alive[2]["Sentinel"],
                    
                    "round_winner": winner
                }
                all_rows.append(row)
            
            if winning_team == 1: t1_score += 1
            elif winning_team == 2: t2_score += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Processed {len(all_rows)} events into {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process matches for training or test data.")
    parser.add_argument("mode", choices=["training", "test"], help="Mode: 'training' or 'test'")
    args = parser.parse_args()
    
    process_matches(args.mode)
