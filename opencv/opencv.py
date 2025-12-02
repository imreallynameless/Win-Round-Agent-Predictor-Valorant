# This file contains the function definitions needed to scrape VCT Champions gameplay screenshots to acquire the data necessary for the classifier model

import cv2
import easyocr
import os
import numpy as np

reader = easyocr.Reader(['en'])
img = cv2.imread("screenshots/val.png")

# =============================================================================
# CREDIT COST DICTIONARIES
# =============================================================================

WEAPON_COSTS = {
    # Sidearms
    "classic": 0,
    "shorty": 300,
    "frenzy": 450,
    "ghost": 500,
    "sheriff": 800,
    
    # SMGs
    "stinger": 950,
    "spectre": 1600,
    
    # Shotguns
    "bucky": 850,
    "judge": 1850,
    
    # Rifles
    "bulldog": 2050,
    "guardian": 2250,
    "phantom": 2900,
    "vandal": 2900,
    
    # Snipers
    "marshal": 950,
    "outlaw": 2400,
    "operator": 4700,
    
    # Machine Guns
    "ares": 1600,
    "odin": 3200,
}

SHIELD_COSTS = {
    "heavy": 1000,   # Heavy Shield (26-50 armor)
    "light": 400,    # Light Shield (1-25 armor)
    "none": 0,       # No shield (0 armor)
}

# Ability costs per agent (C ability, Q ability) - signature E is free, X is ultimate
AGENT_ABILITY_COSTS = {
    # Duelists
    "Iso": {"C": 200, "Q": 200},
    "Jett": {"C": 200, "Q": 150},
    "Neon": {"C": 200, "Q": 200},
    "Phoenix": {"C": 200, "Q": 250},
    "Raze": {"C": 200, "Q": 300},
    "Reyna": {"C": 200, "Q": 200},
    "Waylay": {"C": 200, "Q": 250},
    "Yoru": {"C": 200, "Q": 250},
    
    # Initiators
    "Breach": {"C": 200, "Q": 250},
    "Fade": {"C": 200, "Q": 250},
    "Gekko": {"C": 200, "Q": 250},
    "KAYO": {"C": 200, "Q": 250},
    "Skye": {"C": 250, "Q": 250},
    "Sova": {"C": 250, "Q": 300},
    "Tejo": {"C": 200, "Q": 250},
    
    # Sentinels
    "Chamber": {"C": 200, "Q": 150},
    "Cypher": {"C": 200, "Q": 250},
    "Deadlock": {"C": 200, "Q": 250},
    "Killjoy": {"C": 200, "Q": 250},
    "Sage": {"C": 200, "Q": 250},
    "Vyse": {"C": 200, "Q": 250},
    
    # Controllers
    "Astra": {"C": 200, "Q": 200},
    "Brimstone": {"C": 250, "Q": 250},
    "Clove": {"C": 200, "Q": 250},
    "Harbor": {"C": 250, "Q": 250},
    "Omen": {"C": 150, "Q": 300},
    "Viper": {"C": 200, "Q": 200},
}

# Storing the x coordinate, y coordinate, width, and height of all regions of interest
t1_health_xywh = [
    (280, 553, 50, 30),
    (280, 660, 50, 30),
    (280, 765, 50, 30),
    (280, 872, 50, 30),
    (280, 978, 50, 30)
]

t2_health_xywh = [
    (1600, 553, 50, 30),
    (1600, 660, 50, 30),
    (1600, 765, 50, 30),
    (1600, 872, 50, 30),
    (1600, 978, 50, 30)
]

t1_players_agent_xywh = [
    (18, 549, 60, 38),
    (18, 653, 60, 38),
    (18, 759, 60, 38),
    (18, 865, 60, 38),
    (18, 971, 60, 38)
]

t2_players_agent_xywh = [
    (1840, 549, 60, 38),
    (1840, 654, 60, 38),
    (1840, 759, 60, 38),
    (1840, 864, 60, 38),
    (1840, 970, 60, 38)
]

# Weapon icon ROI coordinates (x, y, w, h) for each player slot
# Calibrated from val.png using coords.py
t1_weapon_xywh = [
    (201, 603, 84, 32),
    (201, 710, 87, 27),
    (200, 816, 80, 31),
    (202, 919, 79, 31),
    (201, 1023, 82, 38)
]

t2_weapon_xywh = [
    (1638, 603, 87, 28),
    (1641, 709, 84, 29),
    (1641, 812, 81, 34),
    (1640, 922, 87, 28),
    (1629, 1034, 93, 30)
]

# Shield value ROI coordinates (the small badge showing 25 or 50)
t1_shield_xywh = [
    (185, 430, 30, 20),
    (185, 510, 30, 20),
    (185, 590, 30, 20),
    (185, 670, 30, 20),
    (185, 750, 30, 20)
]

t2_shield_xywh = [
    (1700, 430, 30, 20),
    (1700, 510, 30, 20),
    (1700, 590, 30, 20),
    (1700, 670, 30, 20),
    (1700, 750, 30, 20)
]

# Ability icon ROI coordinates - C ability (first) and Q ability (second)
# Each ability icon is about 20x20 pixels, arranged in a row
t1_ability_c_xywh = [
    (68, 458, 20, 20),
    (68, 538, 20, 20),
    (68, 618, 20, 20),
    (68, 698, 20, 20),
    (68, 778, 20, 20)
]

t1_ability_q_xywh = [
    (93, 458, 20, 20),
    (93, 538, 20, 20),
    (93, 618, 20, 20),
    (93, 698, 20, 20),
    (93, 778, 20, 20)
]

t2_ability_c_xywh = [
    (1368, 458, 20, 20),
    (1368, 538, 20, 20),
    (1368, 618, 20, 20),
    (1368, 698, 20, 20),
    (1368, 778, 20, 20)
]

t2_ability_q_xywh = [
    (1343, 458, 20, 20),
    (1343, 538, 20, 20),
    (1343, 618, 20, 20),
    (1343, 698, 20, 20),
    (1343, 778, 20, 20)
]

# Helper functions

def load_agent_icons():
    """
    purpose: load webp images of official agent icons to be used to classify on screenshots

    output:
    two dictionaries, each mapping each agent name to the numpy array representation of their icon, first dictionary maps to normal orientation, second dictionary maps to flipped icons
    """
    icon_folder = "icons/agents/"
    templates1 = {}
    templates2 = {}

    for fname in os.listdir(icon_folder):
        if fname.endswith(".webp"):
            # Remove _icon from end of each file name to get agent name
            name = os.path.splitext(fname)[0][:-5]

            img = cv2.imread(os.path.join(icon_folder, fname), cv2.IMREAD_UNCHANGED)

            templates1[name] = img.astype(np.uint8)
            # Store horizontal flip for team 2 agent icons
            templates2[name] = cv2.flip(img, 1).astype(np.uint8)

    return templates1, templates2

def load_gun_icons():
    """
    purpose: load webp images of gun icons to be used for weapon detection

    output:
    two dictionaries mapping gun name to numpy array representation (normal and flipped)
    """
    icon_folder = "icons/guns/"
    templates1 = {}
    templates2 = {}

    for fname in os.listdir(icon_folder):
        if fname.endswith(".webp"):
            name = os.path.splitext(fname)[0].lower()
            img = cv2.imread(os.path.join(icon_folder, fname), cv2.IMREAD_UNCHANGED)
            if img is not None:
                templates1[name] = img.astype(np.uint8)
                templates2[name] = cv2.flip(img, 1).astype(np.uint8)

    return templates1, templates2

def load_shield_icons():
    """
    purpose: load webp images of shield icons for shield detection

    output:
    two dictionaries mapping shield type to numpy array representation (normal and flipped)
    """
    icon_folder = "icons/shields/"
    templates1 = {}
    templates2 = {}

    shield_values = {
        "light_shields": 25,
        "heavy_shields": 50,
        "regen_shields": 50,  # Treat regen as heavy for credit purposes
    }

    for fname in os.listdir(icon_folder):
        if fname.endswith(".webp"):
            name = os.path.splitext(fname)[0].lower()
            img = cv2.imread(os.path.join(icon_folder, fname), cv2.IMREAD_UNCHANGED)
            if img is not None:
                shield_val = shield_values.get(name, 0)
                templates1[shield_val] = img.astype(np.uint8)
                templates2[shield_val] = cv2.flip(img, 1).astype(np.uint8)

    return templates1, templates2

def match_icon_sift(roi, templates, min_matches=1):
    """
    Further explanation found in report.

    purpose: We need a way to classify the agent icons from the HUD and match them with an existing set of agent icons. This function makes use of openCV's SIFT to identify the agents from the HUD

    inputs:
    roi- the region from the screenshot containing the player's agent as displayed on the HUD
    templates- a dictionary containing mappings of agent name to numpy array representation of the official agent icon webp
    min_matches- calibration for the number of feature matches required to consider two images as the same icon

    output:
    the name of the most likely agent match
    """
    roi = roi.astype(np.uint8)
    
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.equalizeHist(roi_gray)
    
    sift = cv2.SIFT_create(nfeatures=500)
    kp1, des1 = sift.detectAndCompute(roi_gray, None)
    
    if des1 is None:
        return None
    
    best_match = None
    best_score = 0
    
    bf = cv2.BFMatcher()
    
    for name, template in templates.items():
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.equalizeHist(tpl_gray)
        
        kp2, des2 = sift.detectAndCompute(tpl_gray, None)
        
        if des2 is None:
            continue
        
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) >= min_matches:
            match_count = len(good_matches)
            avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
            score = match_count / (1 + avg_distance / 100)
            
            if score > best_score:
                best_score = score
                best_match = name
    
    return best_match

def is_red_or_blue(roi):
    """
    purpose: determine whether a ROI is majority red or blue, necessary since the VCT HUD differenciates the attacker and defender teams by colour

    inputs:
    roi- the region of interest from the image that we are looking to identify colour from, which would be the top of the HUD where team 1's score is

    output:
    string 'red' if the roi is dominantly red, otherwise string 'blue'
    """
    # Calculate average color (OpenCV uses BGR order)
    avg_color = cv2.mean(roi)[:3]  # (B, G, R)
    blue_val = avg_color[0]
    red_val = avg_color[2]
    
    return 'red' if red_val > blue_val else 'blue'

# Functions that scrape different ROI for data

def get_time_remaining(img):
    """
    purpose: to get the remaining time left in the round, and if not applicable, determine that the spike has been planted

    inputs:
    img- the gameplay screenshot loaded with openCV

    output:
    an integer value representing the seconds remaining in the round if the spike has not yet been planted, or None if the spike has been planted
    """
    timeleft_xywh = (918, 23, 80, 42)

    x, y, w, h = timeleft_xywh
    roi = img[y:y+h, x:x+w]
    results = reader.readtext(roi)

    if not results:
        return None

    # The return value of reader.readtext returns an array of tuples, but we are only expecting one entry so can hardcode the indexing to get the text contents
    minutes, seconds = results[0][1].split(".")
    time_remaining = int(minutes) * 60 + int(seconds)
    return time_remaining

def get_players_alive(img, t):
    """
    purpose: to identify which players (on which positions slotted in the HUD) are alive

    inputs:
    img- the gameplay screenshot loaded with openCV
    t- the team to get the alive players for (1/2)

    output:
    an array of length 5, at each index i contains 0 if the ith player is eliminated, 1 if they are still alive
    """
    if t != 1 and t != 2:
        return
    res = []
    xywh = []
    if t == 1:
        xywh = t1_health_xywh
    else:
        xywh = t2_health_xywh

    for (x, y, w, h) in xywh:
        roi = img[y:y+h, x:x+w]
        results = reader.readtext(roi)
        if results:
            res.append(1)
        else:
            res.append(0)
    return res

def get_alive_player_agents(img, t, players_alive, templates):
    """
    purpose: to scrape the image for which agents are being played by the alive players on a team (team1 leftside / team2 rightside)

    inputs:
    img- the gameplay screenshot loaded with openCV
    t- the team to get the alive player agents for (1/2)
    players_alive- an array of length 5, at each index i contains 0 if the ith player is eliminated, 1 if they are still alive
    templates- a dictionary of agent name mapped to numpy array representation of the agent's icon

    output:
    an array containing the names of the agents played by the alive players
    """
    if t != 1 and t != 2:
        return
    xywh = []
    if t == 1:
        xywh = t1_players_agent_xywh
    else:
        xywh = t2_players_agent_xywh

    res = []
    curr = 0
    for (x, y, w, h) in xywh:
        if players_alive[curr] == 1:
            roi = img[y:y+h, x:x+w]
            res.append(match_icon_sift(roi, templates))
            curr += 1
    
    return res

def get_roles_alive(agents_alive):
    """
    purpose: to convert agent names to their respective role, and get total alive role counts

    inputs:
    agents_alive- an array containing the names of the agents currently alive

    outputs:
    four separate integer values, representing duelists_alive, initiators_alive, sentinels_alive, controllers_alive respectively
    """
    duelists = ["Iso", "Jett", "Neon", "Phoenix", "Raze", "Reyna", "Waylay", "Yoru"]
    initiators = ["Breach", "Fade", "Gekko", "KAYO", "Skye", "Sova", "Tejo"]
    sentinels = ["Chamber", "Cypher", "Deadlock", "Killjoy", "Sage", "Vyse"]
    controllers = ["Astra", "Brimstone", "Clove", "Harbor", "Omen", "Viper"]

    duelists_alive = 0
    initiators_alive = 0
    sentinels_alive = 0
    controllers_alive = 0

    for agent in agents_alive:
        if agent in duelists:
            duelists_alive += 1
        elif agent in initiators:
            initiators_alive += 1
        elif agent in sentinels:
            sentinels_alive += 1
        elif agent in controllers:
            controllers_alive += 1
    
    return duelists_alive, initiators_alive, sentinels_alive, controllers_alive

def get_atk_team(img):
    (x, y, w, h) = (846, 28, 15, 20)
    roi = img[y:y+h, x:x+w]
    if is_red_or_blue(roi) == 'red':
        return 1
    return 2

def get_player_weapon(img, t, player_idx, templates):
    """
    purpose: identify the weapon a player is holding from the HUD

    inputs:
    img- the gameplay screenshot
    t- team (1 or 2)
    player_idx- index of the player (0-4)
    templates- dictionary of gun templates

    output:
    weapon name string or None
    """
    if t == 1:
        xywh = t1_weapon_xywh
    else:
        xywh = t2_weapon_xywh
    
    x, y, w, h = xywh[player_idx]
    roi = img[y:y+h, x:x+w]
    
    return match_icon_sift(roi, templates, min_matches=1)

def is_ability_available(img, t, player_idx, ability):
    """
    purpose: detect if an ability is available (bright) or not (dim/greyed out)

    inputs:
    img- the gameplay screenshot
    t- team (1 or 2)
    player_idx- index of the player (0-4)
    ability- "C" or "Q"

    output:
    True if ability is available (bright), False if not (dim)
    """
    # Select the correct ROI coordinates
    if t == 1:
        if ability == "C":
            xywh = t1_ability_c_xywh
        else:
            xywh = t1_ability_q_xywh
    else:
        if ability == "C":
            xywh = t2_ability_c_xywh
        else:
            xywh = t2_ability_q_xywh
    
    x, y, w, h = xywh[player_idx]
    roi = img[y:y+h, x:x+w]
    
    # Convert to grayscale and calculate average brightness
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # Threshold: if brightness > 80, ability is available (on)
    # This threshold may need calibration based on actual screenshots
    return avg_brightness > 80

def get_player_shield(img, t, player_idx):
    """
    purpose: get the shield type for a player based on armor value from HUD

    inputs:
    img- the gameplay screenshot
    t- team (1 or 2)
    player_idx- index of the player (0-4)

    output:
    string shield type: "heavy" (26-50), "light" (1-25), or "none" (0)
    """
    if t == 1:
        xywh = t1_shield_xywh
    else:
        xywh = t2_shield_xywh
    
    x, y, w, h = xywh[player_idx]
    roi = img[y:y+h, x:x+w]
    
    results = reader.readtext(roi)
    if results:
        text = results[0][1]
        # Extract numeric value from OCR result
        numeric = ''.join(filter(str.isdigit, text))
        if numeric:
            armor_value = int(numeric)
            if armor_value >= 26:
                return "heavy"
            elif armor_value >= 1:
                return "light"
    return "none"

def get_team_credits(img, t, players_alive, agents_alive, gun_templates):
    """
    purpose: calculate total credits spent by a team on weapons, shields, and abilities

    inputs:
    img- the gameplay screenshot
    t- team (1 or 2)
    players_alive- array indicating which players are alive
    agents_alive- array of agent names for alive players
    gun_templates- dictionary of gun icon templates

    output:
    tuple of (weapon_credits, shield_credits, ability_credits, total_credits)
    """
    weapon_credits = 0
    shield_credits = 0
    ability_credits = 0
    agent_idx = 0
    
    # For debugging - track what's detected
    weapons_detected = []
    shields_detected = []
    abilities_detected = []
    
    for player_idx, alive in enumerate(players_alive):
        if alive == 1:
            # Weapon credits
            weapon = get_player_weapon(img, t, player_idx, gun_templates)
            if weapon:
                cost = WEAPON_COSTS.get(weapon, 0)
                weapon_credits += cost
                weapons_detected.append(f"{weapon}({cost})")
            else:
                weapons_detected.append("None(0)")
            
            # Shield credits (heavy=26-50, light=1-25, none=0)
            shield_type = get_player_shield(img, t, player_idx)
            cost = SHIELD_COSTS.get(shield_type, 0)
            shield_credits += cost
            shields_detected.append(f"{shield_type}({cost})")
            
            # Ability credits - only count if ability is available (on)
            if agent_idx < len(agents_alive) and agents_alive[agent_idx]:
                agent = agents_alive[agent_idx]
                if agent in AGENT_ABILITY_COSTS:
                    agent_ability_costs = AGENT_ABILITY_COSTS[agent]
                    player_abilities = []
                    # Check C ability
                    c_on = is_ability_available(img, t, player_idx, "C")
                    if c_on:
                        cost = agent_ability_costs.get("C", 0)
                        ability_credits += cost
                        player_abilities.append(f"C({cost})")
                    # Check Q ability
                    q_on = is_ability_available(img, t, player_idx, "Q")
                    if q_on:
                        cost = agent_ability_costs.get("Q", 0)
                        ability_credits += cost
                        player_abilities.append(f"Q({cost})")
                    abilities_detected.append(f"{agent}:[{','.join(player_abilities) if player_abilities else 'none'}]")
            
            agent_idx += 1
    
    total_credits = weapon_credits + shield_credits + ability_credits
    
    # Print debug info
    print(f"  Team {t} Weapons: {weapons_detected} = {weapon_credits}")
    print(f"  Team {t} Shields: {shields_detected} = {shield_credits}")
    print(f"  Team {t} Abilities: {abilities_detected} = {ability_credits}")
    
    return weapon_credits, shield_credits, ability_credits, total_credits

# Aggregate scraped data to be fed into the classifier model

time_remaining = get_time_remaining(img)
print(f"Time remaining: {time_remaining}")

spike_planted = False
if not time_remaining:
    spike_planted = True

t1_players_alive = get_players_alive(img, 1)
t2_players_alive = get_players_alive(img, 2)
agent_templates1, agent_templates2 = load_agent_icons()
t1_agents_alive = get_alive_player_agents(img, 1, t1_players_alive, agent_templates1)
t2_agents_alive = get_alive_player_agents(img, 2, t2_players_alive, agent_templates2)

t1_duelists_alive, t1_initiators_alive, t1_sentinels_alive, t1_controllers_alive = get_roles_alive(t1_agents_alive)
t2_duelists_alive, t2_initiators_alive, t2_sentinels_alive, t2_controllers_alive = get_roles_alive(t2_agents_alive)

# Load gun templates for weapon detection
gun_templates1, gun_templates2 = load_gun_icons()

# Calculate team credits (now returns breakdown)
print("\n=== CREDIT BREAKDOWN ===")
t1_weapon_credits, t1_shield_credits, t1_ability_credits, t1_total_credits = get_team_credits(img, 1, t1_players_alive, t1_agents_alive, gun_templates1)
print(f"  Team 1 TOTAL: {t1_total_credits}")
print()
t2_weapon_credits, t2_shield_credits, t2_ability_credits, t2_total_credits = get_team_credits(img, 2, t2_players_alive, t2_agents_alive, gun_templates2)
print(f"  Team 2 TOTAL: {t2_total_credits}")
print("========================\n")

# Currently t1 and t2 refer to the teams on the left and right, need to convert to attacking and defending team by identifying colours
atk_team = get_atk_team(img)

if atk_team == 1:
    atk_duelists_alive, atk_initiators_alive, atk_sentinels_alive, atk_controllers_alive = t1_duelists_alive, t1_initiators_alive, t1_sentinels_alive, t1_controllers_alive
    def_duelists_alive, def_initiators_alive, def_sentinels_alive, def_controllers_alive = t2_duelists_alive, t2_initiators_alive, t2_sentinels_alive, t2_controllers_alive
    atk_weapon_credits, atk_shield_credits, atk_ability_credits, atk_credits = t1_weapon_credits, t1_shield_credits, t1_ability_credits, t1_total_credits
    def_weapon_credits, def_shield_credits, def_ability_credits, def_credits = t2_weapon_credits, t2_shield_credits, t2_ability_credits, t2_total_credits
else:
    atk_duelists_alive, atk_initiators_alive, atk_sentinels_alive, atk_controllers_alive = t2_duelists_alive, t2_initiators_alive, t2_sentinels_alive, t2_controllers_alive
    def_duelists_alive, def_initiators_alive, def_sentinels_alive, def_controllers_alive = t1_duelists_alive, t1_initiators_alive, t1_sentinels_alive, t1_controllers_alive
    atk_weapon_credits, atk_shield_credits, atk_ability_credits, atk_credits = t2_weapon_credits, t2_shield_credits, t2_ability_credits, t2_total_credits
    def_weapon_credits, def_shield_credits, def_ability_credits, def_credits = t1_weapon_credits, t1_shield_credits, t1_ability_credits, t1_total_credits

print("Atk agents alive: " + str(t1_agents_alive))
print(f"Atk - Duelists: {t1_duelists_alive}")
print(f"Atk - Initiators: {t1_initiators_alive}")
print(f"Atk - Sentinels: {t1_sentinels_alive}")
print(f"Atk - Controllers: {t1_controllers_alive}")
print(f"Atk - Weapon Credits: {atk_weapon_credits}")
print(f"Atk - Shield Credits: {atk_shield_credits}")
print(f"Atk - Ability Credits: {atk_ability_credits}")
print(f"Atk - Total Credits: {atk_credits}")
print("======================")
print("Def agents alive: " + str(t2_agents_alive))
print(f"Def - Duelists: {t2_duelists_alive}")
print(f"Def - Initiators: {t2_initiators_alive}")
print(f"Def - Sentinels: {t2_sentinels_alive}")
print(f"Def - Controllers: {t2_controllers_alive}")
print(f"Def - Weapon Credits: {def_weapon_credits}")
print(f"Def - Shield Credits: {def_shield_credits}")
print(f"Def - Ability Credits: {def_ability_credits}")
print(f"Def - Total Credits: {def_credits}")
