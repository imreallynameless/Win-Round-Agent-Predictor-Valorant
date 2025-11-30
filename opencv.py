# This file contains the function definitions needed to scrape VCT Champions gameplay screenshots to acquire the data necessary for the classifier model

import cv2
import easyocr
import os
import numpy as np

reader = easyocr.Reader(['en'])
img = cv2.imread("screenshots/val3.png")

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
    sentinels = ["Chamber", "Cypher", "Deadlock", "Killjoy", "Sage", "Veto", "Vyse"]
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

# Currently t1 and t2 refer to the teams on the left and right, need to convert to attacking and defending team by identifying colours
atk_team = get_atk_team(img)

if atk_team == 1:
    atk_duelists_alive, atk_initiators_alive, atk_sentinels_alive, atk_controllers_alive = t1_duelists_alive, t1_initiators_alive, t1_sentinels_alive, t1_controllers_alive
    def_duelists_alive, def_initiators_alive, def_sentinels_alive, def_controllers_alive = t2_duelists_alive, t2_initiators_alive, t2_sentinels_alive, t2_controllers_alive
else:
    atk_duelists_alive, atk_initiators_alive, atk_sentinels_alive, atk_controllers_alive = t2_duelists_alive, t2_initiators_alive, t2_sentinels_alive, t2_controllers_alive
    def_duelists_alive, def_initiators_alive, def_sentinels_alive, def_controllers_alive = t1_duelists_alive, t1_initiators_alive, t1_sentinels_alive, t1_controllers_alive

print("Atk agents alive: " + str(t1_agents_alive))
print(f"Atk - Duelists: {t1_duelists_alive}")
print(f"Atk - Initiators: {t1_initiators_alive}")
print(f"Atk - Sentinels: {t1_sentinels_alive}")
print(f"Atk - Controllers: {t1_controllers_alive}")
print("======================")
print("Def agents alive: " + str(t2_agents_alive))
print(f"Def - Duelists: {t2_duelists_alive}")
print(f"Def - Initiators: {t2_initiators_alive}")
print(f"Def - Sentinels: {t2_sentinels_alive}")
print(f"Def - Controllers: {t2_controllers_alive}")
