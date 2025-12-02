# OpenCV Screenshot Analysis Module

This module extracts real-time game state data from VCT (VALORANT Champions Tour) gameplay screenshots using computer vision techniques. The extracted data is designed to feed into a round win prediction classifier model.

## Overview

The system analyzes VCT broadcast HUD overlays to extract:
- **Round timer** (or spike planted status)
- **Players alive** on each team
- **Agent compositions** for both teams
- **Role distributions** (Duelists, Initiators, Sentinels, Controllers)
- **Attacker/Defender team identification**

## Architecture

```
opencv/
├── opencv.py           # Main extraction pipeline
├── coords.py           # Coordinate helper utility
├── icons/
│   └── agents/         # 28 official agent icon templates (.webp)
└── screenshots/        # Sample VCT broadcast screenshots
```

## How It Works

### 1. Region of Interest (ROI) Extraction

The VCT broadcast HUD has fixed positions for player information. The module defines pixel coordinates for:

| Region | Purpose | Location |
|--------|---------|----------|
| `t1_health_xywh` | Team 1 (left) player health values | Left sidebar |
| `t2_health_xywh` | Team 2 (right) player health values | Right sidebar |
| `t1_players_agent_xywh` | Team 1 agent icons | Left sidebar |
| `t2_players_agent_xywh` | Team 2 agent icons | Right sidebar |
| Timer region | Round countdown | Top center |
| Team color region | Attacker/Defender identification | Top HUD |

### 2. Agent Icon Recognition (SIFT Algorithm)

The module uses OpenCV's **SIFT (Scale-Invariant Feature Transform)** algorithm to match agent icons from the HUD against official agent icon templates.

**Process:**
1. Load all 28 agent icon templates (normal + horizontally flipped for right-side players)
2. Extract SIFT keypoints and descriptors from both the ROI and templates
3. Use Brute-Force matching with k-NN (k=2)
4. Apply Lowe's ratio test (0.7 threshold) to filter good matches
5. Score matches by count and average distance
6. Return the best-matching agent name

**Why SIFT?**
- Scale-invariant: handles different icon sizes between HUD and templates
- Rotation-invariant: robust to slight positioning variations
- Feature-based: works even with partial occlusions or HUD effects

### 3. Player Alive Detection (EasyOCR)

Uses **EasyOCR** to detect health values in player HUD slots:
- If OCR detects text (health number) → player is alive
- If no text detected → player is eliminated

### 4. Timer & Spike Status Extraction

Reads the round timer from the top-center HUD:
- Returns seconds remaining if timer is visible
- Returns `None` if spike is planted (timer replaced by defuse progress)

### 5. Team Side Detection

Determines which team is attacking/defending by analyzing the color of the team indicator:
- **Red** = Attackers
- **Blue** = Defenders

## Key Functions

### `load_agent_icons()`
Loads all agent icon templates from `icons/agents/`, returning both normal and horizontally-flipped versions (Team 2's icons face the opposite direction).

### `match_icon_sift(roi, templates, min_matches=1)`
Performs SIFT-based template matching to identify which agent is displayed in a given ROI.

### `get_time_remaining(img)`
Extracts the round timer or detects if spike has been planted.

### `get_players_alive(img, t)`
Returns a 5-element array indicating which players on team `t` (1 or 2) are alive.

### `get_alive_player_agents(img, t, players_alive, templates)`
Returns the list of agent names for alive players on a team.

### `get_roles_alive(agents_alive)`
Converts agent names to role counts:
- **Duelists**: Iso, Jett, Neon, Phoenix, Raze, Reyna, Waylay, Yoru
- **Initiators**: Breach, Fade, Gekko, KAY/O, Skye, Sova, Tejo
- **Sentinels**: Chamber, Cypher, Deadlock, Killjoy, Sage, Veto, Vyse
- **Controllers**: Astra, Brimstone, Clove, Harbor, Omen, Viper

### `get_atk_team(img)`
Returns which team (1 or 2) is on the attacking side based on HUD colors.

## Output Data

The module aggregates extracted data into features for the prediction model:

```python
# Example output variables
time_remaining          # int: seconds left (or None if spike planted)
spike_planted           # bool: True if spike is down
atk_duelists_alive      # int: attacking team duelist count
atk_initiators_alive    # int: attacking team initiator count
atk_sentinels_alive     # int: attacking team sentinel count
atk_controllers_alive   # int: attacking team controller count
def_duelists_alive      # int: defending team duelist count
def_initiators_alive    # int: defending team initiator count
def_sentinels_alive     # int: defending team sentinel count
def_controllers_alive   # int: defending team controller count
atk_credits             # int: total credits invested by attacking team
def_credits             # int: total credits invested by defending team
```

## Credit Calculation

The module calculates total credits invested by each team by summing:

### Weapon Costs
| Category | Weapon | Cost |
|----------|--------|------|
| Sidearm | Classic | 0 |
| Sidearm | Shorty | 150 |
| Sidearm | Frenzy | 450 |
| Sidearm | Ghost | 500 |
| Sidearm | Sheriff | 800 |
| SMG | Stinger | 950 |
| SMG | Spectre | 1,600 |
| Shotgun | Bucky | 850 |
| Shotgun | Judge | 1,850 |
| Rifle | Bulldog | 2,050 |
| Rifle | Guardian | 2,250 |
| Rifle | Phantom | 2,900 |
| Rifle | Vandal | 2,900 |
| Sniper | Marshal | 950 |
| Sniper | Outlaw | 2,400 |
| Sniper | Operator | 4,700 |
| Machine Gun | Ares | 1,600 |
| Machine Gun | Odin | 3,200 |

### Shield Costs
Based on current armor HP value displayed in HUD:
| Armor Value | Shield Type | Cost |
|-------------|-------------|------|
| 0 | None | 0 |
| 1-25 | Light | 400 |
| 26-50 | Heavy | 1,000 |

### Ability Costs
Each agent has purchasable C and Q abilities (signature E ability is free). Costs range from 150-300 credits per ability depending on the agent.

## Coordinate Helper (`coords.py`)

A utility script for determining ROI boundaries:
- Click anywhere on a screenshot to print the (x, y) coordinates
- Useful for calibrating ROI positions if HUD layouts change

## Sample Screenshots

The `screenshots/` folder contains test images demonstrating various game states:
- `val.png` - `val3.png`: Standard gameplay states
- `val4_operator.png`: Player with Operator equipped
- `val5_4v4.png`: 4v4 player situation
- `val6_5v4eco.png`: 5v4 with economy differential
- `val7_2v4pistolretake.png`: 2v4 retake with spike planted
- `val8_2v2retake.png`: 2v2 clutch situation

## Dependencies

```
opencv-python
easyocr
numpy
```

## Usage

```python
import cv2
from opencv import (
    load_agent_icons,
    get_time_remaining,
    get_players_alive,
    get_alive_player_agents,
    get_roles_alive,
    get_atk_team
)

# Load screenshot
img = cv2.imread("screenshots/val3.png")

# Extract game state
time_remaining = get_time_remaining(img)
t1_alive = get_players_alive(img, 1)
t2_alive = get_players_alive(img, 2)

templates1, templates2 = load_agent_icons()
t1_agents = get_alive_player_agents(img, 1, t1_alive, templates1)
t2_agents = get_alive_player_agents(img, 2, t2_alive, templates2)

# Get role counts
t1_roles = get_roles_alive(t1_agents)  # (duelists, initiators, sentinels, controllers)
t2_roles = get_roles_alive(t2_agents)
```

## Limitations

- **Resolution dependent**: ROI coordinates are calibrated for 1920x1080 VCT broadcast resolution
- **HUD-specific**: Designed for VCT official broadcast overlay; may not work with other stream layouts
- **Agent updates**: New agents require adding their icons to `icons/agents/` and updating role lists

## Integration with Prediction Model

The extracted features (role counts, spike status, etc.) are formatted to match the training data schema used by the Naive Bayes classifier in the `model/` directory.

