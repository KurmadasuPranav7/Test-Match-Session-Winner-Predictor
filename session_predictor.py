import json
import os
import pandas as pd


def extract_sessions_from_match(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    match_id = os.path.basename(filepath).split('.')[0]
    teams = data['info']['teams']
    innings_data = []
    inn_wickets = 0
    for i, inning in enumerate(data['innings'], start=1):
        inning_name = list(inning.keys())[0]
        inning_info = inning[inning_name]
        # batting_team = inning_info['team']
        batting_team = inning_info
        bowling_team = [t for t in teams if t != batting_team][0]

        runs, wickets = 0, 0
        session = 1
        over_count = 0

        for over in inning.get('overs', []):
            for delivery in over.get('deliveries', []):
                runs += delivery['runs']['total']
                if 'wickets' in delivery:
                    wickets += 1
                    inn_wickets += 1

            over_count += 1

            # Group by sessions (30 overs each)
            if over_count % 30 == 0 or inn_wickets == 10:
                run_rate = runs // 30
                if 3 <= run_rate < 4:
                    if wickets == 1 or wickets == 2:
                        session_winner = 1
                    elif wickets == 3:
                        session_winner = 0
                    elif wickets > 3:
                        session_winner = -1
                    else:
                        session_winner = 1
                elif 2 <= run_rate < 3:
                    if wickets == 1:
                        session_winner = 1
                    elif wickets == 2:
                        session_winner = 0
                    elif wickets > 2:
                        session_winner = -1
                    else:
                        session_winner = 1
                elif 4 <= run_rate < 5:
                    if wickets <= 3:
                        session_winner = 1
                    elif 3 < wickets < 5:
                        session_winner = 0
                    else:
                        session_winner = -1
                elif run_rate >= 5:
                    if wickets >= 5:
                        session_winner = -1
                    elif wickets == 4:
                        session_winner = 0
                    else:
                        session_winner = 1
                else:
                    session_winner = -1

                innings_data.append({
                    'match_id': match_id,
                    'inning_number': i,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'session_number': session,
                    'runs_scored': runs,
                    'wickets_fallen': wickets,
                    'session_winner': session_winner
                })
                session += 1
                runs, wickets = 0, 0  # reset for next session

        inn_wickets = 0

    return innings_data


# ---- Extract from all JSON files ----
folder_path = "D:\\Cric_Session_Predictor\\test_match_data"
# folder_path = "D:\\Cric_Session_Predictor\\test"
all_data = []

for file in os.listdir(folder_path):
    if file.endswith(".json"):
        all_data.extend(extract_sessions_from_match(os.path.join(folder_path, file)))

df = pd.DataFrame(all_data)
df.to_csv("session_data.csv", index=False)
print("Extracted", len(df), "sessions.")
