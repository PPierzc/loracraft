import pandas as pd
import json


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df = df[["Lora Model", 'Link', 'Civit Tag', 'wget_link', 'trigger']]
    df = df.fillna("")
    db_dict = {}

    for idx, row in df.iterrows():
        name = row["Lora Model"].replace(" ", "")
        db_dict[name] = {
            "wget_link": row["wget_link"],
            "triggers": row['trigger'].split('|'),
            "category": row["Civit Tag"],
            "metadata": {
                "civil_link": row['Link']
            }
        }
    with open("../lora_db.json", "w") as f:
        json.dump(db_dict, f, indent=4)
