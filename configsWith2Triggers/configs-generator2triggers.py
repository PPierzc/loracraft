import json
import copy
from itertools import combinations

with open('../lora_db.json') as file:
    loras_db = json.load(file)

def combinations_2(loras, n):
    for lora in loras:
        for rest_of_loras in combinations(loras[1:], n - 1):
            yield [lora] + list(rest_of_loras)

config_template = {
    "loras": [],
    "weights": [],
    "prompts": [""],
    "trials": 1,
    'images_path': '',
    'seed': 42,
    'pipeline_kwargs': {
        'num_inference_steps': 1,
        "guidance_scale": 0,
    }
}

def generate_configs(loras_db, num_loras):
    loras = list(loras_db.keys())
    configs = []

    if num_loras == 0:
        for i in range(0, 10, 2):
            triggers_pair = loras_db[loras[i]]['triggers'][0] + ";" + loras_db[loras[i + 1]]['triggers'][0]

            config = copy.deepcopy(config_template)
            config['loras'] = []
            config['prompts'] = triggers_pair
            config['images_path'] = f"./output/no-loras-{loras[i].lower()}-{loras[i + 1].lower()}/"

            configs.append(config)
    if num_loras == 1:
        for i in range(0, 10, 2):
            triggers_pair = loras_db[loras[i]]['triggers'][0] + ";" + loras_db[loras[i + 1]]['triggers'][0]

            config = copy.deepcopy(config_template)
            config['loras'] = [loras[i]]
            config['prompts'] = triggers_pair
            config['weights'] = [1]
            config['images_path'] = f"./output/{loras[i].lower()}/"

            config2 = copy.deepcopy(config_template)
            config2['loras'] = [loras[i + 1]]
            config2['prompts'] = triggers_pair
            config['weights'] = [1]
            config2['images_path'] = f"./output/{loras[i + 1].lower()}/"

            configs.append(config)
            configs.append(config2)
    if num_loras == 2:
        for i in range(0, 10, 2):
            triggers_pair = loras_db[loras[i]]['triggers'][0] + ";" + loras_db[loras[i + 1]]['triggers'][0]

            config = copy.deepcopy(config_template)
            config['loras'] = [loras[i], loras[i + 1]]
            config['prompts'] = triggers_pair
            config['weights'] = [1, 1]
            config['images_path'] = f"./output/{loras[i].lower()}-{loras[i + 1].lower()}/"

            configs.append(config)

    all_configs_path = f'./configs_with_2_triggers_with_{num_loras}_loras.json'
    with open(all_configs_path, 'w') as file:
        json.dump(configs, file, indent=4)

    print(f"All configurations saved to {all_configs_path}.")


num_loras = int(input("Enter the number of LoRAs to include in each configuration: "))
generate_configs(loras_db, num_loras)