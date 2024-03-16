import json
import copy

# Load LoRAs configuration
with open('lora_db.json') as file:
    loras_db = json.load(file)

config_template = {
    "loras": [],
    "weights": [1, 1],
    "prompts": [""],
    "trials": 1,
    'images_path': '',
    'seed': 42,
    'pipeline_kwargs': {
        'num_inference_steps': 1,
        "guidance_scale": 0,
    }
}


def generate_configs(loras_db):
    loras_names = list(loras_db.keys())
    configs = []

    for i in range(len(loras_names)):
        if loras_names[i] == "Details":
            continue

        for j in range(len(loras_names)):
            if i == j or loras_names[j] == "Details":
                continue

            lora_main, lora_secondary = loras_names[i], loras_names[j]
            config = copy.deepcopy(config_template)
            config['loras'] = [lora_main, lora_secondary]
            config['prompts'] = [loras_db[lora_main]['triggers'][0]] if loras_db[lora_main]['triggers'] else [""]
            config['images_path'] = f'./output/{lora_main.lower()}-{lora_secondary.lower()}/'

            configs.append(config)

    all_configs_path = './all_configs.json'
    with open(all_configs_path, 'w') as configs_file:
        json.dump(configs, configs_file, indent=4)

    print(f"All configs saved to {all_configs_path}")


generate_configs(loras_db)

