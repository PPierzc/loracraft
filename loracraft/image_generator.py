import sys
sys.append('../')


import json
import torch
import os
from loracraft.pipeline import get_pipeline
from loracraft.generator_utils import save_image
import wandb
from loracraft.api import query_runs


wandb.login()

with open ("/content/lora_db.json", 'r') as f:
    lora_db = json.load(f)

pipe = get_pipeline()


def run_lora(config):
    # Download required LoRAs
    for lora in config["loras"]:

        lora_path = f"./{lora}.safetensors"
        lora_wget = lora_db[lora]["wget_link"]
        command = f'wget -O {lora_path} "{lora_wget}'

        if not os.path.exists(lora_path):
            os.system(command)
        try:
            pipe.load_lora_weights(
                lora_path,
                adapter_name=lora
            )
        except Exception as e:
            print(e)

    pipe.set_adapters(config["loras"],
        adapter_weights=config["weights"][
            :len(config["loras"])
        ]
    )

    run = wandb.init(
        entity='ppierzc',
        project='loracraft',
        config=config
    )

    runs = query_runs({
        "loras": config["loras"],
        "prompts": config['prompts']
    })

    if len(runs) > 0:
        print(f"Duplicates found for {config}")
        return

    os.makedirs(config['images_path'], exist_ok=True)

    for trial in range(config['trials']):
        for prompt in config['prompts']:
            image = pipe(
                prompt=prompt,
                generator=torch.manual_seed(config['seed']), # Remove
                **config['pipeline_kwargs']
            ).images[0]
            save_image(image, config, prompt=prompt, trial=trial, run=run)

    run.finish()

configs = json.load(open(sys.argv[1], 'r'))

for config in configs:
    run_lora(config)
