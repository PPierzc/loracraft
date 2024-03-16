import os
import pandas as pd
import torch
import wandb
from loracraft.api import log_images, query_runs, get_images_from_run
import wandb

def save_image(image, config, prompt=None, trial=None, run=None, rank=0):
    # images_path = config['images_path']
    # image.save(f'{images_path}/{prompt}_{trial}_{rank}.png')

    if run:
        log_images(run, [image])


def load_lora(model, config):
    steps = config['steps']
    os.makedirs(config['images_path'], exist_ok=True)

    adapters, weights = config['loras'], config['weights']
    for (adapter_id, adapter_path) in adapters.items():
        try:
            model.load_lora_weights(
                adapter_path,
                adapter_name=adapter_id
            )
        except Exception as e:
            print(e)

    model.set_adapters(adapters.keys(), adapter_weights=weights)

    for trial in range(config['trials']):
        for prompt in config['prompts']:
            image = model(
                prompt=prompt,
                numAutoPipelineForText2Image_inference_steps=steps,
                guidance_scale=0,
                generator=torch.manual_seed(config['seed']),
                **config['pipeline_kwargs']
                # cross_attention_kwargs={"scale": 1}
            ).images[0]

            save_image(image, config, prompt=prompt, trial=trial)