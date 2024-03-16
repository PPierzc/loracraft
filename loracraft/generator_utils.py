import os
import pandas as pd
import torch





def save_image(image, config, prompt=None, trial=None):
    images_path = config['images_path']
    image.save(f'{images_path}/{prompt}{trial}.png')


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