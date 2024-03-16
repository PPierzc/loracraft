from diffusers import AutoPipelineForText2Image
import torch
import tomesd


def get_pipeline():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16,
                                                     variant="fp16")
    pipe.enable_xformers_memory_efficient_attention()
    tomesd.apply_patch(pipe, ratio=0.5)
    pipe.to("cuda")

    return pipe

if __name__ == '__main__':
    pipe = get_pipeline()