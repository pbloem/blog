from diffusers import StableDiffusionPipeline
import torch

TOKEN = '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' # my token, please generate your own if you copy this: https://huggingface.co/settings/tokens

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=TOKEN)
pipe.to('cuda')

for i in range(5):
    prompt = "Something which is upside-down."

    image = pipe(prompt)["sample"][0]
    image.save(f'prompt{i}.png')