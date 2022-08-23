from diffusers import StableDiffusionPipeline
import torch, fire, unicodedata

TOKEN = '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' # my token, please generate your own if you copy this: https://huggingface.co/settings/tokens

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    value = unicode(re.sub('[-\s]+', '-', value))
    # ...
    return value

def go(prompt):

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=TOKEN)
    pipe.to('cuda')

    for i in range(5):
        image = pipe(prompt)["sample"][0]
        image.save(f'{slugify(prompt)}-{i}.png')


if __name__ == '__main__':
    fire.Fire(go)