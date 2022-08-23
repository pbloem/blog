from diffusers import StableDiffusionPipeline
import torch, fire, unicodedata, re

TOKEN = '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' # my token, please generate your own if you copy this: https://huggingface.co/settings/tokens


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def go(prompt):

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=TOKEN)
    pipe.to('cuda')

    for i in range(5):
        image = pipe(prompt)["sample"][0]
        image.save(f'{slugify(prompt)}-{i}.png')


if __name__ == '__main__':
    fire.Fire(go)