import os, os.path

import numpy as np
from datasets import load_dataset
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

# name of folder being converted
foldername = "Good Form"

if foldername=="Bad Form":
    filename = 'bad'
else:
    filename = 'good'


def demo(model, processor, tokenizer, imagenette, device):
    prompt = "a dog in the snow"
    inputs = tokenizer(prompt, return_tensors="pt")

    text_emb = model.get_text_features(**inputs)
    
    image = processor(
        text=None,
        images=imagenette[0]['image'],
        return_tensors="pt"
    )['pixel_values'].to(device)
    # print(image.shape)

    image_emb = model.get_image_features(image)
    # print(image_emb.shape)

def main():

    imagenette = load_dataset(
        foldername,
        'full_size',
        split='train',
        verification_mode='no_checks'
    )

    model_id = "openai/clip-vit-base-patch32"

    device = "cpu"
    with torch.inference_mode():
        model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # embed several images
    np.random.seed(0)
    # TODO use `np.random.choice` instead because we might want to avoid duplicates
    # sample_idx = np.random.randint(0, len(imagenette) + 1, 25).tolist()
    images = [imagenette[i]['image'] for i in range(len(os.listdir(foldername)))]
    print(len(images))

    batch_size = 16
    image_arr = None

    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i: i + batch_size]
        batch = processor(
            text=None,
            images=batch,
            return_tensors="pt",
            padding=True
        )['pixel_values'].to(device)

        with torch.inference_mode():
            batch_emb = model.get_image_features(pixel_values=batch)

        batch_emb = batch_emb.squeeze(0)
        batch_emb = batch_emb.cpu().detach().numpy()
        if image_arr is None:
            image_arr = batch_emb
        else:
            image_arr = np.concatenate((image_arr, batch_emb), axis=0)
    print(image_arr.shape)
    np.save(filename,image_arr)
    print(len(os.listdir(foldername)))
    print("First Embedding", end='')
    print(image_arr[0])
    print()

if __name__ == '__main__':
    main()
