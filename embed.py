import os

import numpy as np
from datasets import load_dataset
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

def main(foldername, filename):
    print(f'Extracting images from {foldername}')

    # extracts images in full size from folders
    imagenette = load_dataset(
        foldername,
        'full_size',
        split='train',
        verification_mode='no_checks'
    )

    # initialize modelid
    model_id = "openai/clip-vit-base-patch32"

    # assigns cpu and processor, pulls model from CLIP
    device = "cpu"
    with torch.inference_mode():
        model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # embed several images
    images = [imagenette[i]['image'] for i in range(len(os.listdir(foldername)))]
    # print(len(images))

    batch_size = 16
    image_arr = None

    # embeds models batch_size at a time
    for i in tqdm(range(0, len(images), batch_size), desc=f'Training model on batch size of {batch_size}'):
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
    # saves embeddings to .npy file
    np.save(filename, image_arr)

if __name__ == '__main__':
    # runs program for both good and bad form
    main('Good Form', 'good')
    main('Bad Form', 'bad')
