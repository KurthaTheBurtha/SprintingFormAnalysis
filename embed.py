import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

imagenette = load_dataset(
    'Good Form',
    'full_size',
    split='train',
    ignore_verifications=True
)

model_id = "openai/clip-vit-base-patch32"

device = "cpu"
model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

prompt = "a dog in the snow"
inputs = tokenizer(prompt, return_tensors="pt")

text_emb = model.get_text_features(**inputs)

image = processor(
    text = None,
    images= imagenette[0]['image'],
    return_tensors="pt"
)['pixel_values'].to(device)
# print(image.shape)

plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
plt.show()

image_emb = model.get_image_features(image)
# print(image_emb.shape)

# embed several images
np.random.seed(0)
sample_idx = np.random.randint(0, len(imagenette)+1,25).tolist()
images = [imagenette[i]['image'] for i in sample_idx]
print(len(images))

from tqdm.auto import tqdm
batch_size = 16
image_arr = None

for i in tqdm(range(0,len(images),batch_size)):
    batch = images[i:i+batch_size]
    batch = processor(
        text=None,
        images = batch,
        return_tensors="pt",
        padding=True
    )['pixel_values'].to(device)
    batch_emb = model.get_image_features(pixel_values=batch)
    batch_emb = batch_emb.squeeze(0)
    batch_emb = batch_emb.cpu().detach().numpy()
    if image_arr is None:
        image_arr = batch_emb
    else:
        image_arr = np.concatenate((image_arr, batch_emb), axis=0)
print(image_arr.shape)
print("First Embedding", end='')
print(image_arr[0])
print()