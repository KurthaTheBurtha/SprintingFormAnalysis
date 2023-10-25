import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor

def main():
    # initialize modelid
    model_id = "openai/clip-vit-base-patch32"

    # assigns cpu and processor, pulls model from CLIP
    device = "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

    # read input from text file
    f = open("eng_desc.txt", "r")
    goodinput = f.readline()
    badinput = f.readline()

    # encode text to vectors
    with torch.inference_mode():
        goodtoken = tokenizer(goodinput, return_tensors="pt")
        badtoken = tokenizer(badinput, return_tensors="pt")
        good_emb = model.get_text_features(**goodtoken).numpy()
        bad_emb = model.get_text_features(**badtoken).numpy()

    # read from embeddings
    good = np.load('good.npy')
    bad = np.load('bad.npy')

    # for good
    length = min(len(good), len(bad))
    gooddis = 0
    baddis = 0
    for i in range(length):
        gooddis = gooddis + pow(np.linalg.norm(good[i] - good_emb),2)
        baddis = baddis + pow(np.linalg.norm(bad[i] - good_emb),2)
    print("Good Embeddings")
    print("Sum of Squared Distance from Good Vectors: " + str(gooddis))
    print("Sum of Squared Distance from Bad Vectors: " + str(baddis))
    if gooddis < baddis:
        ans = "Good"
    else:
        ans = "Bad"
    print("Closer to " + ans + " Embeddings")

    print()

    # for bad
    gooddis = 0
    baddis = 0
    for i in range(length):
        gooddis = gooddis + pow(np.linalg.norm(good[i] - bad_emb),2)
        baddis = baddis + pow(np.linalg.norm(bad[i] - bad_emb),2)
    print("Bad Embeddings")
    print("Sum of Squared Distance from Good Vectors: " + str(gooddis))
    print("Sum of Squared Distance from Bad Vectors: " + str(baddis))
    if gooddis < baddis:
        ans = "Good"
    else:
        ans = "Bad"
    print("Closer to " + ans + " Embeddings")

if __name__ == "__main__":
    main()
