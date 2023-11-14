import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor

def main():
    # initialize modelid
    model_id = "openai/clip-vit-base-patch32"

    # assigns cpu and processor, pulls model from CLIP
    device = "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

    # read input from text file
    f = open("eng_desc.txt", "r")
    goodinput = f.readline()
    badinput = f.readline()
    # encode text to vectors
    with torch.inference_mode():
        good_form_desc_emb = model.get_text_features(
            **tokenizer(goodinput.strip().split(', '), padding=True, return_tensors='pt')
        ).detach().numpy()
        bad_form_desc_emb = model.get_text_features(
            **tokenizer(badinput.strip().split(', '), padding=True, return_tensors='pt')
        ).detach().numpy()

    # read from embeddings
    good = np.load('good.npy')
    bad = np.load('bad.npy')

    # good
    diff = 0
    totalgood = 0
    totalbad = 0
    for i in good_form_desc_emb:
        goodmean = np.mean([np.linalg.norm(img - i) for img in good])
        # print("Good Mean: "+str(goodmean))
        badmean = np.mean([np.linalg.norm(img - i) for img in bad])
        # print("Bad Mean: "+str(badmean))
        diff = diff+ (goodmean-badmean)
        if goodmean < badmean:
            totalgood +=1
        else:
            totalbad += 1
    print("Total difference between Images and Good Text Embeddings: " + str(diff))
    print()
    # print("Good") if diff < 0 else print("Bad")
    # print("Total Good: " + str(totalgood))
    # print("Total Bad: " + str(totalbad))

    # bad
    diff = 0
    totalgood = 0
    totalbad = 0
    for i in bad_form_desc_emb:
        goodmean = np.mean([np.linalg.norm(img - i) for img in good])
        # print("Good Mean: "+str(goodmean))
        badmean = np.mean([np.linalg.norm(img - i) for img in bad])
        # print("Bad Mean: "+str(badmean))
        diff = diff+ (goodmean-badmean)
        if goodmean < badmean:
            totalgood +=1
        else:
            totalbad += 1
    print("Total difference between Images and Bad Text Embeddings: " + str(diff))
    # print("Good") if diff < 0 else print("Bad")
    # print("Total Good: " + str(totalgood))
    # print("Total Bad: " + str(totalbad))

if __name__ == "__main__":
    main()
