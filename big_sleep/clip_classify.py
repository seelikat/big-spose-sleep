import os
import glob
import clip
import torch
from torchvision.datasets import CIFAR100

if __name__=="__main__":
    
    onlocal = True
    vocab = "data/imagenet21k_wordnet_lemmas.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if onlocal:
        imgdir = '/Users/katja/ownCloud/Share/roi_profiles'
    else:
        imgdir = None
    
    imgfns = glob.glob( os.path.join(imgdir, '*.png') )

    classes = []
    with open(vocab, 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            classes.append( line.strip() )
    
    model, preprocess = clip.load('ViT-B/32', device)   # TODO: replace with ViT-L/14

    text_inputs = torch.cat( [clip.tokenize(f"a photo of a {c}") for classtoken in classes] ).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for imgfn in imgfns:
        image_input = preprocess( Image.open( os.path.join(imgdir, imgfn) ).convert("RGB") ).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        # Print the result
        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
    
        break
