import os
import gc
import glob
import clip
import torch
import pdb
from PIL import Image
from torchvision.datasets import CIFAR100

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

if __name__=="__main__":
    
    onlocal = False
    vocab = "data/gpt3semantics.txt"
    outfile = 'classpredictions.txt'
    n_batch = 1250
    
    # imagenet21k_wordnet_lemmas.txt things_classes.txt gpt3semantics.txt  TODO clip vocab?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if onlocal:
        imgdir = '/Users/katja/ownCloud/Share/roi_profiles'
    else:
        imgdir = '/LOCAL/kamue/big-spose-sleep/big_sleep'

    imgfns = glob.glob( os.path.join(imgdir, '*.png') )

    classes = []
    with open(vocab, 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            classes.append( line.strip() )

    model, preprocess = clip.load('ViT-B/32', device)   # TODO: replace with ViT-L/14

    class_simils = { imgfn : torch.empty([1,len(classes)]).to(device) for imgfn in imgfns }

    # collect class similarities
    cur_i = 0
    break_i = 0
    for classesbatch in chunker(classes, n_batch):
        print(break_i)

        text_inputs = torch.cat( [clip.tokenize(c) for c in classesbatch] ).to(device)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # TODO: norm across all?

        for imgfn in imgfns:
            image_input = preprocess( Image.open( os.path.join(imgdir, imgfn) ).convert("RGB") ).unsqueeze(0).to(device)

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)

            # Pick the top 10 most similar labels for the imepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            class_simils[imgfn][ 0, cur_i:cur_i+len(classesbatch) ] = similarity.data

        cur_i += len(classesbatch)

        del text_features
        torch.cuda.empty_cache()

        break_i += 1
        if break_i > 5:
            break

    # evaluate class similarities
    classpredfile = open(outfile, 'w')
    for imgfn in imgfns:
        
        # take top 10 classes
        vals, idxs = class_simils[imgfn][0].topk(10)   # auto-chooses last dimension

        # Print the result
        #print("\nTop predictions:\n")
        #for value, index in zip(values, indices):
        #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")

        classpredline = imgfn
        for val, idx in zip(vals, idxs):
            classpredline += f',{100 * val.item():.2f}%|classes[idx]'

        classpredfile.write('%s\n' % classpredline)
    
    classpredfile.close()
    
