import numpy as np
from big_sleep.clip import load, tokenize


if __name__=="__main__": 

    print("hello")
    
    perceptor, normalize_image = load('ViT-B/32', jit = False)





    # zero mean / unit var on SPoSE vectors