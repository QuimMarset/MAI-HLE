import numpy as np
from utils.path_utils import *
import numpy as np


if __name__ == '__main__':

    senna = join_path('embeddings', 'embed50.senna.npy')
    with open(senna, 'rb') as file:
        embed = np.load(file)

    words = join_path('embeddings', 'senna_words.lst')
    with open(words, 'r') as file:
        x = file.readlines()
    
    path_2 = join_path('embeddings', 'w2v_50.npy')
    with open(senna, 'rb') as file:
        embed_2 = np.load(file)

    print()