import numpy as np
from utils.embedding_utils import create_vectorizer, map_to_indices



class AttentionBiLSTMFeatureExtractor:

    def __init__(self, word_to_index, max_length):
        self.word_vectorizer = create_vectorizer(word_to_index, max_length)
        self.max_length = max_length
            

    def compute_features(self, data, labels):
        features = []
        for sentence_index in data:
            sentence = data[sentence_index]['sentence_tags']
            word_indices = map_to_indices(self.word_vectorizer, sentence.split())
            features.append(word_indices)

        return np.array(features), np.array(labels)