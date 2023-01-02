import pandas as pd
import numpy as np
from utils.embedding_utils import map_to_indices, create_vectorizer



class EntityAwareAttentionFeatureExtractor:

    def __init__(self, word_to_index, rel_pos_to_index, max_sentence_length):
        self.max_sentence_length = max_sentence_length
        self.word_vectorizer = create_vectorizer(word_to_index, max_sentence_length)
        self.rel_pos_vectorizer = create_vectorizer(rel_pos_to_index, max_sentence_length)


    def compute_features(self, data, labels):
        features = []
        for sentence_index in data:
            sentence_data = data[sentence_index]

            e1_end = sentence_data['e1_end']
            e2_end = sentence_data['e2_end']
            word_indices = map_to_indices(self.word_vectorizer, sentence_data['sentence'].split())
            rel_pos_1_indices = map_to_indices(self.rel_pos_vectorizer, sentence_data['pos_e1'].split())
            rel_pos_2_indices = map_to_indices(self.rel_pos_vectorizer, sentence_data['pos_e2'].split())

            features.append([e1_end, e2_end, *word_indices.tolist(), *rel_pos_1_indices.tolist(), *rel_pos_2_indices.tolist()])

        return np.array(features), np.array(labels)





def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
