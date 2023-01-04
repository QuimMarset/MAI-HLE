import numpy as np
from utils.file_io_utils import read_data_file


class WordEmbeddingFeatureExtractor:


    def __init__(self, pre_trained_path, embed_dim, max_length):
        self.pre_trained_path = pre_trained_path
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.embedding_matrix = []
        self.word_to_index = {}
        self.__load_pre_trained_embeddings()


    def __add_special_tokens(self):
        self.word_to_index['PAD'] = 0
        self.word_to_index['UNK'] = 1
        self.word_to_index['e11'] = 2
        self.word_to_index['e12'] = 3
        self.word_to_index['e21'] = 4
        self.word_to_index['e22'] = 5


    def __fill_with_pre_trained_embeddings(self):
        lines = read_data_file(self.pre_trained_path)
        for line in lines:
            word, vector = line.strip().split(maxsplit=1)
            vector = np.fromstring(vector, dtype=float, sep=' ')
            self.word_to_index[word] = len(self.word_to_index)
            self.embedding_matrix.append(vector)


    def __add_embeddings_for_special_tokens(self):
        mean = self.embedding_matrix.mean()
        std = self.embedding_matrix.std()
        special_embeddings = np.random.normal(mean, std, (6, self.embed_dim))
        special_embeddings[0] = 0  # PAD is a vector of 0s
        self.embedding_matrix = np.concatenate((special_embeddings, self.embedding_matrix), axis=0)


    def __load_pre_trained_embeddings(self):
        self.__add_special_tokens()
        self.__fill_with_pre_trained_embeddings()
        self.embedding_matrix = np.stack(self.embedding_matrix)
        self.__add_embeddings_for_special_tokens()
        self.embedding_matrix = self.embedding_matrix.astype(np.float32)


    def __compute_sentence_features(self, sentence):
        length = min(self.max_length, len(sentence))
        mask = [1] * length
        word_indices = []

        for i in range(length):
            word = sentence[i].lower()
            index = self.word_to_index.get(word, self.word_to_index['UNK'])
            word_indices.append(index)

        for i in range(length, self.max_length):
            mask.append(0)
            word_indices.append(self.word_to_index['PAD'])

        return word_indices, mask


    def compute_features(self, data, labels):
        features = []
        for sentence_index in data:
            sentence_data = data[sentence_index]
            sentence = sentence_data['sentence'].split()

            word_indices, mask = self.__compute_sentence_features(sentence)
            sentence_features = np.array([word_indices, mask], dtype=np.int32)
            features.append(sentence_features)

        return np.array(features), np.array(labels)