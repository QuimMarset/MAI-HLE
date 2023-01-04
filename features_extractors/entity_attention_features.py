import numpy as np
from features_extractors.word_embedding_features import WordEmbeddingFeatureExtractor
from utils.file_io_utils import load_npy_file_to_np_array
from utils.constants_paths import relative_positions_vocabulary_path



class EntityAttentionFeatureExtractor(WordEmbeddingFeatureExtractor):

    def __init__(self, pre_trained_path, embed_dim, max_length):
        super().__init__(pre_trained_path, embed_dim, max_length)
        self.__compute_rel_pos_to_index()


    def __compute_rel_pos_to_index(self):
        self.rel_pos_to_index = {}
        self.rel_pos_to_index['PAD'] = 0
        self.rel_pos_to_index['UNK'] = 1
        rel_positions = load_npy_file_to_np_array(relative_positions_vocabulary_path)
        for rel_pos in rel_positions:
            self.rel_pos_to_index[rel_pos] = len(self.rel_pos_to_index)


    def __compute_sentence_features(self, sentence, pos_e1_sequence, pos_e2_sequence):
        word_indices = []
        rel_pos_e1 = []
        rel_pos_e2 = []
        mask = []
        length = min(self.max_length, len(sentence))

        for i in range(length):
            mask.append(1)
            
            word = sentence[i].lower()
            word_index = self.word_to_index.get(word, self.word_to_index['UNK'])
            word_indices.append(word_index)

            pos_e1 = int(pos_e1_sequence[i])
            #pos_e1_index = self.rel_pos_to_index.get(pos_e1, self.rel_pos_to_index['UNK'])
            rel_pos_e1.append(pos_e1)

            pos_e2 = int(pos_e2_sequence[i])
            #pos_e2_index = self.rel_pos_to_index.get(pos_e2, self.rel_pos_to_index['UNK'])
            rel_pos_e2.append(pos_e2)

        for i in range(length, self.max_length):
            mask.append(0)
            word_indices.append(self.word_to_index['PAD'])
            rel_pos_e1.append(2 * self.max_length + 1)
            rel_pos_e2.append(2 * self.max_length + 1)

        return word_indices, rel_pos_e1, rel_pos_e2, mask


    def compute_features(self, data, labels):
        features = []
        for sentence_index in data:
            sentence_data = data[sentence_index]

            sentence = sentence_data['sentence'].split()
            pos_e1_sequence = sentence_data['pos_e1'].split()
            pos_e2_sequence = sentence_data['pos_e2'].split()

            word_indices, pos_e1, pos_e2, mask = self.__compute_sentence_features(sentence, pos_e1_sequence, pos_e2_sequence)

            temp = np.zeros(self.max_length, dtype=np.int32)
            temp[0] = sentence_data['e1_end']
            temp[1] = sentence_data['e2_end']

            sentence_features = np.array([word_indices, pos_e1, pos_e2, temp, mask], dtype=np.int32)
            features.append(sentence_features)

        return np.array(features), np.array(labels)