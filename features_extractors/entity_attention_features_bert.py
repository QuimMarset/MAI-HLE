import numpy as np
from features_extractors.bert_embedding_features import BERTFeatureExtractor
from utils.file_io_utils import load_npy_file_to_np_array
from utils.constants_paths import relative_positions_vocabulary_path



class EntityAttentionBERTFeatureExtractor(BERTFeatureExtractor):

    def __init__(self, max_length, words_vocabulary):
        super().__init__(max_length, words_vocabulary)
        self.__compute_rel_pos_to_index()


    def __compute_rel_pos_to_index(self):
        self.rel_pos_to_index = {}
        self.rel_pos_to_index['PAD'] = 0
        self.rel_pos_to_index['UNK'] = 1
        rel_positions = load_npy_file_to_np_array(relative_positions_vocabulary_path)
        for rel_pos in rel_positions:
            self.rel_pos_to_index[rel_pos] = len(self.rel_pos_to_index)


    def __compute_rel_position_features(self, pos_e1_sequence, pos_e2_sequence):
        rel_pos_e1 = []
        rel_pos_e2 = []
        length = min(self.max_length, len(pos_e1_sequence))

        for i in range(length):
            pos_e1 = int(pos_e1_sequence[i])
            rel_pos_e1.append(pos_e1)

            pos_e2 = int(pos_e2_sequence[i])
            rel_pos_e2.append(pos_e2)

        for i in range(length, self.max_length):
            rel_pos_e1.append(2 * self.max_length + 1)
            rel_pos_e2.append(2 * self.max_length + 1)

        return rel_pos_e1, rel_pos_e2


    def compute_features(self, data, labels):
        features = []
        for sentence_index in data:

            sentence_data = data[sentence_index]
            sentence = sentence_data['sentence']
            e1_start = sentence_data['e1_start'] - 1
            e1_end = sentence_data['e1_end'] + 1
            e2_start = sentence_data['e2_start'] - 1
            e2_end = sentence_data['e2_end'] + 1

            pos_e1_sequence = sentence_data['pos_e1'].split()
            pos_e2_sequence = sentence_data['pos_e2'].split()

            pos_e1, pos_e2 = self.__compute_rel_position_features(pos_e1_sequence, pos_e2_sequence)

            temp = np.zeros(self.max_length, dtype=np.int32)
            temp[0] = sentence_data['e1_end']
            temp[1] = sentence_data['e2_end']

            tokenized_sentence = self.tokenize_sentence(sentence)
            mask = self.create_mask(len(tokenized_sentence), e1_start, e1_end,
                e2_start, e2_end)
            
            tokenized_sentence, mask = self.truncate_if_needed(tokenized_sentence, mask)
            tokenized_sentence, mask = self.add_bert_tokens(tokenized_sentence, mask)

            sentence_features = np.array([tokenized_sentence, pos_e1, pos_e2, temp, mask], dtype=np.int32)
            features.append(sentence_features)

        return np.array(features), np.array(labels)