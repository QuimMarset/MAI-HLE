import numpy as np
from features_extractors.word_embedding_features import WordEmbeddingFeatureExtractor



class CNNFeatureExtractor(WordEmbeddingFeatureExtractor):


    def __init__(self, pre_trained_path, embed_dim, max_length, max_distance):
        super().__init__(pre_trained_path, embed_dim, max_length)
        self.max_distance = max_distance


    def __get_position_index(self, distance):
        # Scale from [-max_distance, max_distance] to [0, 2*max_distance]
        # 2 extra tokens to represent words out of the initial range
        if distance < -self.max_distance:
            return 0
        elif distance >= -self.max_distance and distance <= self.max_distance:
            return distance + self.max_distance + 1
        else:
            return 2 * self.max_distance + 2


    def __get_relative_position(self, index, entity_start, entity_end):
        if index < entity_start:
            return self.__get_position_index(index - entity_start)
        elif index > entity_end:
            return self.__get_position_index(index - entity_end)
        else:
            return self.__get_position_index(0)


    def compute_sentence_features(self, sentence, e1_pos, e2_pos):
        length = min(self.max_length, len(sentence))
        mask = []
        word_indices = []
        pos_e1 = []
        pos_e2 = []

        for i in range(length):
            word = sentence[i].lower()
            mask.append(1)
            index = self.word_to_index.get(word, self.word_to_index['*UNKNOWN*'])
            word_indices.append(index)
            pos_e1.append(self.__get_relative_position(i, *e1_pos))
            pos_e2.append(self.__get_relative_position(i, *e2_pos))

        for i in range(length, self.max_length):
            mask.append(0)
            word_indices.append(self.word_to_index['PAD'])
            pos_e1.append(self.__get_relative_position(i, *e1_pos))
            pos_e2.append(self.__get_relative_position(i, *e2_pos))

        return word_indices, mask, pos_e1, pos_e2   


    def compute_features(self, data, labels):
        features = []
        for index in data:
            sentence_data = data[index]
            sentence = sentence_data['sentence'].split()
            e1_pos = (sentence_data['e1_start'], sentence_data['e1_end'])
            e2_pos = (sentence_data['e2_start'], sentence_data['e2_end'])

            word_indices, mask, pos_e1, pos_e2 = self.compute_sentence_features(sentence, e1_pos, e2_pos)
            
            sentence_features = np.array([word_indices, pos_e1, pos_e2, mask], dtype=np.int64)
            features.append(sentence_features)

        return np.array(features), np.array(labels)