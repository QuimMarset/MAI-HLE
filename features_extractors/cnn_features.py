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


    def __compute_sentence_features(self, sentence, e1_pos, e2_pos):
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

    
    def __get_left_word(self, sentence_indices, start_position):
        if start_position > 0:
            return sentence_indices[start_position - 1]
        return sentence_indices[start_position]


    def __get_right_word(self, sentence_indices, end_position):
        if end_position < len(sentence_indices) - 1:
            return sentence_indices[end_position + 1]
        return sentence_indices[end_position]


    def __compute_lexical_features(self, indices, e1_pos, e2_pos):
        # Consider tags that are not removed
        left_e1 = self.__get_left_word(indices, e1_pos[0])
        left_e2 = self.__get_left_word(indices, e2_pos[0])
        right_e1 = self.__get_right_word(indices, e1_pos[1])
        right_e2 = self.__get_right_word(indices, e2_pos[1])
        e1 = indices[e1_pos[0]]
        e2 = indices[e2_pos[0]]
        return  [left_e1, e1, right_e1, left_e2, e2, right_e2]    



    def compute_features(self, data, labels):
        features = []
        for sentence_data in data:
            sentence = sentence_data['sentence']
            e1_pos = (sentence_data['subj_start'], sentence_data['subj_end'])
            e2_pos = (sentence_data['obj_start'], sentence_data['obj_end'])

            word_indices, mask, pos_e1, pos_e2 = self.__compute_sentence_features(sentence, e1_pos, e2_pos)
            
            lexical_features = self.__compute_lexical_features(word_indices, e1_pos, e2_pos)
            temp = np.zeros(len(word_indices), dtype=int)
            temp[:6] = lexical_features
            
            sentence_features = np.array([word_indices, pos_e1, pos_e2, mask], dtype=np.int64)
            features.append(sentence_features)

        return np.array(features), np.array(labels)