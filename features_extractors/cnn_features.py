from utils.file_io_utils import save_array_to_npy_file, load_npy_file_to_np_array
from utils.path_utils import join_path, exists_path
from utils.embedding_utils import preprocess_sentence_for_embedding



class CNNFeatureExtractor(object):

    def __init__(self, split_name, data, words_to_index, max_length, features_path, max_distance):
        self.split_name = split_name
        self.max_distance = max_distance
        self.features = []
        self.file_path = join_path(features_path, f'{self.split_name}_cnn_features.npy')

        if exists_path(self.file_path):
            self.features = load_npy_file_to_np_array(self.file_path)
        else:
            self.__compute_samples_features(data, words_to_index, max_length)
            save_array_to_npy_file(self.features, self.file_path)
            

    def get_features(self):
        assert len(self.features) > 0
        return self.features


    def __compute_samples_features(self, data, words_to_index, max_length):
        for sentence_index in data:
            sentence_data = data[sentence_index]
            sample_featuress = self.__compute_sample_features(sentence_data, words_to_index, max_length)
            self.features.append(sample_featuress)


    def __compute_sample_features(self, sentence_data, words_to_index, max_length):
        words = sentence_data['words']
        e1_start = sentence_data['e1_start'] 
        e1_end = sentence_data['e1_end'] 
        e2_start = sentence_data['e2_start'] 
        e2_end = sentence_data['e2_end'] 

        words_idx = preprocess_sentence_for_embedding(words, words_to_index, max_length)

        lexical_features = self.__compute_lexical_features(words_idx, e1_start, e1_end, e2_start, e2_end)
        e1_positions, e2_positions = self.__compute_position_features(len(words_idx), e1_start, e1_end, e2_start, e2_end)

        sample_features = [*words_idx, *e1_positions, *e2_positions, *lexical_features]
        return sample_features

    
    def __compute_lexical_features(self, sentence_indices, e1_start, e1_end, e2_start, e2_end):
        left_e1 = self.__get_left_word(sentence_indices, e1_start)
        left_e2 = self.__get_left_word(sentence_indices, e2_start)
        right_e1 = self.__get_right_word(sentence_indices, e1_end)
        right_e2 = self.__get_right_word(sentence_indices, e2_end)
        return sentence_indices[e1_start], sentence_indices[e2_start], left_e1, left_e2, right_e1, right_e2


    def __compute_position_features(self, sentence_length, e1_start, e1_end, e2_start, e2_end):
        relative_e1_positions = []
        relative_e2_positions = []
        
        for index in range(sentence_length):
            relative_e1 = self.__compute_relative_pos(index, e1_start, e1_end)
            relative_e2 = self.__compute_relative_pos(index, e2_start, e2_end)
            relative_e1_positions.append(relative_e1)
            relative_e2_positions.append(relative_e2)

        return relative_e1_positions, relative_e2_positions


    def __get_left_word(self, sentence_indices, start_position):
        # 0 is the padding token index
        if start_position > 0:
            return sentence_indices[start_position - 1]
        return 0


    def __get_right_word(self, sentence_indices, end_position):
        # 0 is the padding token index
        if end_position < len(sentence_indices) - 1:
            return sentence_indices[end_position + 1]
        return 0

    
    def __compute_relative_pos(self, index, start, end):
        if index < start:
            return self.__get_pos_index(index - start)
        elif index > end:
            return self.__get_pos_index(index - end)
        else:
            return self.__get_pos_index(0)


    def __get_pos_index(self, distance):
        if distance < -self.max_distance:
            return 0
        
        if distance >= -self.max_distance and distance <= self.max_distance:
            return distance + self.max_distance + 1
        
        if distance > self.max_distance:
            return 2 * self.max_distance + 2