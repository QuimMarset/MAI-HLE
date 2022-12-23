from tensorflow import keras
import numpy as np
from utils.path_utils import *
from utils.file_io_utils import *



class CNNRNNFeatureExtractor:


    def __init__(self, split_name, data, word_to_index, max_length, features_path):
        file_path = join_path(features_path, f'{split_name}_rnn_cnn_features.npy')

        if exists_path(file_path):
            self.features = load_npy_file_to_np_array(file_path)
        else:
            self.compute_vocabularies(data)
            word_to_index['<E1>'] = len(word_to_index)
            word_to_index['<E2>'] = len(word_to_index)
            self.compute_samples_features(data, word_to_index, max_length)
            save_array_to_npy_file(self.features, file_path)


    def get_features(self):
        assert len(self.features) > 0
        return self.features


    def compute_vocabularies(self, data):
        lemmas = set()
        pos = set()
        lemmas.add('<E1>')
        lemmas.add('<E2>')
        pos.add('<E1>')
        pos.add('<E2>')

        for index in data:
            sentence_data = data[index]
            for word_data in sentence_data['words']:
                lemmas.add(word_data['lemma'])
                pos.add(word_data['pos'])

        self.lemma_to_index = {lemma: i + 2 for (i, lemma) in enumerate(sorted(list(lemmas)))}
        self.lemma_to_index[''] = 0
        self.lemma_to_index['[UNK]'] = 1

        self.pos_to_index = {pos: i + 2 for (i, pos) in enumerate(sorted(list(pos)))}
        self.pos_to_index[''] = 0
        self.pos_to_index['[UNK]'] = 1


    def compute_samples_features(self, data, word_to_index, max_length):
        lc_words = []
        lemmas = []
        pos = []
        for sentence_index in data:
            sentence_data = data[sentence_index]
            lc_words_i, lemmas_i, pos_i = self.compute_sample_features(sentence_data, word_to_index, max_length)
            lc_words.append(lc_words_i)
            lemmas.append(lemmas_i)
            pos.append(pos_i)

        lc_words = np.expand_dims(lc_words, axis=-1)
        lemmas = np.expand_dims(lemmas, axis=-1)
        pos = np.expand_dims(pos, axis=-1)
        self.features = np.concatenate([lc_words, lemmas, pos], axis=-1)


    def get_index(self, element_to_index, element):
        return element_to_index.get(element, element_to_index['[UNK]'])


    def compute_sample_features(self, sample_data, word_to_index, max_length):
        lc_words = []
        lemmas = []
        pos = []
        e1_start, e1_end = sample_data['e1_span']
        e2_start, e2_end = sample_data['e2_span']

        for i, token in enumerate(sample_data['words']):
            if i == e1_start:
                lc_word = lemma = pos_ = '<E1>'
            elif i == e2_start:
                lc_word = lemma = pos_ = '<E2>'
            elif i > e1_start and i <= e1_end or i > e2_start and i <= e2_end:
                continue
            else:
                lc_word = token['lc_form']
                lemma = token['lemma']
                pos_ = token['pos']

            lc_words.append(self.get_index(word_to_index, lc_word))
            lemmas.append(self.get_index(self.lemma_to_index, lemma))
            pos.append(self.get_index(self.pos_to_index, pos_))

        pad_function = keras.preprocessing.sequence.pad_sequences
        lc_words = pad_function([lc_words], max_length, padding="post", value=word_to_index[''])[0]
        lemmas = pad_function([lemmas], max_length, padding="post", value=self.lemma_to_index[''])[0]
        pos = pad_function([pos], max_length, padding="post", value=self.pos_to_index[''])[0]
        return lc_words, lemmas, pos