import sys
sys.path.append('./')
from utils.file_io_utils import load_json_to_dict, save_array_to_npy_file
from utils.constants_paths import *
from utils.path_utils import create_folder



def create_word_vocabulary_set(data, vocabulary):
    for sentence_index in data:
        sentence = data[sentence_index]['sentence']
        for word in sentence.split():
            if word in ['e11', 'e12', 'e21', 'e22']:
                continue
            vocabulary.add(word)
    return vocabulary


def create_word_vocabulary(train_data, test_data):
    vocabulary = set()
    vocabulary = create_word_vocabulary_set(train_data, vocabulary)
    vocabulary = create_word_vocabulary_set(test_data, vocabulary)
    vocabulary = list(vocabulary)
    save_array_to_npy_file(vocabulary, words_vocabulary_path)


def create_relative_position_vocabulary_set(data, vocabulary):
    for sentence_index in data:
        relative_pos_e1 = data[sentence_index]['pos_e1'].split()
        relative_pos_e2 = data[sentence_index]['pos_e2'].split()
        for pos_e1, pos_e2 in zip(relative_pos_e1, relative_pos_e2):
            vocabulary.add(int(pos_e1))
            vocabulary.add(int(pos_e2))
    return vocabulary


def create_relative_position_vocabulary(train_data, test_data):
    vocabulary = set()
    vocabulary = create_relative_position_vocabulary_set(train_data, vocabulary)
    vocabulary = create_relative_position_vocabulary_set(test_data, vocabulary)
    vocabulary = list(vocabulary)
    save_array_to_npy_file(vocabulary, relative_positions_vocabulary_path)



if __name__ == '__main__':
    

    train_data = load_json_to_dict(train_data_path)
    test_data = load_json_to_dict(test_data_path)

    create_folder(vocabularies_path)

    create_word_vocabulary(train_data, test_data)
    create_relative_position_vocabulary(train_data, test_data)