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



if __name__ == '__main__':
    

    train_data = load_json_to_dict(train_data_path)
    test_data = load_json_to_dict(test_data_path)

    create_folder(vocabularies_path)

    create_word_vocabulary(train_data, test_data)