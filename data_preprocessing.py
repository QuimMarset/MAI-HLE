import re
import numpy as np
from nltk.tokenize import word_tokenize
from utils.file_io_utils import save_array_to_npy_file, write_dict_to_json, read_data_file
from utils.path_utils import create_folder
from constants_paths import *



def extract_entities(sentence):
    entity_1 = re.search(r'<e1>(.*)</e1>', sentence).group(1)
    entity_2 = re.search(r'<e2>(.*)</e2>', sentence).group(1)
    return entity_1, entity_2


def separate_entities_from_tag(sentence, entity_1, entity_2):
    sentence = sentence.replace('<e1>' + entity_1 + '</e1>', ' <e1> ' + entity_1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + entity_2 + '</e2>', ' <e2> ' + entity_2 + ' </e2> ', 1)
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< e1 >', '<e1>')
    sentence = sentence.replace('< e2 >', '<e2>')
    sentence = sentence.replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< /e2 >', '</e2>')
    return sentence


def remove_tags(processed_sentence):
    splitted_sentence = processed_sentence.split()
    e1_start = e1_end = e2_start = e2_end = 0
    words = []

    for word in splitted_sentence:
        if word == '<e1>':
            e1_start = len(words)
        elif word == '</e1>':
            e1_end = len(words) -1
        elif word == '<e2>':
            e2_start = len(words)
        elif word == '</e2>':
            e2_end = len(words) - 1
        else:
            words.append(word)

    return e1_start, e1_end, e2_start, e2_end, words


def process_sentence(sentence):
    entity_1, entity_2 = extract_entities(sentence)
    processed_sentence = separate_entities_from_tag(sentence, entity_1, entity_2)
    e1_start, e1_end, e2_start, e2_end, words = remove_tags(processed_sentence)
    return words, entity_1, entity_2, e1_start, e1_end, e2_start, e2_end


def create_class_to_index_dict(labels):
    classes = np.unique(labels)
    sorted_classes = np.sort(classes)
    class_to_index = dict(zip(sorted_classes, range(classes.shape[0])))
    return class_to_index


def preprocess_file(file_path):
    file_lines = read_data_file(file_path)
    processed_data = {}
    labels = []
    num_lines = len(file_lines) - 1

    for i in range(0, num_lines, 4):
        id, sentence = file_lines[i].strip().split('\t')
        sentence = sentence[1:-1]
        label = file_lines[i+1].strip()
        words, entity_1, entity_2, e1_start, e1_end, e2_start, e2_end = process_sentence(sentence)
        
        processed_data[id] = {
            'words' : words,
            'relation' : label,
            'entity_1' : entity_1,
            'entity_2' : entity_2,
            'e1_start' : e1_start,
            'e1_end' : e1_end,
            'e2_start' : e2_start,
            'e2_end' : e2_end,
        }

        labels.append(label)

    return processed_data, labels


def preprocess_train_data(train_file_path, save_path):
    train_data, train_labels = preprocess_file(train_file_path)
    class_to_index = create_class_to_index_dict(train_labels)
    write_dict_to_json(train_data, save_path, 'processed_train_data')
    save_array_to_npy_file(train_labels, save_path, 'train_labels')
    write_dict_to_json(class_to_index, save_path, 'class_to_index')


def preprocess_test_data(test_file_path, save_path):
    test_data, test_labels = preprocess_file(test_file_path)
    write_dict_to_json(test_data, save_path, 'processed_test_data')
    save_array_to_npy_file(test_labels, save_path, 'test_labels')



if __name__ == '__main__':

    create_folder(processed_data_path)
    preprocess_train_data(raw_train_data_path, processed_data_path)
    preprocess_test_data(raw_test_data_path, processed_data_path)
