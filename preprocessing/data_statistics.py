import numpy as np
from utils.constants_paths import *
from utils.file_io_utils import load_json_to_dict, load_npy_file_to_np_array
from utils.plot_utils import plot_classes_histogram



def compute_numeric_statistics(train_data):
    max_distance = 0
    max_length = 0
    unique_tokens = 0
    train_vocabulary = set()
    index_max_dist = 0

    for index in train_data:
        e1_start, e1_end = train_data[index]['e1_span']
        e2_start, e2_end = train_data[index]['e2_span']
        words_data = train_data[index]['words']

        sen_length = len(words_data)
        if sen_length > max_length:
            max_length = sen_length

        for i, word_data in enumerate(words_data):
            lc_word = word_data['lc_form']
            if lc_word not in train_vocabulary:
                train_vocabulary.add(lc_word)
                unique_tokens += 1

        d1 = max(e1_start, sen_length - e1_end - 1)
        d2 = max(e2_start, sen_length - e2_end - 1)
        d = max(d1, d2)
        if d > max_distance:
            index_max_dist = index
            max_distance = d

    print(f'Max sentence length: {max_length}')
    print(f'Number of unique tokens: {unique_tokens}')
    print(f'Max distance to entity: {max_distance}')
    print(f'Index of max distance {index_max_dist}')


def compute_class_distributions(train_labels, test_labels, class_to_index, save_path):
    class_names = list(class_to_index.keys())
    plot_classes_histogram(train_labels, class_names, class_to_index, save_path, 'train')
    plot_classes_histogram(test_labels, class_names, class_to_index, save_path  , 'test')


if __name__ == '__main__':

    train_data = load_json_to_dict(train_data_path)
    train_labels = load_npy_file_to_np_array(train_labels_path)
    test_labels = load_npy_file_to_np_array(test_labels_path)
    class_to_index = load_json_to_dict(class_to_index_path)

    compute_numeric_statistics(train_data)
    compute_class_distributions(train_labels, test_labels, class_to_index, processed_data_path)