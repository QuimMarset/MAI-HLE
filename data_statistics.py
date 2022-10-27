import numpy as np
from utils.path_utils import join_path
from utils.file_io_utils import load_json_to_dict



if __name__ == '__main__':

    processed_train_data_path = join_path('processed_data', 'processed_train_data.json')
    processed_train_data = load_json_to_dict(processed_train_data_path)

    max_length = 0
    unique_tokens = 0
    train_vocabulary = set()

    for sentence_index in processed_train_data:
        words = processed_train_data[sentence_index]['words']

        if len(words) > max_length:
            max_length = len(words)

        for word in words:
            if word not in train_vocabulary:
                train_vocabulary.add(word)
                unique_tokens += 1

    print(f'Max sentence length: {max_length}')
    print(f'Number of unique tokens: {unique_tokens}')