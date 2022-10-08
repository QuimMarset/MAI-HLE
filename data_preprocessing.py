import numpy as np
import re
from utils.file_io_utils import save_array_to_npy_file, read_data_file
from utils.path_utils import join_path, create_folder



def is_line_useful(line):
    return line != '' and line != 'Comment'


def extract_entity(sentence, limiter_token):
    pattern = f'<{limiter_token}>.+<\/{limiter_token}>'
    match = re.search(pattern, sentence)
    entity = match.group(0)[4:-5]
    return entity


def process_sample(sentence, label):
    entity_1 = extract_entity(sentence, 'e1')
    entity_2 = extract_entity(sentence, 'e2')
    processed_label = label.strip('\n')
    processed_sentence = sentence.strip('\n').split('\t')[1][1:-1]

    return (processed_sentence, entity_1, entity_2, processed_label)


def preprocess_train_file_content(file_lines, save_path):
    processed_data = []
    num_lines = len(file_lines) - 1

    for i in range(0, num_lines, 4):
        sentence = file_lines[i]
        label = file_lines[i+1]
        proceseed_sample = process_sample(sentence, label)
        processed_data.append(proceseed_sample)

    save_array_to_npy_file(processed_data, save_path, 'processed_train_data')



if __name__ == '__main__':
    data_path = 'data'
    train_file_path = join_path(data_path, 'TRAIN_FILE.txt')
    processed_data_path = 'processed_data'

    create_folder(processed_data_path)

    train_content = read_data_file(train_file_path)
    preprocess_train_file_content(train_content, processed_data_path)
