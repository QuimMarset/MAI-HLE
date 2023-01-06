import re
import sys
sys.path.append('./')
from nltk.tokenize import word_tokenize
from utils.constants_paths import *
from utils.file_io_utils import *
from utils.path_utils import create_folder


def get_entities(sentence):
    e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
    e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
    return e1, e2


def tokenize_sentence(sentence, e1, e2):
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< e1 >', '<e1>')
    sentence = sentence.replace('< e2 >', '<e2>')
    sentence = sentence.replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< /e2 >', '</e2>')
    return sentence.split()


def remove_tags(sentence_words):
    sentence_without_tags = []
    
    for word in sentence_words:
        if '<e1>' == word:
            e1_start = len(sentence_without_tags)
            continue

        if '</e1>' == word:
            e1_end = len(sentence_without_tags) - 1
            continue

        if '<e2>' == word:
            e2_start = len(sentence_without_tags)
            continue

        if '</e2>' == word:
            e2_end = len(sentence_without_tags) - 1
            continue

        sentence_without_tags.append(word)

    return e1_start, e1_end, e2_start, e2_end, sentence_without_tags


def extract_sentence_features(sentence):
    e1, e2 = get_entities(sentence)
    words = tokenize_sentence(sentence, e1, e2)
    e1_start, e1_end, e2_start, e2_end, sentence_without_tags = remove_tags(words)
    sentence_without_tags = ' '.join(sentence_without_tags)

    return {
        'sentence' : sentence_without_tags,
        'entity_1' : e1,
        'entity_2' : e2,
        'e1_start' : e1_start,
        'e1_end' : e1_end,
        'e2_start' : e2_start,
        'e2_end' : e2_end
    }


def preprocess_data(raw_data_path):
    file_lines = read_data_file(raw_data_path)
    processed_data = {}
    num_lines = len(file_lines) - 1

    for i in range(0, num_lines, 4):
        id, sentence = file_lines[i].strip().split('\t')
        # Remove quotations
        sentence = sentence[1:-1]
        processed_data[id] = extract_sentence_features(sentence)

    return processed_data



if __name__ == '__main__':
    create_folder(processed_data_path)

    train_data = preprocess_data(raw_train_data_path)
    test_data = preprocess_data(raw_test_data_path)

    write_dict_to_json(train_data, train_data_cnn_path)
    write_dict_to_json(test_data, test_data_cnn_path)