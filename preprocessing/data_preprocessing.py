import nltk
import re
import sys
sys.path.append('./')
from utils.file_io_utils import read_data_file
from utils.label_dicts import class2label
from utils.file_io_utils import write_dict_to_json, save_array_to_npy_file
from utils.constants_paths import *



def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", sentence)
    sentence = re.sub(r"what's", "what is ", sentence)
    sentence = re.sub(r"that's", "that is ", sentence)
    sentence = re.sub(r"there's", "there is ", sentence)
    sentence = re.sub(r"it's", "it is ", sentence)
    sentence = re.sub(r"\'s", " ", sentence)
    sentence = re.sub(r"\'ve", " have ", sentence)
    sentence = re.sub(r"can't", "can not ", sentence)
    sentence = re.sub(r"n't", " not ", sentence)
    sentence = re.sub(r"i'm", "i am ", sentence)
    sentence = re.sub(r"\'re", " are ", sentence)
    sentence = re.sub(r"\'d", " would ", sentence)
    sentence = re.sub(r"\'ll", " will ", sentence)
    sentence = re.sub(r",", " ", sentence)
    sentence = re.sub(r"\.", " ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\/", " ", sentence)
    sentence = re.sub(r"\^", " ^ ", sentence)
    sentence = re.sub(r"\+", " + ", sentence)
    sentence = re.sub(r"\-", " - ", sentence)
    sentence = re.sub(r"\=", " = ", sentence)
    sentence = re.sub(r"'", " ", sentence)
    sentence = re.sub(r"(\d+)(k)", r"\g<1>000", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r" e g ", " eg ", sentence)
    sentence = re.sub(r" b g ", " bg ", sentence)
    sentence = re.sub(r" u s ", " american ", sentence)
    sentence = re.sub(r"\0s", "0", sentence)
    sentence = re.sub(r" 9 11 ", "911", sentence)
    sentence = re.sub(r"e - mail", "email", sentence)
    sentence = re.sub(r"j k", "jk", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip()


def change_entity_markers(sentence):
    # Change them to later facilitate the entities extraction
    sentence = sentence.replace('<e1>', ' _e11_ ')
    sentence = sentence.replace('</e1>', ' _e12_ ')
    sentence = sentence.replace('<e2>', ' _e21_ ')
    sentence = sentence.replace('</e2>', ' _e22_ ')
    return sentence


def extract_entities(tokens):
    e1_start = tokens.index("e11") + 1
    e1_end = tokens.index("e12") - 1
    e2_start = tokens.index("e21") + 1
    e2_end = tokens.index("e22") - 1
    return e1_start, e1_end, e2_start, e2_end


def compute_relative_position(sentence, e1_end, e2_end, max_sentence_length):
    tokens = nltk.word_tokenize(sentence)
    positions_1 = ''
    positions_2 = ''

    for word_index in range(len(tokens)):
        # Add max sentence length to rescale to positive numbers (-max, max) -> (0, 2*max)
        positions_1 += str((max_sentence_length - 1) + word_index - e1_end) + ' '
        positions_2 += str((max_sentence_length - 1) + word_index - e2_end) + ' '

    return positions_1, positions_2


def compute_sentence_without_tags(tokens):
    cleaned_tokens = []
    for token in tokens:
        if token in ['e11', 'e12', 'e21', 'e22']:
            continue
        cleaned_tokens.append(token)
    return ' '.join(cleaned_tokens)


def preprocess_data(file_path, max_sentence_length):
    data = {}
    labels = []
    lines = read_data_file(file_path)

    for idx in range(0, len(lines), 4):
        id = lines[idx].split('\t')[0]
        relation = lines[idx + 1]
        # Remove quotations
        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = change_entity_markers(sentence)
        sentence = clean_sentence(sentence)
        tokens = nltk.word_tokenize(sentence)

        e1_start, e1_end, e2_start, e2_end = extract_entities(tokens)
        sentence = ' '.join(tokens)
        label = class2label[relation]
        pos_1, pos_2 = compute_relative_position(sentence, e1_start, e1_end, max_sentence_length)

        sentence_without_tags = compute_sentence_without_tags(tokens)

        data[id] = {
            'sentence' : sentence,
            'sentence_no_tags' : sentence_without_tags,
            'e1_start' : e1_start,
            'e1_end' : e1_end,
            'e2_start' : e2_start,
            'e2_end' : e2_end,
            'pos_e1' : pos_1,
            'pos_e2' : pos_2,
            'relation' : relation,
            'label' : class2label[relation]
        }

        labels.append(label)

    return data, labels



if __name__ == "__main__":
    max_sentence_length = 100

    train_data, train_labels = preprocess_data(raw_train_data_path, max_sentence_length)
    test_data, test_labels = preprocess_data(raw_test_data_path, max_sentence_length)

    write_dict_to_json(train_data, train_data_path)
    save_array_to_npy_file(train_labels, train_labels_path)

    write_dict_to_json(test_data, test_data_path)
    save_array_to_npy_file(test_labels, test_labels_path)