import re
import sys
sys.path.append('./')
from nltk.tokenize import word_tokenize
from corenlp.deptree import DependencyTree
from utils.constants_paths import *
from utils.file_io_utils import *
from utils.path_utils import create_folder



def extract_entities(sentence):
    entity_1 = re.search(r'<e1>(.*)</e1>', sentence).group(1)
    entity_2 = re.search(r'<e2>(.*)</e2>', sentence).group(1)
    return entity_1, entity_2


def get_entity_positions(sentence):
    words = sentence.split()
    for i, word in enumerate(words):
        if '<e1>' in word:
            e1_start = i
        if '</e1>' in word:
            e1_end = i
        if '<e2>' in word:
            e2_start = i
        if '</e2>' in word:
            e2_end = i

    return e1_start, e1_end, e2_start, e2_end


def separate_entity_tags(sentence, e1, e2):
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< e1 >', '<e1>')
    sentence = sentence.replace('< e2 >', '<e2>')
    sentence = sentence.replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< /e2 >', '</e2>')
    return sentence


def remove_entity_tags(sentence, entity_1, entity_2):
    sentence = re.sub(r'<e1>' + entity_1 + r'</e1>', entity_1, sentence)
    sentence = re.sub(r'<e2>' + entity_2 + r'</e2>', entity_2, sentence)
    return sentence


def extract_dep_tree_features_tokens(dep_tree):
    sentence_data = []
    
    for node_index in range(1, dep_tree.get_num_nodes()):
        if dep_tree.is_empty_node(node_index):
            continue

        token_data = {
            'form': dep_tree.get_word(node_index), 
            'lc_form': dep_tree.get_word(node_index).lower(), 
            'lemma': dep_tree.get_lemma(node_index), 
            'pos': dep_tree.get_tag(node_index),
            'rel': dep_tree.get_rel(node_index)
        }
        sentence_data.append(token_data)
        
    return sentence_data


def extract_token_features(sentence, entity_1, entity_2):
    dep_tree = DependencyTree(sentence, entity_1, entity_2)
    token_features = extract_dep_tree_features_tokens(dep_tree)
    return token_features


def extract_sentence_features(sentence):
    entity_1, entity_2 = extract_entities(sentence)
    e1_start, e1_end, e2_start, e2_end = get_entity_positions(sentence)
    sentence_separated_tags = separate_entity_tags(sentence, entity_1, entity_2)
    sentence = remove_entity_tags(sentence, entity_1, entity_2)
    token_features = extract_token_features(sentence, entity_1, entity_2)

    return {
        'sentence' : sentence,
        'sentence_tags': sentence_separated_tags,
        'entity_1' : entity_1,
        'entity_2' : entity_2,
        'words' : token_features,
        'e1_span' : [e1_start, e1_end],
        'e2_span' : [e2_start, e2_end],
    }


def preprocess_data(raw_data_path):
    file_lines = read_data_file(raw_data_path)
    processed_data = {}
    labels = []
    num_lines = len(file_lines) - 1

    for i in range(0, num_lines, 4):
        id, sentence = file_lines[i].strip().split('\t')
        # Remove quotations
        sentence = sentence[1:-1]
        label = file_lines[i+1].strip()
        labels.append(label)
        processed_data[id] = extract_sentence_features(sentence)

    return processed_data, labels


def create_class_to_index_dict(labels):
    classes = np.unique(labels)
    sorted_classes = np.sort(classes)
    class_to_index = dict(zip(sorted_classes, range(classes.shape[0])))
    return class_to_index



if __name__ == '__main__':
    create_folder(processed_data_path)

    train_data, train_labels = preprocess_data(raw_train_data_path)
    test_data, test_labels = preprocess_data(raw_test_data_path)
    
    class_to_index = create_class_to_index_dict(train_labels)

    write_dict_to_json(train_data, train_data_path)
    save_array_to_npy_file(train_labels, train_labels_path)

    write_dict_to_json(test_data, test_data_path)
    save_array_to_npy_file(test_labels, test_labels_path)

    write_dict_to_json(class_to_index, class_to_index_path)

