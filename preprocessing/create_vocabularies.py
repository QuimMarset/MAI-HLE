import sys
sys.path.append('./')
from utils.file_io_utils import load_json_to_dict, write_dict_to_json
from utils.embedding_utils import create_feature_to_index
from utils.constants_paths import *
from utils.path_utils import create_folder



def create_word_to_index(train_data, max_tokens, save_path):
    word_sentences = [sentence_data['sentence_tags'] for sentence_data in train_data.values()]
    word_to_index = create_feature_to_index(word_sentences, max_tokens)
    word_to_index['<e1>'] = len(word_to_index)
    word_to_index['<e2>'] = len(word_to_index)
    word_to_index['</e1>'] = len(word_to_index)
    word_to_index['</e2>'] = len(word_to_index)
    write_dict_to_json(word_to_index, save_path)


def create_pos_tag_to_index(train_data, max_tokens, save_path):
    pos_tag_sentences = [' '.join([token['pos'] for token in sentence_data['words']]) for sentence_data in train_data.values()]
    pos_tag_to_index = create_feature_to_index(pos_tag_sentences, max_tokens)
    write_dict_to_json(pos_tag_to_index, save_path)


def create_lemma_to_index(train_data, max_tokens, save_path):
    lemma_sentences = [' '.join([token['lemma'] for token in sentence_data['words']]) for sentence_data in train_data.values()]
    lemma_to_index = create_feature_to_index(lemma_sentences, max_tokens)
    write_dict_to_json(lemma_to_index, save_path)


def create_relation_to_index(train_data, max_tokens, save_path):
    relation_sentences = [' '.join([token['rel'] for token in sentence_data['words']]) for sentence_data in train_data.values()]
    relation_to_index = create_feature_to_index(relation_sentences, max_tokens)
    write_dict_to_json(relation_to_index, save_path)



if __name__ == '__main__':
    
    max_tokens = 20000
    train_data = load_json_to_dict(train_data_path)

    create_folder(vocabularies_path)

    create_word_to_index(train_data, max_tokens, word_to_index_path)
    create_pos_tag_to_index(train_data, max_tokens, pos_tag_to_index_path)
    #create_lemma_to_index(train_data, max_tokens, lemma_to_index_path)
    create_relation_to_index(train_data, max_tokens, relation_to_index_path)