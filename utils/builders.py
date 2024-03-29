from keras.optimizer_v2 import adam, adadelta
from utils.constants_paths import *
from utils.file_io_utils import *
from features_extractors import drnn_features, attention_bi_lstm_features, entity_aware_attention_features
from models import rnn_dependency_paths, attention_bi_lstm
from models.entity_aware_attention.entity_aware_attention_bi_lstm import EntityAttentionLSTM
from utils.label_dicts import class2label



# =====================
# Configurations
# =====================

def load_configuration(method):
    if method == 'cnn':
        config_path = cnn_config_path
    elif method == 'cnn_rnn':
        config_path = cnn_rnn_config_path
    elif method == 'drnn':
        config_path = drnn_config_path
    elif method == 'attention_bi_lstm':
        config_path = attention_bi_lstm_config_path
    elif method == 'entity_attention_lstm':
        config_path = entity_attention_lstm_config_path
    else:
        raise NotImplementedError(method)

    return read_yaml_config(config_path)


# =======================
# Pre-trained embeddings
# =======================

def load_word_embed_matrix(pre_trained_name):
    if pre_trained_name == 'glove_50':
        embed_path = glove_50_matrix_path

    elif pre_trained_name == 'glove_100':
        embed_path = glove_100_matrix_path

    elif pre_trained_name == 'glove_200':
        embed_path = glove_200_matrix_path

    elif pre_trained_name == 'glove_300':
        embed_path = glove_300_matrix_path

    elif pre_trained_name == 'glove_300_2':
        embed_path = glove_300_2_matrix_path

    elif pre_trained_name == 'senna_50':
        embed_path = senna_50_matrix_path
    
    return load_npy_file_to_np_array(embed_path)


# =====================
# Models
# =====================

def create_model(method, config, input_shape, optimizer, scorer, logger):
    if method == 'cnn':
        return None

    elif method == 'cnn_rnn':
        return None

    elif method == 'drnn':
        return create_rnn_dep_path_model(config)

    elif method == 'attention_bi_lstm':
        return create_attention_bi_lstm_model(config)

    elif method == 'entity_attention_lstm':
        return create_entity_attention_lstm_model(config, input_shape, optimizer, scorer, logger)

    else:
        raise NotImplementedError(method)


def create_rnn_dep_path_model(config):
    word_embed_matrix = load_word_embed_matrix(config.pre_trained_name)

    num_pos_tags = len(load_json_to_dict(pos_tag_to_index_path))
    num_lemmas = len(load_json_to_dict(lemma_to_index_path))
    num_relations = len(load_json_to_dict(relation_to_index_path))

    num_classes = len(class2label)

    return rnn_dependency_paths.RNNDepPathsModel(num_classes, word_embed_matrix, 
        config.embed_dim, num_pos_tags, num_lemmas, num_relations, config.max_length, 
        config.dense_units, config.dropout_embed, config.dropout_dense, config.l2_coef, config.seed)


def create_attention_bi_lstm_model(config):
    word_embed_matrix = load_word_embed_matrix(config.pre_trained_name)
    num_classes = len(class2label)
    return attention_bi_lstm.AttentionBiLSTM(num_classes, word_embed_matrix, config)


def create_entity_attention_lstm_model(config, input_shape, optimizer, scorer, logger):
    word_embed_matrix = load_word_embed_matrix(config.pre_trained_name)
    num_rel_pos = len(load_json_to_dict(relative_position_to_index_path))
    return EntityAttentionLSTM(input_shape, config, len(class2label), 
        word_embed_matrix, num_rel_pos, optimizer, scorer, logger)



# =====================
# Feature Extractors
# =====================

def create_feature_extractor(method, config):
    if method == 'cnn':
        return None

    elif method == 'cnn_rnn':
        return None

    elif method == 'drnn':
        return create_drnn_feature_extractor(config)

    elif method == 'attention_bi_lstm':
        return create_attention_bi_lstm_extractor(config)

    elif method == 'entity_attention_lstm':
        return create_entity_attention_lstm_extractor(config)

    else:
        raise NotImplementedError(method)


def create_drnn_feature_extractor(config):
    word_to_index = load_json_to_dict(word_to_index_path)
    pos_tag_to_index = load_json_to_dict(pos_tag_to_index_path)
    lemma_to_index = load_json_to_dict(lemma_to_index_path)
    relation_to_index = load_json_to_dict(relation_to_index_path)

    return drnn_features.DRNNFeatureExtractor(word_to_index, pos_tag_to_index, 
        relation_to_index, lemma_to_index, config.max_length)


def create_attention_bi_lstm_extractor(config):
    word_to_index = load_json_to_dict(word_to_index_path)
    return attention_bi_lstm_features.AttentionBiLSTMFeatureExtractor(word_to_index, config.max_length)


def create_entity_attention_lstm_extractor(config):
    word_to_index = load_json_to_dict(word_to_index_path)
    rel_pos_to_index = load_json_to_dict(relative_position_to_index_path)
    return entity_aware_attention_features.EntityAwareAttentionFeatureExtractor(word_to_index, rel_pos_to_index,
        config.max_sentence_length)


# =====================
# Optimizers
# =====================


def create_optimizer(name, learning_rate, gradient_clip, lr_decay):
    if name == 'adam':
        return adam.Adam(learning_rate)
        
    elif name == 'adadelta':
        return adadelta.Adadelta(learning_rate, rho=lr_decay, epsilon=1e-6)

    else:
        raise NotImplementedError(name)