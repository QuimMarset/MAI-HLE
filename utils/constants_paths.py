from utils.path_utils import join_path


data_path = 'data'

# =====================
# Original data
# =====================

raw_data_path = join_path(data_path, 'raw_data')
raw_train_data_path = join_path(raw_data_path, 'TRAIN_FILE.txt')
raw_test_data_path = join_path(raw_data_path, 'TEST_FILE_FULL.txt')
true_test_predictions_path = join_path(raw_data_path, 'TEST_FILE_KEY.txt')

perl_scorer_path = 'scorer.pl'

# =====================
# Processed data
# =====================

processed_data_path = join_path(data_path, 'processed_data')

train_data_path = join_path(processed_data_path, 'processed_train_data.json')
test_data_path = join_path(processed_data_path, 'processed_test_data.json')
train_labels_path = join_path(processed_data_path, 'train_labels.npy')
test_labels_path = join_path(processed_data_path, 'test_labels.npy')

train_data_2_path = join_path(processed_data_path, 'processed_train_data_2.json')
test_data_2_path = join_path(processed_data_path, 'processed_test_data_2.json')
train_labels_2_path = join_path(processed_data_path, 'train_labels_2.npy')
test_labels_2_path = join_path(processed_data_path, 'test_labels_2.npy')


# =====================
# Vocabularies
# =====================

vocabularies_path = join_path(data_path, 'vocabularies')
word_to_index_path = join_path(vocabularies_path, 'word_to_index.json')
pos_tag_to_index_path = join_path(vocabularies_path, 'pos_tag_to_index.json')
lemma_to_index_path = join_path(vocabularies_path, 'lemma_to_index.json')
relation_to_index_path = join_path(vocabularies_path, 'relation_to_index.json')
relative_position_to_index_path = join_path(vocabularies_path, 'rel_position_to_index.json')

# =====================
# Word embeddings
# =====================

embeddings_path = join_path(data_path, 'embeddings')

glove_path = join_path(embeddings_path, 'glove')
glove_50_path = join_path(glove_path, 'glove.6B.50d.txt')
glove_100_path = join_path(glove_path, 'glove.6B.100d.txt')
glove_200_path = join_path(glove_path, 'glove.6B.200d.txt')
glove_300_path = join_path(glove_path, 'glove.6B.300d.txt')
glove_300_2_path = join_path(glove_path, 'glove.840B.300d.txt')

senna_path = join_path(embeddings_path, 'senna')
senna_words_path = join_path(senna_path, 'senna_words.lst')
senna_vectors_path = join_path(senna_path, 'embed50.senna.npy')

embedding_matrices_path = join_path(embeddings_path, 'matrices')
glove_50_matrix_path = join_path(embedding_matrices_path, 'embedding_matrix_glove_50.npy')
glove_100_matrix_path = join_path(embedding_matrices_path, 'embedding_matrix_glove_100.npy')
glove_200_matrix_path = join_path(embedding_matrices_path, 'embedding_matrix_glove_200.npy')
glove_300_matrix_path = join_path(embedding_matrices_path, 'embedding_matrix_glove_300.npy')
glove_300_2_matrix_path = join_path(embedding_matrices_path, 'embedding_matrix_glove_300_2.npy')
senna_50_matrix_path = join_path(embedding_matrices_path, 'embedding_matrix_senna_50.npy')

glove_embeddings = [glove_50_path, glove_100_path, glove_200_path, glove_300_path, glove_300_2_path]
glove_matrices = [glove_50_matrix_path, glove_100_matrix_path, glove_200_matrix_path, glove_300_matrix_path, glove_300_2_matrix_path]

# =====================
# Config files
# =====================

configs_path = 'config_files'
cnn_config_path = join_path(configs_path, 'cnn_config.yaml')
cnn_rnn_config_path = join_path(configs_path, 'cnn_rnn_config.yaml')
drnn_config_path = join_path(configs_path, 'drnn_config.yaml')
attention_bi_lstm_config_path = join_path(configs_path, 'config_attention_bi_lstm.yaml')
entity_attention_lstm_config_path = join_path(configs_path, 'entity_attention_lstm_config.yaml')


# =====================
# Extracted features
# =====================

extracted_features_path = join_path(data_path, 'extracted_features')
cnn_train_features_path = join_path(extracted_features_path, 'train_cnn_features.npy')
cnn_test_features_path = join_path(extracted_features_path, 'test_cnn_features.npy')

# =====================
# Experiment results
# =====================

experiments_path = 'experiments'

