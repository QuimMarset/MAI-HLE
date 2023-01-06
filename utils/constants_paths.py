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

bert_path = join_path(data_path, 'bert')

# =====================
# Processed data
# =====================

processed_data_path = join_path(data_path, 'processed_data')

train_data_path = join_path(processed_data_path, 'processed_train_data.json')
test_data_path = join_path(processed_data_path, 'processed_test_data.json')

train_data_cnn_path = join_path(processed_data_path, 'processed_train_data_cnn.json')
test_data_cnn_path = join_path(processed_data_path, 'processed_test_data_cnn.json')

train_labels_path = join_path(processed_data_path, 'train_labels.npy')
test_labels_path = join_path(processed_data_path, 'test_labels.npy')


# =====================
# Vocabularies
# =====================

vocabularies_path = join_path(data_path, 'vocabularies')
words_vocabulary_path = join_path(vocabularies_path, 'words_vocabulary.npy')
relative_positions_vocabulary_path = join_path(vocabularies_path, 'relative_positions.npy')

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

tacred_50_path = join_path(embeddings_path, 'tacred_50.txt')

google_300_path = join_path(embeddings_path, 'google_300.npy')




# =====================
# Config files
# =====================

configs_path = 'config_files'

entity_attention_config_path = join_path(configs_path, 'entity_attention_config.yaml')
rbert_config_path = join_path(configs_path, 'RBERT_config.yaml')
attention_lstm_config_path = join_path(configs_path, 'attention_lstm_config.yaml')
cnn_config_path = join_path(configs_path, 'cnn_config.yaml')
attention_lstm_bert_config_path = join_path(configs_path, 'attention_lstm_BERT_config.yaml')


# =====================
# Experiment results
# =====================

experiments_path = 'experiments'

