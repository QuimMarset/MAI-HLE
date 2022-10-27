from utils.path_utils import join_path


data_path = 'data'
processed_data_path = 'processed_data'
embeddings_path = 'embedding_matrices'
features_path = 'extracted_features'
experiments_path = 'experiments'

raw_train_data_path = join_path(data_path, 'TRAIN_FILE.txt')
raw_test_data_path = join_path(data_path, 'TEST_FILE_FULL.txt')

train_data_path = join_path(processed_data_path, 'processed_train_data.json')
test_data_path = join_path(processed_data_path, 'processed_test_data.json')

train_labels_path = join_path(processed_data_path, 'train_labels.npy')
test_labels_path = join_path(processed_data_path, 'test_labels.npy')

class_to_index_path = join_path(processed_data_path, 'class_to_index.json')
word_to_index_path = join_path(embeddings_path, 'word_to_index.json')

embedding_matrix_path = join_path(embeddings_path, 'embedding_matrix_glove_100.npy')
