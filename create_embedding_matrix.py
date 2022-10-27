from matplotlib.pyplot import text
from utils.file_io_utils import load_json_to_dict, save_array_to_npy_file, write_dict_to_json
from utils.path_utils import join_path, create_folder
from utils.embedding_utils import *



if __name__ == '__main__':
    
    embedding_matrices_path = 'embedding_matrices'
    create_folder(embedding_matrices_path)
    train_data_path = join_path('processed_data', 'processed_train_data.json')
    glove_embeddings_path = join_path('glove_embeddings', 'glove.6B.100d.txt')
    
    max_sentence_length = 100
    max_tokens = 20000
    embedding_dim = 100

    train_data = load_json_to_dict(train_data_path)
    train_sentence_words = [' '.join(sentence_data['words']) for sentence_data in train_data.values()]

    text_vectorizer = create_text_vectorizer(train_sentence_words, max_sentence_length, max_tokens)
    train_vocabulary = text_vectorizer.get_vocabulary()
    word_to_index = dict(zip(train_vocabulary, range(len(train_vocabulary))))
    write_dict_to_json(word_to_index, embedding_matrices_path, 'word_to_index')
    
    embedding_matrix = create_embedding_matrix(glove_embeddings_path, train_vocabulary, embedding_dim)
    save_array_to_npy_file(embedding_matrix, embedding_matrices_path, 'embedding_matrix_glove_100')