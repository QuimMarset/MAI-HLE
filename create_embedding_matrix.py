from constants_paths import *
from utils.file_io_utils import load_json_to_dict, save_array_to_npy_file, write_dict_to_json
from utils.embedding_utils import *



def create_glove_embedding_matrices(train_vocabulary):
    for embed_path, matrix_path in zip(glove_embeddings, glove_matrices):
        word_to_vector, embedding_dim = create_word_to_vector(embed_path)
        embedding_matrix = create_embedding_matrix(word_to_vector, train_vocabulary, embedding_dim)
        save_array_to_npy_file(embedding_matrix, matrix_path)


def create_senna_embedding_matrix(train_vocabulary):
    word_to_vector, embedding_dim = create_word_to_vector_separate_files(senna_words_path, senna_vectors_path)
    embedding_matrix = create_embedding_matrix(word_to_vector, train_vocabulary, embedding_dim)
    save_array_to_npy_file(embedding_matrix, senna_50_matrix_path)



if __name__ == '__main__':
       
    max_sentence_length = 100
    max_tokens = 20000
    
    train_data = load_json_to_dict(train_data_path)
    train_sentence_words = [' '.join(sentence_data['words']) for sentence_data in train_data.values()]

    word_to_index = create_word_to_index(train_sentence_words, max_sentence_length, max_tokens)
    write_dict_to_json(word_to_index, word_to_index_path)

    train_vocabulary = list(word_to_index.keys())
    create_glove_embedding_matrices(train_vocabulary)
    create_senna_embedding_matrix(train_vocabulary)