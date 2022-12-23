import tensorflow as tf
from keras.layers import TextVectorization
import numpy as np
from utils.file_io_utils import read_data_file, load_npy_file_to_np_array



def create_feature_to_index(train_data, max_tokens):
    vectorizer = TextVectorization(max_tokens)
    text_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(128)
    vectorizer.adapt(text_ds)
    vocabulary = vectorizer.get_vocabulary()
    return dict(zip(vocabulary, range(len(vocabulary))))


def create_word_to_vector(embedding_path):
    word_to_vector = {}
    file_lines = read_data_file(embedding_path)
    for line in file_lines:
        word, vector = line.strip().split(maxsplit=1)
        vector = np.fromstring(vector, dtype=float, sep=' ')
        word_to_vector[word] = vector
    return word_to_vector, len(vector)


def create_word_to_vector_separate_files(vocabulary_path, vectors_npy_path):
    vocabulary = []
    file_lines = read_data_file(vocabulary_path)
    for line in file_lines:
        vocabulary.append(line.strip())
    vectors = load_npy_file_to_np_array(vectors_npy_path)
    word_to_vector = dict(zip(vocabulary, vectors))
    return word_to_vector, len(vectors[0])


def create_embedding_matrix(word_to_vector, train_vocabulary, embedding_dim):
    num_tokens = len(train_vocabulary) + 2
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    hits = 0
    misses = 0

    values = list(word_to_vector.values())
    mean = np.mean(values)
    std = np.std(values)
    
    for i, word in enumerate(train_vocabulary):
        embedding_vector = word_to_vector.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        elif i > 0:
            # i = 0 is pad token and i = 1 is unknown token
            embedding_matrix[i] = np.random.normal(mean, std, embedding_dim)
            misses += 1
    
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix


def create_vectorizer(feature_to_index, max_length=None):
    vectorizer = TextVectorization(output_sequence_length=max_length)
    vectorizer.set_vocabulary(list(feature_to_index.keys()))
    return vectorizer


def map_to_indices(vectorizer, sentence_words):
    sentence = ' '.join(sentence_words)
    indices = vectorizer([sentence]).numpy()[0]
    return indices