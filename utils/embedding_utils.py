import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils.file_io_utils import read_data_file, load_npy_file_to_np_array



def create_word_to_index(train_data, max_sentence_length, max_tokens):
    vectorizer = keras.layers.TextVectorization(max_tokens, output_sequence_length=max_sentence_length)
    text_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(128)
    vectorizer.adapt(text_ds)
    vocabulary = vectorizer.get_vocabulary()
    word_to_index = dict(zip(vocabulary, range(len(vocabulary))))
    return word_to_index


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
    
    for i, word in enumerate(train_vocabulary):
        embedding_vector = word_to_vector.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        elif i > 2:
            # i = 0 is pad token and i = 2 is unknown token
            embedding_matrix[i] = np.random.normal(0, 0.1, embedding_dim)
            misses += 1
    
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix


def get_word_index(word, word_to_index):
    if word in word_to_index:
        return word_to_index[word]
    elif word.lower() in word_to_index:
        return word_to_index[word.lower()]
    return word_to_index['[UNK]']


def preprocess_sentence_for_embedding(sentence_words, word_to_index, max_length):
    sentence_indices = [get_word_index(word, word_to_index) for word in sentence_words]
    num_words = len(sentence_words)
    padding_index = word_to_index['']

    if num_words > max_length:
        sentence_indices = sentence_indices[:max_length]
    else:
        sentence_indices.extend([padding_index] * (max_length - num_words))

    return sentence_indices
