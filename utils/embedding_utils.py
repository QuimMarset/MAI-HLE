import tensorflow as tf
from tensorflow import keras
import numpy as np



def create_text_vectorizer(train_data, max_sentence_length, max_tokens):
    vectorizer = keras.layers.TextVectorization(max_tokens, output_sequence_length=max_sentence_length)
    text_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(128)
    vectorizer.adapt(text_ds)
    return vectorizer


def create_pre_trained_embedding_dict(pre_trained_embed_file_path):
    embedding_dict = {}
    with open(pre_trained_embed_file_path, 'r', encoding='utf8') as file:
        for line in file:
            word, coefs = line.strip().split(maxsplit=1)
            coefs = np.fromstring(coefs, dtype=float, sep=' ')
            embedding_dict[word] = coefs
    return embedding_dict


def compute_oov_embedding(word_embeddings):
    embedding_vectors = word_embeddings.values()
    return np.mean(embedding_vectors, axis=0)


def create_embedding_matrix(pre_trained_embed_file_path, train_vocabulary, embedding_dim):
    embedding_dict = create_pre_trained_embedding_dict(pre_trained_embed_file_path)
    num_tokens = len(train_vocabulary) + 2
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    hits = 0
    misses = 0
    
    for i, word in enumerate(train_vocabulary):
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            # padding and oov token
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    
    print("Converted %d words (%d misses)" % (hits, misses))

    # Out of vocabulary token
    #embedding_matrix[1] = compute_oov_embedding(embedding_dict)
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
