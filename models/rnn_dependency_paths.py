import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, MaxPooling1D, Dropout
from keras import Input, Model
from keras.initializers.initializers_v2 import Constant
from keras.regularizers import L2
from models.basic_model import BasicModel



class RNNDepPathsModel(BasicModel):

    def __init__(self, num_classes, word_embed_matrix, others_embed_dim, num_pos_tags, num_lemmas, num_relations, max_length, 
        dense_units, dropout_embed, dropout_dense, l2_coef, seed):

        super().__init__(num_classes, seed)
        self.create_model(word_embed_matrix, others_embed_dim, num_pos_tags, num_lemmas, num_relations, max_length,
            dense_units, dropout_embed, dropout_dense, l2_coef)


    def create_model(self, word_embed_matrix, others_embed_dim, num_pos_tags, num_lemmas, num_relations, max_length, 
        dense_units, dropout_embed, dropout_dense, l2_coef):
        
        input = Input((2, max_length, 4))
        path_1_input = input[:, 0]
        path_2_input = input[:, 1]

        num_words, word_embed_dim = word_embed_matrix.shape
        word_embedding = Embedding(num_words, word_embed_dim, embeddings_initializer=Constant(word_embed_matrix), mask_zero=True)
        pos_tag_embedding = Embedding(num_pos_tags, others_embed_dim, mask_zero=True)
        lemma_embedding = Embedding(num_lemmas, others_embed_dim, mask_zero=True)
        relation_embedding = Embedding(num_relations, others_embed_dim, mask_zero=True)

        words_output = self.__compute_word_channel(path_1_input, path_2_input, word_embedding, dense_units, 
            dropout_embed, dropout_dense, l2_coef)
        
        pos_tags_output = self.__compute_pos_tag_channel(path_1_input, path_2_input, pos_tag_embedding, 
            dense_units, dropout_embed, dropout_dense, l2_coef)
        
        lemmas_output = self.__compute_lemma_channel(path_1_input, path_2_input, lemma_embedding, 
            dense_units, dropout_embed, dropout_dense, l2_coef)
        
        relations_output = self.__compute_relation_channel(path_1_input, path_2_input, relation_embedding, 
            dense_units, dropout_embed, dropout_dense, l2_coef)

        concat = tf.concat([words_output, pos_tags_output, lemmas_output, relations_output], axis=-1)
        dense = Dense(dense_units, activation='relu', kernel_regularizer=L2(l2_coef))(concat)
        dropout = Dropout(dropout_dense)(dense)
        classif = Dense(self.num_classes, activation='softmax')(dropout)

        self.model = Model(input, classif)


    def __compute_word_channel(self, path_1_input, path_2_input, word_embedding, dense_units, dropout_dense, dropout_embed, l2_coef):
        return self.__compute_channel(path_1_input, path_2_input, 0, word_embedding, dense_units, dropout_dense, dropout_embed, l2_coef)


    def __compute_pos_tag_channel(self, path_1_input, path_2_input, pos_tag_embedding, dense_units, dropout_dense, dropout_embed, l2_coef):
        return self.__compute_channel(path_1_input, path_2_input, 1, pos_tag_embedding, dense_units, dropout_dense, dropout_embed, l2_coef)


    def __compute_lemma_channel(self, path_1_input, path_2_input, lemma_embedding, dense_units, dropout_dense, dropout_embed, l2_coef):
        return self.__compute_channel(path_1_input, path_2_input, 2, lemma_embedding, dense_units, dropout_dense, dropout_embed, l2_coef)


    def __compute_relation_channel(self, path_1_input, path_2_input, relations_embedding, dense_units, dropout_dense, dropout_embed, l2_coef):
        return self.__compute_channel(path_1_input, path_2_input, 3, relations_embedding, dense_units, dropout_dense, dropout_embed, l2_coef)


    def __compute_channel(self, path_1_input, path_2_input, channel_index, embedding_layer, dense_units, dropout_dense, dropout_embed, l2_coef):
        features_1 = path_1_input[:, :, channel_index]
        features_2 = path_2_input[:, :, channel_index]

        embed_1 = embedding_layer(features_1)
        embed_2 = embedding_layer(features_2)

        dropout_1 = Dropout(dropout_embed)(embed_1)
        dropout_2 = Dropout(dropout_embed)(embed_2)

        lstm_1 = LSTM(embed_1.shape[-1], return_sequences=True, kernel_regularizer=L2(l2_coef), recurrent_regularizer=L2(l2_coef))(dropout_1)
        lstm_2 = LSTM(embed_2.shape[-1], return_sequences=True, kernel_regularizer=L2(l2_coef), recurrent_regularizer=L2(l2_coef))(dropout_2)

        pool_1 = MaxPooling1D(embed_1.shape[1])(lstm_1)
        pool_2 = MaxPooling1D(embed_2.shape[1])(lstm_2)

        outputs = tf.concat([pool_1, pool_2], axis=-1)
        outputs = tf.squeeze(outputs, axis=1)

        dense = Dense(dense_units, activation='relu', kernel_regularizer=L2(l2_coef))(outputs)
        dropout = Dropout(dropout_dense)(dense)

        return dropout