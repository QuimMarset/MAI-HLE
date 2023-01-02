from keras.layers import Layer, Bidirectional, LSTM, Dense, Dropout, Embedding
from keras.initializers.initializers_v2 import Constant
from keras import Input, Model
import tensorflow as tf
from models.basic_model import BasicModel



class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), 
            initializer='random_normal', trainable=True)
        super().build(input_shape)
 

    def call(self, H):
        M = tf.tanh(H)
        alpha = tf.math.softmax(tf.matmul(M, self.W), axis=1)
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), alpha)
        return tf.tanh(tf.squeeze(r, axis=-1))



class AttentionBiLSTM(BasicModel):


    def __init__(self, num_classes, word_embed_matrix, config, optimizer, scorer, logger):
        super().__init__(num_classes, config, optimizer, scorer, logger)
        self.create_model(word_embed_matrix, config)


    def create_model(self, word_embed_matrix, config):
        max_length = config.max_length
        num_words, embed_dim = word_embed_matrix.shape
        lstm_units = config.lstm_units
        emb_dropout_value = config.dropout_embed
        lstm_dropout_value = config.dropout_lstm
        attention_dropout_value = config.dropout_attention
        
        input = Input((max_length))

        word_embed_layer = Embedding(num_words, embed_dim, Constant(word_embed_matrix), 
            input_length=max_length, mask_zero=True)

        word_embed = word_embed_layer(input)
        embed_dropout = Dropout(emb_dropout_value)(word_embed)

        lstm = LSTM(lstm_units, return_sequences=True)
        bi_lstm = Bidirectional(lstm, merge_mode='sum')(embed_dropout)

        bi_lstm_dropout = Dropout(lstm_dropout_value)(bi_lstm)

        attention = Attention()(bi_lstm_dropout)
        attention_dropout = Dropout(attention_dropout_value)(attention)

        output = Dense(self.num_classes, activation='softmax')(attention_dropout)

        self.model = Model(input, output)