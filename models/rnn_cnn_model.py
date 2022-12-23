import tensorflow as tf
from tensorflow import keras
from models.basic_model import BasicModel



class CNNRNNModel(BasicModel):

    def __init__(self, num_classes, input_shape, word_embed_matrix, num_pos, num_lemmas, emb_dim_pos, emb_dim_lemmas, seed):
        super().__init__(num_classes, seed)
        self.create_model(input_shape, word_embed_matrix, num_pos, num_lemmas, emb_dim_pos, emb_dim_lemmas)


    def create_model(self, input_shape, word_embed_matrix, num_pos, num_lemmas, emb_dim_pos, emb_dim_lemmas):
        input = keras.Input(input_shape)
        lc_words = input[:, :, 0]
        lemmas = input[:, :, 1]
        pos = input[:, :, 2]

        vocab_size, emb_dim = word_embed_matrix.shape
        emb_words = keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True,
            embeddings_initializer=keras.initializers.Constant(word_embed_matrix))(lc_words)

        emb_pos = keras.layers.Embedding(input_dim=num_pos, output_dim=emb_dim_pos, mask_zero=True)(pos)
        emb_lemmas =  keras.layers.Embedding(input_dim=num_lemmas, output_dim=emb_dim_lemmas, mask_zero=True)(lemmas)

        concat = tf.concat([emb_words, emb_pos, emb_lemmas], axis=-1)

        conv_1 = keras.layers.Conv1D(150, 3, activation='relu', padding='same')(concat)
        bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=200, return_sequences=True))(conv_1)
        dense_1 = keras.layers.TimeDistributed(keras.layers.Dense(100, activation='relu'))(bi_lstm)
        conv_2 = keras.layers.Conv1D(50, 5, activation='relu')(dense_1)
        pooling = keras.layers.GlobalMaxPooling1D()(conv_2)
        dense_2 = keras.layers.Dense(50, activation='relu')(pooling)

        classif = keras.layers.Dense(self.num_classes, activation='softmax')(dense_2)

        self.model = keras.Model(input, classif)