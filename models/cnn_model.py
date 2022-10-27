import tensorflow as tf
from tensorflow import keras
from models.basic_model import BasicModel



class CNNModel(BasicModel):
    
    def __init__(self, input_shape, num_classes, word_embed_matrix, pos_embed_dim, max_distance, 
        max_length, window_size, conv_filters, dense_units, dropout_value, seed):
        
        super().__init__(num_classes, seed)
        self.create_model(input_shape, word_embed_matrix, max_length, max_distance, pos_embed_dim, conv_filters, window_size, dense_units, dropout_value)


    def create_model(self, input_shape, word_embed_matrix, max_length, max_distance, pos_embed_dim, conv_filters, window_size, dense_units, dropout_value):
        input = keras.Input(input_shape)

        words_indices = keras.layers.Lambda(lambda batch: batch[:, :max_length])(input)
        pos_e1 = keras.layers.Lambda(lambda batch: batch[:, max_length:2*max_length])(input)
        pos_e2 = keras.layers.Lambda(lambda batch: batch[:, 2*max_length:3*max_length])(input)
        lexical_features = keras.layers.Lambda(lambda batch: batch[:, 3*max_length:])(input)

        vocabulary_size, word_embed_dim = word_embed_matrix.shape
        word_embed_layer = keras.layers.Embedding(vocabulary_size, word_embed_dim, 
            embeddings_initializer=keras.initializers.Constant(word_embed_matrix), trainable=False)

        embed_words = word_embed_layer(words_indices)
        embed_lexical = word_embed_layer(lexical_features)

        pos_embedding_layer_1 = keras.layers.Embedding(max_distance, pos_embed_dim)
        pos_embedding_layer_2 = keras.layers.Embedding(max_distance, pos_embed_dim)

        embed_pos_e1 = pos_embedding_layer_1(pos_e1)
        embed_pos_e2 = pos_embedding_layer_2(pos_e2)

        # Output: (batch_size, max_sentence_length, word_embedding_dim + 2*pos_embedding_dim)
        sentence_level_features = tf.concat([embed_words, embed_pos_e1, embed_pos_e2], axis=-1)

        dropout_out = keras.layers.Dropout(dropout_value)(sentence_level_features)

        # Output: (batch_size, max_sentence_length, num_conv_filters)
        conv_out = keras.layers.Conv1D(conv_filters, window_size, padding='same')(dropout_out)
        # Output: (batch_size, 1, num_conv_filters)
        max_pool_out = keras.layers.MaxPooling1D(max_length)(conv_out)
        # Output: (batch_size, dense_units)
        dense_out = keras.layers.Dense(dense_units, activation='tanh')(keras.layers.Flatten()(max_pool_out))

        dropout_out_2 = keras.layers.Dropout(dropout_value)(dense_out)

        # Output: (batch_size, dense_units + 5*word_embed_dimension)
        embed_lexical = keras.layers.Flatten()(embed_lexical)
        all_features = tf.concat([embed_lexical, dropout_out_2], axis=-1)

        dropout_out_3 = keras.layers.Dropout(dropout_value)(all_features)

        classif = keras.layers.Dense(self.num_classes, activation='softmax')(dropout_out_3)

        self.model = keras.Model(input, classif)
       