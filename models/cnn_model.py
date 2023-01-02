import tensorflow as tf
from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, MaxPool1D, Dropout, Flatten
from keras.initializers.initializers_v2 import Constant
from models.basic_model import BasicModel



class CNNModel(BasicModel):
    
    def __init__(self, input_shape, config, num_classes, word_embed_matrix, optimizer, scorer, logger):
        super().__init__(num_classes, config, optimizer, scorer, logger)
        self.create_model(input_shape, word_embed_matrix, config)


    def generate_sentence_level_features(self, embed_window_processing, num_filters):
        max_length = embed_window_processing.shape[1]
        outputs = []

        for kernel_size in [3, 4, 5]:
            # Output: (batch_size, max_sentence_length, num_conv_filters)
            conv_out = Conv1D(num_filters, kernel_size, padding='same', activation='relu')(embed_window_processing)
            # Output: (batch_size, 1, num_conv_filters)
            pool_out = MaxPool1D(max_length)(conv_out)
            outputs.append(pool_out)

        sentence_level_features = tf.reshape(tf.concat(outputs, axis=-1), [-1, 3*num_filters])
        return sentence_level_features


    def create_model(self, input_shape, word_embed_matrix, config):
        max_distance = config.max_distance 
        pos_embed_dim = config.pos_embed_dim
        conv_filters = config.conv_filters
        dropout_value = config.dropout
        max_length = config.max_sentence_length
        vocabulary_size, word_embed_dim = word_embed_matrix.shape

        input = Input(input_shape)

        words_indices = input[:, :max_length]
        pos_e1 = input[:, max_length : 2*max_length]
        pos_e2 = input[:, 2*max_length : 3*max_length]
        lexical_features = input[:, 3*max_length:]

        word_embed_layer = Embedding(vocabulary_size, word_embed_dim, Constant(word_embed_matrix), 
            input_length=max_length, mask_zero=True)
        pos_embed_layer_1 = Embedding(max_distance, pos_embed_dim)
        pos_embed_layer_2 = Embedding(max_distance, pos_embed_dim)

        embed_words = word_embed_layer(words_indices)
        # Output: (batch_size, 6*word_embed_dim)
        embed_lexical = word_embed_layer(lexical_features)
        embed_pos_e1 = pos_embed_layer_1(pos_e1)
        embed_pos_e2 = pos_embed_layer_2(pos_e2)
        # Output: (batch_size, max_sentence_length, word_embedding_dim + 2*pos_embedding_dim)
        embed_window_process = tf.concat([embed_words, embed_pos_e1, embed_pos_e2], axis=-1)

        dropout_out = Dropout(dropout_value)(embed_window_process)
        # Output: (batch_size, 3*num_conv_filters)
        sentence_level_features = self.generate_sentence_level_features(dropout_out, conv_filters)

        flatten_lexical = Flatten()(embed_lexical)
        # Output: (batch_size, 3*num_filters + 6*word_embed_dimension)
        all_features = tf.concat([flatten_lexical, sentence_level_features], axis=-1)

        dropout_out_2 = Dropout(dropout_value)(all_features)
        logits = Dense(self.num_classes)(dropout_out_2)
        self.model = Model(input, logits)
       