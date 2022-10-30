import tensorflow as tf
from tensorflow import keras
from models.basic_model import BasicModel



class CNNModel(BasicModel):
    
    def __init__(self, input_shape, num_classes, word_embed_matrix, pos_embed_dim, max_distance, conv_filters, dropout_value, train_word, max_length, seed):
        super().__init__(num_classes, seed)
        self.create_model(input_shape, word_embed_matrix, max_distance, pos_embed_dim, conv_filters, dropout_value, train_word, max_length)


    def create_embedding_layers(self, word_embed_matrix, max_distance, pos_embed_dim, train_word_embed):
        vocabulary_size, word_embed_dim = word_embed_matrix.shape
        word_embed_layer = keras.layers.Embedding(vocabulary_size, word_embed_dim, 
            embeddings_initializer=keras.initializers.Constant(word_embed_matrix), trainable=train_word_embed)

        pos_embed_layer_1 = keras.layers.Embedding(max_distance, pos_embed_dim)
        pos_embed_layer_2 = keras.layers.Embedding(max_distance, pos_embed_dim)

        return word_embed_layer, pos_embed_layer_1, pos_embed_layer_2

    
    def embed_input_features(self, input, word_embed_layer, pos_embed_layer_1, pos_embed_layer_2, max_length):
        words_indices = keras.layers.Lambda(lambda batch: batch[:, :max_length])(input)
        pos_e1 = keras.layers.Lambda(lambda batch: batch[:, max_length:2*max_length])(input)
        pos_e2 = keras.layers.Lambda(lambda batch: batch[:, 2*max_length:3*max_length])(input)
        lexical_features = keras.layers.Lambda(lambda batch: batch[:, 3*max_length:])(input)

        embed_words = word_embed_layer(words_indices)
        # Output: (batch_size, 6*word_embed_dim)
        embed_lexical = word_embed_layer(lexical_features)
        embed_pos_e1 = pos_embed_layer_1(pos_e1)
        embed_pos_e2 = pos_embed_layer_2(pos_e2)
        # Output: (batch_size, max_sentence_length, word_embedding_dim + 2*pos_embedding_dim)
        embed_window_proces = tf.concat([embed_words, embed_pos_e1, embed_pos_e2], axis=-1)
        return embed_lexical, embed_window_proces


    def generate_sentence_level_features(self, embed_window_processing, num_filters):
        max_length = embed_window_processing.shape[1]
        outputs = []
        for kernel_size in [3, 4, 5]:

            # Output: (batch_size, max_sentence_length, num_conv_filters)
            conv_out = keras.layers.Conv1D(num_filters, kernel_size, padding='same', activation='relu')(embed_window_processing)
            # Output: (batch_size, 1, num_conv_filters)
            pool_out = keras.layers.MaxPooling1D(max_length)(conv_out)

            outputs.append(pool_out)

        sentence_level_features = tf.reshape(tf.concat(outputs, axis=-1), [-1, 3*num_filters])
        return sentence_level_features


    def create_model(self, input_shape, word_embed_matrix, max_distance, pos_embed_dim, conv_filters, dropout_value, train_word, max_length):
        input = keras.Input(input_shape)

        word_embed_layer, pos_embed_layer_1, pos_embed_layer_2 = self.create_embedding_layers(word_embed_matrix, max_distance, pos_embed_dim, train_word)
        embed_lexical, embed_window_proces = self.embed_input_features(input, word_embed_layer, pos_embed_layer_1, pos_embed_layer_2, max_length)

        dropout_out = keras.layers.Dropout(dropout_value)(embed_window_proces)
        # Output: (batch_size, 3*num_conv_filters)
        sentence_level_features = self.generate_sentence_level_features(dropout_out, conv_filters)

        flatten_lexical = keras.layers.Flatten()(embed_lexical)
        # Output: (batch_size, 3*num_filters + 6*word_embed_dimension)
        all_features = tf.concat([flatten_lexical, sentence_level_features], axis=-1)

        dropout_out_2 = keras.layers.Dropout(dropout_value)(all_features)

        classif = keras.layers.Dense(self.num_classes, activation='softmax',
            kernel_regularizer=keras.regularizers.L2(0.01))(dropout_out_2)

        self.model = keras.Model(input, classif)
       