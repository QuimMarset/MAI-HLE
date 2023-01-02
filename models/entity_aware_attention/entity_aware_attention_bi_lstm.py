from keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional, BatchNormalization
from keras.initializers.initializers_v2 import Constant
from keras import Input, Model
from models.basic_model import BasicModel
from models.entity_aware_attention.entity_aware_attention_layer import EntityAwareAttention
from models.entity_aware_attention.self_attention import MultiHeadAttention



class EntityAttentionLSTM(BasicModel):


    def __init__(self, input_shape, config, num_classes, word_embedding_matrix, num_rel_pos, optimizer, scorer, logger):
        super().__init__(num_classes, config, optimizer, scorer, logger)
        self.create_model(input_shape, word_embedding_matrix, config, num_classes, num_rel_pos)


    def create_model(self, input_shape, word_embedding_matrix, config, num_classes, num_rel_pos):
        max_length = config.max_sentence_length
        num_words, word_embed_dim = word_embedding_matrix.shape

        input = Input(input_shape)

        # Input separation
        e1_end = input[:, 0]
        e2_end = input[:, 1]
        word_indices = input[:, 2: 2 + max_length]
        rel_pos_1_indices = input[:, 2 + max_length : 2 + 2*max_length]
        rel_pos_2_indices = input[:, 2 + 2*max_length : 2 + 3*max_length]

        # Word embedding generation
        word_embed_layer = Embedding(num_words, word_embed_dim, Constant(word_embedding_matrix), 
            input_length=max_length, mask_zero=True)
        word_embed = word_embed_layer(word_indices)

        word_embed = BatchNormalization()(word_embed)

        word_embed_dropout = Dropout(config.word_dropout)(word_embed)

        # Relative position embedding generation
        rel_pos_embed_layer = Embedding(num_rel_pos, config.pos_embed_dim, input_length=max_length, mask_zero=True)
        rel_pos_1_embed = rel_pos_embed_layer(rel_pos_1_indices)
        rel_pos_2_embed = rel_pos_embed_layer(rel_pos_2_indices)

        rel_pos_1_embed = BatchNormalization()(rel_pos_1_embed)
        rel_pos_2_embed = BatchNormalization()(rel_pos_2_embed)

        # Multi-head Self-Attention
        mhsa_layer = MultiHeadAttention(config.num_heads, word_embed_dim)
        improved_word_embed = mhsa_layer([word_embed_dropout, word_embed_dropout, word_embed_dropout])

        improved_word_embed = BatchNormalization()(improved_word_embed)

        # Bi-LSTM
        lstm = LSTM(config.hidden_size, return_sequences=True, dropout=config.lstm_dropout)
        bi_lstm_output = Bidirectional(lstm)(improved_word_embed)

        bi_lstm_output = BatchNormalization()(bi_lstm_output)

        # Entity-aware Attention
        latent_size = bi_lstm_output.shape[-1]
        entity_aware_layer = EntityAwareAttention(config.latent_types, latent_size, config.attention_size)
        z_vector = entity_aware_layer([bi_lstm_output, rel_pos_1_embed, rel_pos_2_embed, e1_end, e2_end])

        z_vector = BatchNormalization()(z_vector)

        # Output
        z_dropout = Dropout(config.z_dropout)(z_vector)
        output = Dense(num_classes)(z_dropout)

        self.model = Model(input, output)
