import tensorflow as tf
from keras.initializers.initializers_v2 import GlorotNormal
from keras.layers import Dense
from keras import Model
from models.entity_aware_attention.latent_entity_typing import LatentEntityType




class EntityAwareAttention(Model):
    def __init__(self, num_types, latent_size, attention_size, **kwargs):
        super().__init__(**kwargs)
        self.latent_layer = LatentEntityType(num_types, latent_size)
        self.dense_pos_features = Dense(attention_size, use_bias=False, kernel_initializer=GlorotNormal())
        self.dense_entity_features = Dense(attention_size, use_bias=False, kernel_initializer=GlorotNormal())
        self.v = tf.Variable(GlorotNormal()(shape=(attention_size,)), trainable=True)

    
    def get_entity_last_hidden_state(self, word_hiddens, end_indices):
        # BxH, Bx1
        batch_size = tf.shape(word_hiddens)[0]
        batch_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=-1)
        entity_indices = tf.expand_dims(tf.cast(end_indices, dtype=tf.int32), axis=-1)
        gather_indices = tf.concat([batch_indices, entity_indices], axis=-1)
        return tf.gather_nd(word_hiddens, gather_indices)

    
    def call(self, inputs):
        # BxSxH
        word_hiddens = inputs[0]
        # BxSxP
        pos_e1_embeddings = inputs[1]
        pos_e2_embeddings = inputs[2]
        # Bx1
        e1_end = inputs[3]
        e2_end = inputs[4]

        # BxH
        e1_h = self.get_entity_last_hidden_state(word_hiddens, e1_end)
        e2_h = self.get_entity_last_hidden_state(word_hiddens, e2_end)
        # BxL
        e1_type = self.latent_layer(e1_h)
        e2_type = self.latent_layer(e2_h)

        # Bx(2*H + 2*L)
        entity_features = tf.concat([e1_h, e1_type, e2_h, e2_type], axis=-1)
        # BxSx(W + 2*P)
        pos_features = tf.concat([word_hiddens, pos_e1_embeddings, pos_e2_embeddings], axis=-1)

        # BxSxA
        dense_pos_output = self.dense_pos_features(pos_features)
        # BxA
        dense_entity_output = self.dense_entity_features(entity_features)
        seq_length = dense_pos_output.shape[1]
        dense_entity_output = tf.repeat(dense_entity_output, seq_length, axis=1)
        # BxSxA
        dense_entity_output = tf.reshape(dense_entity_output, [-1, seq_length, dense_pos_output.shape[-1]])

        # BxSxA
        u = tf.tanh(dense_pos_output + dense_entity_output)
        # BxS
        vu = tf.tensordot(u, self.v, axes=1)
        # BxS
        alpha = tf.nn.softmax(vu)
        # BxH
        z = tf.reduce_sum(word_hiddens * tf.expand_dims(alpha, axis=-1), axis=1)
        return z