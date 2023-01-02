import tensorflow as tf
from keras import Model
from keras.layers import Dense, LayerNormalization, Dropout
from keras.initializers.initializers_v2 import GlorotNormal



class MultiHeadAttention(Model):

    def __init__(self, num_heads, attention_size, dropout_rate=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.attention_size = attention_size
        self.dropout_rate = dropout_rate
        self.dense_Q = Dense(self.attention_size, kernel_initializer=GlorotNormal())
        self.dense_K = Dense(self.attention_size, kernel_initializer=GlorotNormal())
        self.dense_V = Dense(self.attention_size, kernel_initializer=GlorotNormal())
        self.dense_output = Dense(self.attention_size, activation='relu', kernel_initializer=GlorotNormal())
        self.layer_norm = LayerNormalization()
    

    def call(self, inputs):
        queries = inputs[0]
        keys = inputs[1]
        values = inputs[2]

        Q = self.dense_Q(queries) # (N, T_q, C)
        K = self.dense_K(keys) # (N, T_k, C)
        V = self.dense_V(values) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs /= K_.get_shape().as_list()[-1] ** 0.5

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        alphas = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        alphas *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        alphas = Dropout(self.dropout_rate)(alphas)

        # Weighted sum
        outputs = tf.matmul(alphas, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Linear
        outputs = self.dense_output(outputs)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = self.layer_norm(outputs)  # (N, T_q, C)
        return outputs

