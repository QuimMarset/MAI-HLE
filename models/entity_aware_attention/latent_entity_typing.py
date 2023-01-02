import tensorflow as tf
from keras.layers import Layer
from keras.initializers.initializers_v2 import GlorotNormal



class LatentEntityType(Layer):
    def __init__(self, num_types, latent_size, **kwargs):
        super().__init__(**kwargs)
        self.num_types = num_types
        self.latent_size = latent_size

    
    def build(self, input_shape):
        self.latent_types = self.add_weight(name='latent_types', 
            shape=(self.num_types, self.latent_size), initializer=GlorotNormal(), 
            trainable=True)

    
    def call(self, entity_hidden):
        # BxL * LxK -> BxK
        logits = tf.matmul(entity_hidden, tf.transpose(self.latent_types))
        # BxK
        alphas = tf.nn.softmax(logits)
        # BxK * KxL -> BxL
        entity_types = tf.matmul(alphas, self.latent_types)
        return entity_types