import tensorflow as tf
from tensorflow import keras
import numpy as np
from contextlib import redirect_stdout
from utils.file_io_utils import load_json_to_dict, load_json_to_string, write_json_string
from utils.path_utils import join_path
from utils.label_dicts import label2class



class BasicModel:

    def __init__(self, num_classes, config, optimimzer, scorer, logger):
        self.num_classes = num_classes
        self.l2_coef = config.l2_coef
        self.seed = config.seed
        self.config = config
        self.optimizer = optimimzer
        self.scorer = scorer
        self.logger = logger
        self.model = None
        self.set_rng()

    
    def set_rng(self):
        tf.random.set_seed(self.seed)
        keras.utils.set_random_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.rng_labels = np.random.default_rng(self.seed)


    @classmethod
    def create_test_model(cls, model_path):
        instance = super().__new__(cls)
        #instance.__load_architecture(model_path)
        instance.load_weights(model_path)
        return instance


    def save_model(self, save_path):
        self.save_summary(save_path)
        #self.save_architecture(save_path)
        self.save_weights(save_path)


    def save_summary(self, save_path):
        file_path = join_path(save_path, 'model_summary.txt')
        with open(file_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary(expand_nested=True)


    def load_architecture(self, load_path):
        architecture = load_json_to_string(join_path(load_path, 'model_architecture.json'))
        self.model =  keras.models.model_from_json(architecture)
        
    
    def save_architecture(self, save_path):
        model_json_string = self.model.to_json()
        write_json_string(model_json_string, join_path(save_path, 'model_architecture.json'))


    def load_weights(self, load_path):
        self.model.load_weights(join_path(load_path, 'model_weights')).expect_partial()


    def save_weights(self, save_path):
        self.model.save_weights(join_path(save_path, 'model_weights'))


    def compute_num_batches(self, train_x, batch_size):
        num_batches = train_x.shape[0] // batch_size
        modulo = train_x.shape[0] % batch_size
        if modulo > 0:
            num_batches += 1
        return num_batches


    def compute_l2_loss(self):
        l2_loss = self.l2_coef * tf.add_n(
            [
                tf.nn.l2_loss(weight) 
                for weight in self.model.trainable_variables 
            ]
        )
        return l2_loss


    def compute_loss(self, logits, batch_y):
        loss = tf.nn.softmax_cross_entropy_with_logits(batch_y, logits)
        loss = tf.reduce_mean(loss)
        l2_loss = self.compute_l2_loss()
        return loss + l2_loss


    def compute_predictions(self, logits):
        return tf.argmax(logits, axis=-1)

    
    def compute_accuracy(self, logits, batch_y):
        predictions = self.compute_predictions(logits)
        correct_predictions = tf.equal(predictions, tf.argmax(batch_y, axis=-1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy.numpy()


    def update_weights(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            logits = self.model(batch_x)
            loss = self.compute_loss(logits, batch_y)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_value(grad, -self.config.gradient_clip, self.config.gradient_clip) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        accuracy = self.compute_accuracy(logits, batch_y)
        return loss.numpy(), accuracy

    
    def evaluate(self, test_x, test_y, batch_size):
        num_batches = self.compute_num_batches(test_x, batch_size)
        predictions = []
        loss = 0
        accuracy = 0

        for index in range(num_batches):
            start = index * batch_size
            end = start + batch_size

            batch_x = test_x[start : end]
            batch_y = test_y[start : end]

            logits = self.model(batch_x, training=False)
            predictions_i = self.compute_predictions(logits).numpy().tolist()
            predictions.extend(predictions_i)

            loss += self.compute_loss(logits, batch_y).numpy()
            accuracy += self.compute_accuracy(logits, batch_y)
        
        loss /= num_batches
        accuracy /= num_batches
        return loss, accuracy, predictions

    
    def save_predictions(self, predictions, save_path):
        self.predictions_path = join_path(save_path, 'best_test_predictions.txt')
        start_index = 8001
        with open(self.predictions_path, 'w') as file:
            for i, prediction in enumerate(predictions):
                file.write(f'{start_index+i}\t{label2class[prediction]}\n')


    def shuffle_train_data(self, train_x, train_y):
        self.rng.shuffle(train_x)
        self.rng_labels.shuffle(train_y)


    def train(self, train_x, train_y, test_x, test_y, epochs, batch_size, save_path):
        num_batches = self.compute_num_batches(train_x, batch_size)
        best_macro_f1 = 0

        for epoch in range(epochs):

            self.shuffle_train_data(train_x, train_y)
            train_loss = 0
            train_accuracy = 0

            for index in range(num_batches):
                start = index * batch_size
                end = start + batch_size

                batch_x = train_x[start : end]
                batch_y = train_y[start : end]

                train_loss_i, train_accuracy_i = self.update_weights(batch_x, batch_y)
                train_loss += train_loss_i
                train_accuracy += train_accuracy_i

            train_loss /= num_batches
            train_accuracy /= num_batches

            self.logger.logging_train(epoch, train_loss, train_accuracy)

            test_loss, test_accuracy, predictions = self.evaluate(test_x, test_y, batch_size)
            test_macro_f1 = self.scorer.compute_f1_score(predictions)

            self.logger.logging_test(epoch, test_loss, test_accuracy, test_macro_f1)

            if test_macro_f1 > best_macro_f1:
                best_macro_f1 = test_macro_f1
                self.save_weights(save_path)
                self.save_predictions(predictions, save_path)