import numpy as np
from tensorflow import keras



class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, data, labels):
        self.batch_size = batch_size
        self.data = data
        self.labels = labels

        self.num_batches = int(len(data) / self.batch_size)
        if self.num_batches * batch_size < len(data):
            self.num_batches += 1

    def __len__(self):
        return self.num_batches


    def __getitem__(self, index):
        batch_index = index * self.batch_size
        return self.load_batch(batch_index)


    def load_batch(self, batch_index):
        batch_data = self.data[batch_index : batch_index + self.batch_size]
        batch_labels = self.labels[batch_index : batch_index + self.batch_size]
        return batch_data, batch_labels


class TrainDataGenerator(DataGenerator):

    def __init__(self, batch_size, data, labels, seed):
        super().__init__(batch_size, data, labels)

        self.rng = np.random.default_rng(seed)
        self.labels_rng = np.random.default_rng(seed)
        self.on_epoch_end()

    
    def on_epoch_end(self):
        self.rng.shuffle(self.data)
        self.labels_rng.shuffle(self.labels)