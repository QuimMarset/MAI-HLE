import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout
from utils.file_io_utils import load_json_to_dict, load_json_to_string, write_json_string
from utils.path_utils import join_path



class BasicModel:

    def __init__(self, num_classes, seed):
        super().__init__()
        self.num_classes = num_classes
        self.seed = seed
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(seed)


    @classmethod
    def create_test_model(cls, model_path):
        instance = super().__new__(cls)
        instance.load_architecture(model_path)
        instance.load_weights(model_path)
        return instance


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