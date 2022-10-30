from types import SimpleNamespace
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from constants_paths import *
from utils.data_utils import get_test_features, labels_to_one_hot
from utils.path_utils import *
from utils.plot_utils import plot_confusion_matrix
from utils.file_io_utils import load_json_to_dict, load_npy_file_to_np_array, read_data_file
from data_generator import DataGenerator
from models import cnn_model



def create_model(model_name, experiment_path):
    if model_name == 'CNN':
        return cnn_model.CNNModel.create_test_model(experiment_path)
    else:
        raise NotImplementedError(model_name)


def save_predictions(predictions, experiment_path, index_to_relation):
    file_path = join_path(experiment_path, 'test_predictions.txt')
    start_index = 8001
    with open(file_path, 'w') as file:
        for i, prediction in enumerate(predictions):
            file.write(f'{start_index+i}\t{index_to_relation[prediction]}\n')


def test(model_name, experiment_path, test_gen, index_to_relation):
    model = create_model(model_name, experiment_path)

    true_labels = np.argmax(test_gen.labels, axis=1)
    predictions = model.model.predict(test_gen)
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f'\nTest accuracy: {accuracy:.2f}\n')
    save_predictions(predictions, experiment_path, index_to_relation)



if __name__ == '__main__':
    
    experiment_path = join_path(experiments_path, 'experiment_2')

    config = load_json_to_dict(join_path(experiment_path, 'hyperparameters.json'))
    config = SimpleNamespace(**config)

    class_to_index = load_json_to_dict(class_to_index_path)
    index_to_class = dict(zip(range(len(class_to_index)), list(class_to_index.keys())))
    word_to_index = load_json_to_dict(word_to_index_path)
    
    test_data = load_json_to_dict(test_data_path)
    test_data = read_data_file(join_path('data', 'test.cln'))
    
    test_features = get_test_features(config.method, test_data, word_to_index, config.max_length, extracted_features_path, config.max_distance)
    
    test_labels = load_npy_file_to_np_array(test_labels_path)
    test_labels = labels_to_one_hot(test_labels, class_to_index)
    
    test_gen = DataGenerator(config.batch_size, test_features, test_labels)
    
    test(config.model, experiment_path, test_gen, index_to_class)
