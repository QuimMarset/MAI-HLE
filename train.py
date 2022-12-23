from utils.constants_paths import *
from utils.file_io_utils import *
from utils.path_utils import create_new_experiment_folder
from utils.builders import *
from utils.plot_utils import plot_learning_curves
from utils.data_generator import DataGenerator, TrainDataGenerator
from models.basic_model import BasicModel



def labels_to_one_hot(labels, class_to_index):
    num_labels = len(labels)
    numeric_labels = [class_to_index[label] for label in labels]
    one_hot_labels = np.zeros((num_labels, len(class_to_index)))
    one_hot_labels[range(num_labels), numeric_labels] = 1
    return one_hot_labels


def train(method, config, train_gen, test_gen, experiment_path):
    optimizer = create_optimizer(config.optimizer, config.learning_rate)

    model = create_model(method, config)
    model.compile(optimizer, config.label_smoothing)
    history = model.fit(train_gen, config.epochs, test_gen, config.patience)

    model.save_model(experiment_path)
    write_dict_to_json(history.history, join_path(experiment_path, 'train_metrics.json'))
    write_dict_to_json(vars(config), join_path(experiment_path, 'hyperparameters.json'))
    plot_learning_curves(history.history, experiment_path)
    return model


def save_predictions(predictions, experiment_path, index_to_relation):
    file_path = join_path(experiment_path, 'test_predictions.txt')
    start_index = 8001
    with open(file_path, 'w') as file:
        for i, prediction in enumerate(predictions):
            file.write(f'{start_index+i}\t{index_to_relation[prediction]}\n')


def test(experiment_path, test_gen, class_to_index, model):
    #model = BasicModel.create_test_model(experiment_path)
    predictions = model.model.predict(test_gen)
    predictions = np.argmax(predictions, axis=1)
    index_to_class = dict(zip(range(len(class_to_index)), list(class_to_index.keys())))
    save_predictions(predictions, experiment_path, index_to_class)



if __name__ == '__main__':

    method = 'attention_bi_lstm'
    config = load_configuration(method)
    experiment_path = create_new_experiment_folder(experiments_path)

    train_data = load_json_to_dict(train_data_path)
    test_data = load_json_to_dict(test_data_path)
    train_labels = load_npy_file_to_np_array(train_labels_path)
    test_labels = load_npy_file_to_np_array(test_labels_path)
    class_to_index = load_json_to_dict(class_to_index_path)

    train_labels = labels_to_one_hot(train_labels, class_to_index)
    test_labels = labels_to_one_hot(test_labels, class_to_index)

    feature_extractor = create_feature_extractor(method, config)
    train_features, train_labels = feature_extractor.compute_features(train_data, train_labels)
    test_features, test_labels = feature_extractor.compute_features(test_data, test_labels)

    train_gen = TrainDataGenerator(config.batch_size, train_features, train_labels, config.seed)
    test_gen = DataGenerator(config.batch_size, test_features, test_labels)

    model = train(method, config, train_gen, test_gen, experiment_path)
    test(experiment_path, test_gen, class_to_index, model)