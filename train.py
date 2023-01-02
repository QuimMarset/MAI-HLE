from utils.constants_paths import *
from utils.file_io_utils import *
from utils.path_utils import create_new_experiment_folder
from utils.builders import *
from utils.label_dicts import class2label
from misc.perl_scorer_2 import OfficialF1Scorer
from misc.logger import Logger



def labels_to_one_hot(labels, class_to_index):
    num_labels = len(labels)
    numeric_labels = [class_to_index[label] for label in labels]
    one_hot_labels = np.zeros((num_labels, len(class_to_index)))
    one_hot_labels[range(num_labels), numeric_labels] = 1
    return one_hot_labels


def numeric_labels_to_one_hot(numeric_labels):
    num_labels = len(numeric_labels)
    one_hot_labels = np.zeros((num_labels, len(class2label)))
    one_hot_labels[range(num_labels), numeric_labels] = 1
    return one_hot_labels


def smooth_labels(labels, label_smoothing):
    labels *= (1 - label_smoothing)
    labels += (label_smoothing / labels.shape[1])
    return labels


def train(method, config, train_x, train_y, test_x, test_y, input_shape, experiment_path):
    logger = Logger(experiment_path)
    logger.print_hyperparameters(config)

    optimizer = create_optimizer(config.optimizer, config.learning_rate, config.gradient_clip, config.lr_decay)
    scorer = OfficialF1Scorer(true_test_predictions_path, perl_scorer_path, experiment_path)
    model = create_model(method, config, input_shape, optimizer, scorer, logger)

    model.train(train_x, train_y, test_x, test_y, config.epochs, config.batch_size, experiment_path)

    #write_dict_to_json(history.history, join_path(experiment_path, 'train_metrics.json'))
    write_dict_to_json(vars(config), join_path(experiment_path, 'hyperparameters.json'))
    logger.close()




if __name__ == '__main__':

    method = 'entity_attention_lstm'
    config = load_configuration(method)
    experiment_path = create_new_experiment_folder(experiments_path)

    train_data = load_json_to_dict(train_data_2_path)
    test_data = load_json_to_dict(test_data_2_path)
    train_labels = load_npy_file_to_np_array(train_labels_2_path)
    test_labels = load_npy_file_to_np_array(test_labels_2_path)

    train_labels = numeric_labels_to_one_hot(train_labels)
    test_labels = numeric_labels_to_one_hot(test_labels)

    if config.label_smoothing > 0:
        train_labels = smooth_labels(train_labels, config.label_smoothing)

    print('Computing Features...')

    feature_extractor = create_feature_extractor(method, config)
    train_features, train_labels = feature_extractor.compute_features(train_data, train_labels)
    test_features, test_labels = feature_extractor.compute_features(test_data, test_labels)

    input_shape = train_features.shape[1:]

    print('Start Training...')

    train(method, config, train_features, train_labels, test_features, test_labels, input_shape, experiment_path)