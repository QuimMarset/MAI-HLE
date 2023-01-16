import numpy as np
import sys
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn
sys.path.insert(0, './')
from utils.path_utils import create_folder
from utils.constants_paths import *
from utils.plot_utils import *
from utils.label_dicts import *
from utils.file_io_utils import read_data_file


def load_predictions(predictions_path):
    predictions = []
    content = read_data_file(predictions_path)
    for line in content:
        class_name = line.split('\t')[1]
        label = class2label[class_name]
        predictions.append(label)
    return predictions


def build_confusion_matrix_using_directionality(ground_truths, predictions, save_path, method_name):
    class_names = list(class2label.keys())

    # Rows means true, columns means predictions
    confusion_matrix = confusion_matrix_sklearn(ground_truths, predictions)
    plot_confusion_matrix(confusion_matrix, class_names, save_path, method_name)


def build_confusion_matrix_considering_but_not_using_directionality(ground_truths, predictions, save_path, method_name):
    # Unlike the previous, we take directionality into account, but we plot only 9+1 classes
    # like the official score

    num_classes = len(class_names_no_direction)
    # Rows means true, columns means predictions
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for ground_truth, prediction in zip(ground_truths, predictions):
        
        # Index without the direction
        ground_base = math.ceil(ground_truth / 2)
        predicted_base = math.ceil(prediction / 2)

        # Match the base class and the order of the relation
        if ground_truth == prediction:
            confusion_matrix[ground_base][predicted_base] += 1

        elif ground_base != predicted_base:
            # Neither the base class nor the order match
            # Here we ignore the cases where the order does not match
            confusion_matrix[ground_base][predicted_base] += 1

    plot_confusion_matrix(confusion_matrix, class_names_no_direction, save_path, method_name, 
        with_direction=False)



if __name__ == '__main__':

    create_folder(evaluation_path)

    predictions_paths = [predictions_cnn, predictions_att_lstm, predictions_entity_att, 
        predictions_rbert, predictions_att_lstm_bert, predictions_entity_att_bert]

    names = ['CNN', 'Attention Bi-LSTM', 'Entity-Aware Attention', 'R-BERT', 'Attention Bi-LSTM with BERT', 
        'Entity-Aware Attention with BERT']

    ground_truths = load_predictions(true_test_predictions_path)

    for predictions_path, name in zip(predictions_paths, names):
        predictions = load_predictions(predictions_path)

        build_confusion_matrix_using_directionality(ground_truths, predictions, evaluation_path, name)
        build_confusion_matrix_considering_but_not_using_directionality(ground_truths, predictions, evaluation_path, name)
