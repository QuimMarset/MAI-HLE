import sys
sys.path.insert(0, './')
from utils.path_utils import create_folder
from utils.constants_paths import *
from utils.label_dicts import *
from utils.file_io_utils import read_data_file, write_dict_to_json


def load_predictions(predictions_path):
    predictions = []
    content = read_data_file(predictions_path)
    for line in content:
        class_name = line.split('\t')[1]
        label = class2label[class_name]
        predictions.append(label)
    return predictions


def load_test_sentences(test_data_path):
    file_lines = read_data_file(test_data_path)
    test_sentences = []
    num_lines = len(file_lines) - 1
    for i in range(0, num_lines, 4):
        _, sentence = file_lines[i].strip().split('\t')
        sentence = sentence[1:-1]
        test_sentences.append(sentence)
    return test_sentences



def check_wrong_other_predictions(ground_truths, predictions, test_sentences, save_path, method_name):
    false_positives = {}
    false_negatives = {}

    for i, (ground_truth, prediction) in enumerate(zip(ground_truths, predictions)):

        if ground_truth == 0 and ground_truth != prediction:
            false_negatives[i] = {
                'sentence': test_sentences[i],
                'prediction' : label2class[prediction]
            }
        
        if prediction == 0 and ground_truth != prediction:
            false_positives[i] = {
                'sentence' : test_sentences[i],
                'ground_truth' : label2class[ground_truth]
            }

    method_name = method_name.replace(' ', '_')
    path_false_positives = join_path(save_path, f'other_false_positives_{method_name}.json')
    path_false_negatives = join_path(save_path, f'other_false_negatives_{method_name}.json')
    write_dict_to_json(false_positives, path_false_positives)
    write_dict_to_json(false_negatives, path_false_negatives)




if __name__ == '__main__':

    create_folder(evaluation_path)

    predictions_paths = [predictions_cnn, predictions_att_lstm, predictions_entity_att, 
        predictions_rbert, predictions_att_lstm_bert, predictions_entity_att_bert]

    names = ['CNN', 'Attention Bi-LSTM', 'Entity-Aware Attention', 'R-BERT', 'Attention Bi-LSTM with BERT', 
        'Entity-Aware Attention with BERT']

    ground_truths = load_predictions(true_test_predictions_path)

    test_sentences = load_test_sentences(raw_test_data_path)

    for predictions_path, name in zip(predictions_paths, names):
        predictions = load_predictions(predictions_path)

        check_wrong_other_predictions(ground_truths, predictions, test_sentences, evaluation_path, name)