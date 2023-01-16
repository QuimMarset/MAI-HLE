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



def check_predictions_rbert_fails_but_not_entity_aware_bert(ground_truths, rbert_predictions, entity_bert_preds, test_sentences, save_path):
    failures = {}

    for i, (ground_truth, rbert_prediction, entity_bert_pred) in enumerate(zip(ground_truths, rbert_predictions, entity_bert_preds)):

        if entity_bert_pred == ground_truth and rbert_prediction != ground_truth:
            failures[i] = {
                'sentence' : test_sentences[i],
                'rbert_pred' : label2class[rbert_prediction],
                'ground_truth' : label2class[ground_truth]
            }
            
    path_ = join_path(save_path, f'comparison_rbert_entity_bert.json')
    write_dict_to_json(failures, path_)


def check_predictions_rbert_fails_but_not_att_bert(ground_truths, rbert_predictions, att_bert_preds, test_sentences, save_path):
    failures = {}

    for i, (ground_truth, rbert_prediction, att_bert_pred) in enumerate(zip(ground_truths, rbert_predictions, att_bert_preds)):

        if att_bert_pred == ground_truth and rbert_prediction != ground_truth:
            failures[i] = {
                'sentence' : test_sentences[i],
                'rbert_pred' : label2class[rbert_prediction],
                'ground_truth' : label2class[ground_truth]
            }
            
    path_ = join_path(save_path, f'comparison_rbert_att_bert.json')
    write_dict_to_json(failures, path_)



if __name__ == '__main__':

    create_folder(evaluation_path)

    ground_truths = load_predictions(true_test_predictions_path)
    rbert_predictions = load_predictions(predictions_rbert)
    att_bert_preds = load_predictions(predictions_att_lstm_bert)

    test_sentences = load_test_sentences(raw_test_data_path)

    check_predictions_rbert_fails_but_not_att_bert(ground_truths, rbert_predictions, att_bert_preds, 
        test_sentences, evaluation_path)