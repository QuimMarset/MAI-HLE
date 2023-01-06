import numpy as np
import subprocess
import math
from utils.label_dicts import label2class
from utils.path_utils import join_path



class OfficialF1Scorer:

    def __init__(self, true_predictions_path, scorer_path, experiment_path):
        super().__init__()
        self.true_predictions_path = true_predictions_path
        self.scorer_path = scorer_path
        self.experiment_path = experiment_path

    
    def save_predictions(self, predictions):
        self.predictions_path = join_path(self.experiment_path, 'temp_test_predictions.txt')
        start_index = 8001
        with open(self.predictions_path, 'w') as file:
            for i, prediction in enumerate(predictions):
                file.write(f'{start_index+i}\t{label2class[prediction]}\n')


    def __compute_confusion_matrix(self, predicted_labels, true_labels, num_classes):
        confusion_matrix = np.zeros(shape=[num_classes, num_classes], dtype=np.float32)
        # Number of matches without considering directionality
        match_no_direction = np.zeros(num_classes)

        for predicted_label, true_label in zip(predicted_labels, true_labels):
            true_index = math.ceil(true_label / 2)
            predicted_index = math.ceil(predicted_label / 2)

            if true_label == predicted_label:
                confusion_matrix[predicted_index][true_index] += 1
            else:
                if true_index == predicted_index:
                    match_no_direction[predicted_index] += 1
                else:
                    confusion_matrix[predicted_index][true_index] += 1

        return confusion_matrix, match_no_direction


    def __compute_f1_scores(self, confusion_matrix, match_no_direction):
        num_classes = match_no_direction.shape[0]
        cols_sum = np.sum(confusion_matrix, axis=0).reshape(-1)
        rows_sum = np.sum(confusion_matrix, axis=1).reshape(-1)
        f1_scores = np.zeros(shape=[num_classes], dtype=np.float32)

        for i in range(0, num_classes):  # ignore the 'Other'
            try:
                precision = float(confusion_matrix[i][i]) / float(cols_sum[i] + match_no_direction[i])
                recall = float(confusion_matrix[i][i]) / float(rows_sum[i] + match_no_direction[i])
                f1_scores[i] = (2 * precision * recall / (precision + recall))
            except:
                pass

        return f1_scores


    def __compute_macro_average_f1(self, f1_scores):
        num_classes = f1_scores.shape[0]
        num_found_classes = 0
        total_f1 = 0.0
        
        for i in range(1, num_classes):
            # classes that not in the predicted labels are not considered
            if f1_scores[i] > 0.0:
                num_found_classes += 1
                total_f1 += f1_scores[i]
        
        if num_found_classes > 0:
            return total_f1 / num_found_classes
        return 0

    
    def __compute_macro_f1_manual(self, predicted_labels, true_labels, num_classes=10):
        assert true_labels.shape[0] == predicted_labels.shape[0]
        confusion_matrix, match_no_direction = \
            self.__compute_confusion_matrix(predicted_labels, true_labels, num_classes)
        f1_scores = self.__compute_f1_scores(confusion_matrix, match_no_direction)
        macro_f1 = self.__compute_macro_average_f1(f1_scores)
        return macro_f1


    def __compute_macro_f1_perl(self):
        process = subprocess.Popen(["perl", self.scorer_path, self.predictions_path, 'TEST_FILE_KEY.txt'], 
            stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE, executable=None)
        str_parse = str(process.communicate()[0]).split("\\n")[-2]
        idx = str_parse.find('%')
        f1_score = float(str_parse[idx-5:idx])
        return f1_score


    def compute_f1_score(self, predicted_labels, true_labels):
        return self.__compute_macro_f1_manual(np.array(predicted_labels), np.array(true_labels))


        
        