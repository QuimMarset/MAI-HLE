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

    
    def semeval_scorer(self, predict_label, true_label, class_num=10):
        assert true_label.shape[0] == predict_label.shape[0]
        confusion_matrix = np.zeros(shape=[class_num, class_num], dtype=np.float32)
        xDIRx = np.zeros(shape=[class_num], dtype=np.float32)
        for i in range(true_label.shape[0]):
            true_idx = math.ceil(true_label[i]/2)
            predict_idx = math.ceil(predict_label[i]/2)
            if true_label[i] == predict_label[i]:
                confusion_matrix[predict_idx][true_idx] += 1
            else:
                if true_idx == predict_idx:
                    xDIRx[predict_idx] += 1
                else:
                    confusion_matrix[predict_idx][true_idx] += 1

        col_sum = np.sum(confusion_matrix, axis=0).reshape(-1)
        row_sum = np.sum(confusion_matrix, axis=1).reshape(-1)
        f1 = np.zeros(shape=[class_num], dtype=np.float32)

        for i in range(0, class_num):  # ignore the 'Other'
            try:
                p = float(confusion_matrix[i][i]) / float(col_sum[i] + xDIRx[i])
                r = float(confusion_matrix[i][i]) / float(row_sum[i] + xDIRx[i])
                f1[i] = (2 * p * r / (p + r))
            except:
                pass
        actual_class = 0
        total_f1 = 0.0
        for i in range(1, class_num):
            if f1[i] > 0.0:  # classes that not in the predict label are not considered
                actual_class += 1
                total_f1 += f1[i]
        try:
            macro_f1 = total_f1 / actual_class
        except:
            macro_f1 = 0.0
        return macro_f1


    def __compute_f1_score(self):
        process = subprocess.Popen(["perl", self.scorer_path, self.predictions_path, 'TEST_FILE_KEY.txt'], 
            stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE, executable=None)

        print(process.communicate())
            
        str_parse = str(process.communicate()[0]).split("\\n")[-2]
        idx = str_parse.find('%')
        f1_score = float(str_parse[idx-5:idx])
        return f1_score


    def compute_f1_score(self, predictions, true_labels):
        return self.semeval_scorer(np.array(predictions), np.array(true_labels))


        
        