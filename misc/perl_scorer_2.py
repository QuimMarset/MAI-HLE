import numpy as np
import subprocess
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


    def __compute_f1_score(self):
        process = subprocess.Popen(["perl", self.scorer_path, self.predictions_path, self.true_predictions_path], 
            stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE, executable=None)
            
        str_parse = str(process.communicate()[0]).split("\\n")[-2]
        idx = str_parse.find('%')
        f1_score = float(str_parse[idx-5:idx])
        return f1_score


    def compute_f1_score(self, predictions):
        self.save_predictions(predictions)
        return self.__compute_f1_score()


        
        