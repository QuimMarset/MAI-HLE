from utils.constants_paths import *
from utils.file_io_utils import *
from utils.path_utils import create_new_experiment_folder
from utils.train_pytorch_utils import *
from utils.label_dicts import label2class
from trainers.trainer_RBERT import TrainerRBERT
from misc.perl_scorer import OfficialF1Scorer
from misc.logger import Logger




def train(method, train_x, train_y, test_x, test_y, experiment_path):
    logger = Logger(experiment_path)
    scorer = OfficialF1Scorer(true_test_predictions_path, perl_scorer_path, experiment_path)
    
    trainer = TrainerRBERT(len(label2class))
    trainer.run(experiment_path, logger, scorer, train_x, train_y, test_x, test_y)
    
    logger.close()




if __name__ == '__main__':

    method = 'RBERT'
    
    experiment_path = create_new_experiment_folder(experiments_path)

    train_x = load_json_to_dict(train_data_path)
    train_y = load_npy_file_to_np_array(train_labels_path)
    
    test_x = load_json_to_dict(test_data_path)
    test_y = load_npy_file_to_np_array(test_labels_path)

    train(method, train_x, train_y, test_x, test_y, experiment_path)
