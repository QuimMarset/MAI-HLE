from utils.constants_paths import *
from utils.file_io_utils import *
from utils.path_utils import create_new_experiment_folder, create_folder
from utils.train_pytorch_utils import *
from utils.label_dicts import label2class
from trainers.trainer_entity_attention import TrainerEntityAttention
from misc.perl_scorer import OfficialF1Scorer
from misc.logger import Logger




def train(method, train_x, train_y, test_x, test_y, experiment_path):
    logger = Logger(experiment_path)
    scorer = OfficialF1Scorer(true_test_predictions_path, perl_scorer_path, experiment_path)
    num_classes = len(label2class)
    
    trainer = TrainerEntityAttention(num_classes, glove_300_path)
    trainer.run(experiment_path, logger, scorer, train_x, train_y, test_x, test_y)
    
    logger.close()



def get_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = json.loads(line.strip())
            data.append(line)
    return data


if __name__ == '__main__':

    method = 'RBERT'
    
    create_folder(experiments_path)
    experiment_path = create_new_experiment_folder(experiments_path)

    train_x = load_json_to_dict(train_data_path)
    #train_x = get_data('data/train.json')
    train_y = load_npy_file_to_np_array(train_labels_path)
    
    test_x = load_json_to_dict(test_data_path)
    #test_x = get_data('data/test.json')
    test_y = load_npy_file_to_np_array(test_labels_path)

    train(method, train_x, train_y, test_x, test_y, experiment_path)
