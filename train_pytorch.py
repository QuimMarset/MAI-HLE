from utils.constants_paths import *
from utils.file_io_utils import *
from utils.path_utils import create_new_experiment_folder, create_folder
from utils.train_pytorch_utils import *
from utils.label_dicts import label2class

from trainers.trainer_cnn import TrainerCNN
from trainers.trainer_attention_bi_lstm import TrainerAttentionBiLSTM
from trainers.trainer_entity_attention import TrainerEntityAttention
from trainers.trainer_RBERT import TrainerRBERT
from trainers.trainer_attention_lstm_BERT import TrainerAttentionLSTMBERT
from trainers.trainer_entity_attention_bert import TrainerEntityAttentionBERT

from misc.official_scorer import OfficialF1Scorer
from misc.logger import Logger



def get_trainer(method, num_classes):
    if method == 'CNN':
        return TrainerCNN(num_classes)
    elif method == 'Attention_LSTM':
        return TrainerAttentionBiLSTM(num_classes)
    elif method == 'Entity_Attention':
        return TrainerEntityAttention(num_classes)
    elif method == 'RBERT':
        return TrainerRBERT(num_classes)
    elif method == 'Attention_LSTM_BERT':
        return TrainerAttentionLSTMBERT(num_classes)
    elif method == 'Entity_Attention_BERT':
        return TrainerEntityAttentionBERT(num_classes)


def train(method, train_x, train_y, test_x, test_y, experiment_path):
    logger = Logger(experiment_path)
    scorer = OfficialF1Scorer(true_test_predictions_path, perl_scorer_path, experiment_path)
    num_classes = len(label2class)
    
    trainer = get_trainer(method, num_classes)
    trainer.run(experiment_path, logger, scorer, train_x, train_y, test_x, test_y)
    
    logger.close()


if __name__ == '__main__':

    method = 'Entity_Attention_BERT'
    
    create_folder(experiments_path)
    experiment_path = create_new_experiment_folder(experiments_path)

    if method == 'CNN':
        train_path = train_data_cnn_path
        test_path = test_data_cnn_path
    else:
        train_path = train_data_path
        test_path = test_data_path

    train_x = load_json_to_dict(train_path)
    train_y = load_npy_file_to_np_array(train_labels_path)
    
    test_x = load_json_to_dict(test_path)
    test_y = load_npy_file_to_np_array(test_labels_path)

    train(method, train_x, train_y, test_x, test_y, experiment_path)
