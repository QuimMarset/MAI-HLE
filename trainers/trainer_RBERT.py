from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from models.r_bert import RBERT
from features_extractors.r_bert_features import RBERTFeatureExtractor
from utils.constants_paths import rbert_config_path, words_vocabulary_path
from utils.file_io_utils import read_yaml_config, load_npy_file_to_np_array
from utils.train_pytorch_utils import train
from utils.dataset import create_train_data_loader, create_test_data_loader




class TrainerRBERT:


    def __init__(self, num_classes):
        self.__init_config()
        self.__create_model(num_classes)
        self.__create_optimizer()
        self.__create_feature_extractor()


    def __init_config(self):
        self.config = read_yaml_config(rbert_config_path)


    def __create_model(self, num_classes):
        self.model = RBERT(num_classes, self.config)


    def __create_optimizer(self):
        bert_params = list(map(id, self.model.bert.parameters()))
        rest_params = filter(lambda p: id(p) not in bert_params, self.model.parameters())

        grouped_parameters = [
            {'params': self.model.bert.parameters()},
            {'params': rest_params,  'lr': self.config.learning_rate_BERT},
        ]

        learning_rate = self.config.learning_rate
        self.optimizer = AdamW(grouped_parameters, learning_rate, eps=1e-8)


    def __create_lr_schedule(self):
        gradient_accum_steps = self.config.gradient_accumulation_steps
        epoch_training_steps = len(self.train_loader) // gradient_accum_steps
        num_training_steps = self.config.epochs * epoch_training_steps

        num_warmup_steps = int(num_training_steps * self.config.warmup_proportion)
        
        self.schedule = get_linear_schedule_with_warmup(self.optimizer, 
            num_warmup_steps, num_training_steps)


    def __create_feature_extractor(self):
        words_vocabulary = load_npy_file_to_np_array(words_vocabulary_path)
        self.feature_extractor = RBERTFeatureExtractor(self.config.max_length, words_vocabulary)

    
    def __create_loaders(self, train_x, train_y, test_x, test_y):
        train_x, train_y = self.feature_extractor.compute_features(train_x, train_y)
        test_x, test_y = self.feature_extractor.compute_features(test_x, test_y)

        self.train_loader = create_train_data_loader(train_x, train_y, self.config.batch_size)
        self.test_loader = create_test_data_loader(test_x, test_y, self.config.batch_size)


    def run(self, save_path, logger, scorer, train_x, train_y, test_x, test_y):
        logger.print_hyperparameters(self.config)
        self.__create_loaders(train_x, train_y, test_x, test_y)

        self.__create_lr_schedule()

        train(self.config, self.model, self.optimizer, logger, 
            scorer, self.train_loader, self.test_loader, 
            save_path, self.schedule)