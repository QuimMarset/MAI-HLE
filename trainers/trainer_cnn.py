from torch.optim import Adam
from models.cnn import CNN
from features_extractors.cnn_features import CNNFeatureExtractor
from utils.constants_paths import cnn_config_path
from utils.file_io_utils import read_yaml_config
from utils.train_pytorch_utils import train
from utils.pre_trained_embed_utils import get_pre_trained_path
from utils.dataset import create_train_data_loader, create_test_data_loader




class TrainerCNN:


    def __init__(self, num_classes):
        self.__init_config()
        self.__create_feature_extractor()
        self.__create_model(num_classes)
        self.__create_optimizer()


    def __init_config(self):
        self.config = read_yaml_config(cnn_config_path)


    def __create_feature_extractor(self):
        max_length = self.config.max_length
        word_dim = self.config.word_dim
        max_distance = self.config.max_distance
        pre_trained_path = get_pre_trained_path(self.config.pre_trained_path)
        self.feature_extractor = CNNFeatureExtractor(pre_trained_path, 
            word_dim, max_length, max_distance)


    def __create_model(self, num_classes):
        embedding_matrix = self.feature_extractor.embedding_matrix
        self.model = CNN(num_classes, self.config, embedding_matrix)

    
    def __create_optimizer(self):
        learning_rate = self.config.learning_rate
        l2_decay = self.config.l2_decay
        self.optimizer = Adam(self.model.parameters(), learning_rate, weight_decay=l2_decay)

    
    def __create_loaders(self, train_x, train_y, test_x, test_y):
        train_x, train_y = self.feature_extractor.compute_features(train_x, train_y)
        test_x, test_y = self.feature_extractor.compute_features(test_x, test_y)

        self.train_loader = create_train_data_loader(train_x, train_y, self.config.batch_size)
        self.test_loader = create_test_data_loader(test_x, test_y, self.config.batch_size)


    def run(self, save_path, logger, scorer, train_x, train_y, test_x, test_y):
        logger.print_hyperparameters(self.config)
        self.__create_loaders(train_x, train_y, test_x, test_y)

        train(self.config, self.model, self.optimizer, logger, 
            scorer, self.train_loader, self.test_loader, save_path)