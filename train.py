from tensorflow import keras
from utils.path_utils import *
from utils.embedding_utils import *
from utils.file_io_utils import *
from utils.plot_utils import *
from utils.data_utils import *
from constants_paths import *
from models.cnn_model import CNNModel
from data_generator import DataGenerator



def create_model(input_shape, num_classes, word_embedding_matrix, config):
    if config.model == 'CNN':
        return CNNModel(input_shape, num_classes, word_embedding_matrix, config.pos_embedding_dim, config.max_distance, 
            config.max_length, config.window_size, config.conv_filters, config.dense_units, config.dropout, config.seed)
    else:
        raise NotImplementedError(config.model)


def train(input_shape, train_gen, val_gen, config, num_classes, word_embed_matrix, experiment_path):
    optimizer = keras.optimizers.Adam(config.learning_rate)
    
    model = create_model(input_shape, num_classes, word_embed_matrix, config)
    model.save_architecture(experiment_path)
    model.save_summary(experiment_path)

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, 
        restore_best_weights=True, min_delta=1e-5)
    
    loss_function = keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing)
    model.model.compile(optimizer, loss=loss_function, metrics=["accuracy"])
    history = model.model.fit(train_gen, epochs=config.epochs, validation_data=val_gen, workers=6, 
        shuffle=False, callbacks=[early_stopping])

    model.save_weights(experiment_path)
    
    # Save training metrics and experiment hyperparameters
    training_metrics = history.history
    write_dict_to_json(training_metrics, experiment_path, 'train_metrics')
    
    hyperparameters = vars(config)
    write_dict_to_json(hyperparameters, experiment_path, 'hyperparameters')

    plot_learning_curves(training_metrics['loss'], training_metrics['val_loss'], 
        training_metrics['accuracy'], training_metrics['val_accuracy'], experiment_path)



if __name__ == '__main__':

    config = read_yaml_config('config.yaml')

    create_folder(features_path)
    create_folder(experiments_path)
    experiment_path = create_new_experiment_folder(experiments_path)

    train_labels = load_npy_file_to_np_array(train_labels_path)
    class_to_index = load_json_to_dict(class_to_index_path)
    words_to_index = load_json_to_dict(word_to_index_path)
    train_data = load_json_to_dict(train_data_path)
    word_embedding_matrix = load_npy_file_to_np_array(embedding_matrix_path)

    num_classes = len(class_to_index)
    
    train_features = get_train_features(config.method, train_data, words_to_index, config.max_length, features_path)
    train_labels = labels_to_one_hot(train_labels, class_to_index)
    train_features, val_features, train_labels, val_labels = create_train_val_split(train_features, train_labels, 0.1, config.seed)

    train_gen = DataGenerator(config.batch_size, train_features, train_labels, config.seed)
    val_gen = DataGenerator(config.batch_size, val_features, val_labels, config.seed)

    input_shape = train_features.shape[1]

    train(input_shape, train_gen, val_gen, config, num_classes, word_embedding_matrix, experiment_path)