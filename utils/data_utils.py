import numpy as np
from sklearn.model_selection import train_test_split
from features_extractors.cnn_features import CNNFeatureExtractor



def labels_to_one_hot(labels, class_to_index):
    num_classes = len(class_to_index)
    num_labels = len(labels)
    numeric_labels = [class_to_index[label] for label in labels]
    one_hot_labels = np.zeros((num_labels, num_classes))
    one_hot_labels[range(num_labels), numeric_labels] = 1
    return one_hot_labels


def create_train_val_split(train_features, train_labels, percentage, seed):
    num_samples = train_labels.shape[0]
    num_val = int(num_samples * percentage)
    output = train_test_split(train_features, train_labels, test_size=num_val, shuffle=True, random_state=seed)
    train_features, val_features, train_labels, val_labels = output
    return np.array(train_features), np.array(val_features), np.array(train_labels), np.array(val_labels)


def get_features(set_name, method_name, data, word_to_index, max_sentence_length, features_path, max_distance):
    if method_name == 'CNN':
        feature_extractor = CNNFeatureExtractor(set_name, data, word_to_index, max_sentence_length, features_path, max_distance)
    else:
        raise NotImplementedError(method_name)

    features = feature_extractor.get_features()
    return features


def get_train_features(method_name, train_data, word_to_index, max_sentence_length, features_path, max_distance):
    return get_features('train', method_name, train_data, word_to_index, max_sentence_length, features_path, max_distance)


def get_test_features(method_name, test_data, word_to_index, max_sentence_length, features_path, max_distance):
    test_features = get_features('test', method_name, test_data, word_to_index, max_sentence_length, features_path, max_distance)
    return np.array(test_features)