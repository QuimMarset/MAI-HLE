import numpy as np
import json
import yaml
from utils.path_utils import join_path
from types import SimpleNamespace



def read_data_file(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        return file.readlines()


def save_array_to_npy_file(array, file_path):
    with open(file_path, 'wb') as file:
        np.save(file, array)


def load_npy_file_to_np_array(file_path):
    with open(file_path, 'rb') as file:
        return np.load(file)


def write_json_string(json_string, file_path):
    with open(file_path, 'w', encoding='utf8') as file:
        json.dump(json.loads(json_string), file, indent=4)


def write_dict_to_json(dict, file_path, indent=4):
    with open(file_path, 'w', encoding='utf8') as file:
        json.dump(dict, file, indent=indent)


def load_json_to_dict(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        dict = json.load(file)
    return dict


def load_json_to_string(file_path):
    dict = load_json_to_dict(file_path)
    return json.dumps(dict)


def read_yaml_config(file_path):
    with open(file_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return SimpleNamespace(**config)