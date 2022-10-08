import numpy as np
from utils.path_utils import join_path



def read_data_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


def save_array_to_npy_file(array, save_path, file_name):
    file_path = join_path(save_path, f'{file_name}.npy')
    with open(file_path, 'wb') as file:
        np.save(file, array)


def load_npy_file_to_np_array(file_path):
    with open(file_path, 'rb') as file:
        return np.load(file)