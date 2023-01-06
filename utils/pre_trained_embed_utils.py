from utils.constants_paths import (glove_50_path, glove_100_path, glove_200_path, glove_300_path,
    glove_300_2_path, tacred_50_path)



def get_pre_trained_path(name):

    if name == 'glove_50':
        return glove_50_path

    elif name == 'glove_100':
        return glove_100_path

    elif name == 'glove_200':
        return glove_200_path

    elif name == 'glove_300':
        return glove_300_path

    elif name == 'glove_300_2':
        return glove_300_2_path

    elif name == 'tacred_50':
        return tacred_50_path
