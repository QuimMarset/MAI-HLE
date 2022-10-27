import os



def join_path(base_path, *path_names):
    return os.path.join(base_path, *path_names)


def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def get_number_subfolders(path):
    return sum(os.path.isdir(join_path(path, elem)) for elem in os.listdir(path))


def create_new_experiment_folder(path):
    num_experiments = get_number_subfolders(path)
    experiment_path = join_path(path, f'experiment_{num_experiments+1}')
    create_folder(experiment_path)
    return experiment_path


def exists_path(path):
    return os.path.exists(path)