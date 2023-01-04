import datetime
from utils.path_utils import join_path


class Logger:


    def __init__(self, save_path):
        self.open_log_file(save_path)
        self.best_f1 = 0.0


    def open_log_file(self, save_path):
        self.log_path = join_path(save_path, "logs.txt")
        self.log_file = open(self.log_path, "w")


    def print_hyperparameters(self, config):
        self.log_file.write("\n================ Hyper-parameters ================\n\n")
        
        for parameter in vars(config):
            value = getattr(config, parameter)
            self.log_file.write(f'{parameter}={value}\n')
        
        self.log_file.write("\n==================================================\n\n")


    def logging_train(self, epoch, loss, accuracy):
        time_str = datetime.datetime.now().isoformat()
        log = f'{time_str}: Epoch {epoch+1}, Train Loss {loss:.2f}, Train Accuracy {accuracy:.2f}'
        self.log_file.write(log + '\n')
        print(log)


    def logging_test(self, epoch, loss, accuracy, f1_score):
        time_str = datetime.datetime.now().isoformat()
        log = f'{time_str}: Epoch {epoch+1}, Test Loss {loss:.2f}, Test Accuracy {accuracy:.2f}'
        self.log_file.write(log + '\n')
        print(log)

        if f1_score > self.best_f1:
            self.best_f1 = f1_score

        f1_log = f'Official macro-averaged F1-score (9+1)-Way considering directionality: Current {f1_score:.4f} '
        f1_log += f'Best {self.best_f1:.4f}'
        self.log_file.write(f1_log + '\n')
        print(f1_log)



    def log_text(self, text):
        self.log_file.write(text + '\n')
        print(text)


    def close(self):
        self.log_file.flush()
        self.log_file.close()