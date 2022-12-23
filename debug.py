# 1964, 1748, 2618, 4962, 5186, 7325, 7341, 7387, 8061, 9519
from utils.file_io_utils import *
from utils.path_utils import *
from utils.constants_paths import *
from utils.builders import *
from models.attention_bi_lstm import Attention, AttentionBiLSTM
import tensorflow as tf
from nltk.tokenize import word_tokenize



if __name__ == '__main__':

    config = read_yaml_config(attention_bi_lstm_config_path)

    word_embed_matrix = load_word_embed_matrix(config.pre_trained_name)

    num_classes = len(load_json_to_dict(class_to_index_path))

    sentence = 'Because <e1>sports broadcast</e1> reports <e2>on-going events</e2> within a constrained physical situation, contextualized reference is extremely high in these texts.'
    
    e1 = 'sports broadcast'
    e2 = 'on-going events'
    
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< e1 >', '<e1>')
    sentence = sentence.replace('< e2 >', '<e2>')
    sentence = sentence.replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< /e2 >', '</e2>')

    print(sentence)