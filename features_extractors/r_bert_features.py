import numpy as np
from transformers import BertTokenizer
from utils.constants_paths import bert_path


class RBERTFeatureExtractor:

    def __init__(self, max_length, words_vocabulary):
        self.max_length = max_length
        self.entity_markers = ['#', '$']
        self.bert_tokens = ['[CLS]', '[SEP]', '[PAD]']
        self.__create_bert_tokenizer()
        self.__create_word_to_index(words_vocabulary)


    def __create_bert_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': self.entity_markers})
    

    def __create_word_to_index(self, words_vocabulary):
        self.word_to_index = {}

        for token in self.bert_tokens + self.entity_markers:
            self.word_to_index[token] = [self.tokenizer.convert_tokens_to_ids(token)]

        for token in words_vocabulary:
            token = token.lower()

            if token in self.word_to_index:
                continue
            
            # Could be a composed word that translates into a list of tokens
            token_temp = self.tokenizer.tokenize(token)

            if len(token_temp) < 1:
                token_idx_list = [self.tokenizer.convert_tokens_to_ids('[UNK]')]
            else:
                token_idx_list = self.tokenizer.convert_tokens_to_ids(token_temp)

            self.word_to_index[token] = token_idx_list


    def __tokenize_sentence(self, sentence):
        tokenized_sentence = []
        for token in sentence.split():
            token = token.lower()
            if token == 'e11' or token == 'e12':
                tokenized_sentence += self.word_to_index['$']
            elif token == 'e21' or token == 'e22':
                tokenized_sentence += self.word_to_index['#']
            else:
                tokenized_sentence += self.word_to_index[token]
        return tokenized_sentence


    def __create_mask(self, sentence_length, e1_start, e1_end, e2_start, e2_end):
        mask = [3] * sentence_length
        mask[e1_start : e1_end + 1] = [4] * (e1_end - e1_start + 1)
        mask[e2_start : e2_end + 1] = [5] * (e2_end - e2_start + 1)
        return mask


    def __truncate_if_needed(self, tokenized_sentence, mask):
        if len(tokenized_sentence) > self.max_length - 2:
            tokenized_sentence = tokenized_sentence[: self.max_length - 2]
            mask = mask[: self.max_length - 2]
        return tokenized_sentence, mask


    def __add_bert_tokens(self, tokenized_sentence, mask):
        pad_length = self.max_length - 2 - len(tokenized_sentence)
        mask = [1] + mask + [2] + [0] * pad_length
        tokenized_sentence = ( 
            self.word_to_index[self.bert_tokens[0]] +
            tokenized_sentence + 
            self.word_to_index[self.bert_tokens[1]] +
            self.word_to_index[self.bert_tokens[2]] * pad_length
        )
        return tokenized_sentence, mask


    def compute_features(self, data, labels):
        features = []
        for sentence_index in data:

            sentence_data = data[sentence_index]
            sentence = sentence_data['sentence']
            e1_start = sentence_data['e1_start'] - 1
            e1_end = sentence_data['e1_end'] + 1
            e2_start = sentence_data['e2_start'] - 1
            e2_end = sentence_data['e2_end'] + 1

            tokenized_sentence = self.__tokenize_sentence(sentence)
            mask = self.__create_mask(len(tokenized_sentence), e1_start, e1_end,
                e2_start, e2_end)
            
            tokenized_sentence, mask = self.__truncate_if_needed(tokenized_sentence, mask)
            tokenized_sentence, mask = self.__add_bert_tokens(tokenized_sentence, mask)

            sentence_features = np.array([tokenized_sentence, mask], dtype=np.int32)
            features.append(sentence_features)

        return np.array(features), np.array(labels)