import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertConfig
from transformers import BertModel
from transformers import WEIGHTS_NAME, CONFIG_NAME
from utils.constants_paths import bert_path
from utils.path_utils import join_path

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class AttentionBiLSTMBERT(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()
        self.num_classes = num_classes
        self.config = config
        
        self.bert = BertModel.from_pretrained(bert_path)
        bert_config = BertConfig.from_pretrained(bert_path)

        self.max_length = config.max_length
        self.word_dim = bert_config.hidden_size
        self.hidden_dim = config.hidden_dim

        self.lstm = nn.LSTM(self.word_dim, self.hidden_dim, config.num_layers, 
            batch_first=True, bidirectional=True)

        self.tanh = nn.Tanh()
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        self.lstm_dropout = nn.Dropout(config.lstm_dropout)
        self.linear_dropout = nn.Dropout(config.linear_dropout)

        self.attention_weight = nn.Parameter(torch.randn(1, config.hidden_dim, 1))

        self.dense_output = nn.Linear(config.hidden_dim, self.num_classes)
        self.__initialize_dense_layer()
        self.loss = nn.CrossEntropyLoss()


    def __initialize_dense_layer(self):
        init.xavier_normal_(self.dense_output.weight)
        init.constant_(self.dense_output.bias, 0.)


    def __bert_layer(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, 
            attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_hidden_states = outputs[0] # B*L*H
        #cls_hidden_state = outputs[1] # B*H
        return sequence_hidden_states


    def __lstm_layer(self, embeddings, mask):
        non_padded_lengths = torch.sum(mask.gt(0), dim=-1).to(torch.device('cpu'))
        # Optimize computations removing padding
        embeddings = pack_padded_sequence(embeddings, non_padded_lengths, batch_first=True, enforce_sorted=False)

        hidden_states, (_, _) = self.lstm(embeddings)
        # Restore padding to keep the same dimensions
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, padding_value=0.0, total_length=self.max_length)
        hidden_states = hidden_states.view(-1, self.max_length, 2, self.hidden_dim)

        # Bidirectional merges by summing
        hidden_states = torch.sum(hidden_states, dim=2)  # B*L*H
        return hidden_states


    def __attention_layer(self, H, mask):
        # B*L*H
        M = self.tanh(H)
        # B*H*1
        attention_weight = self.attention_weight.expand(mask.shape[0], -1, -1)
        # B*L*1
        attention_score = torch.bmm(M, attention_weight)

        # Remove the padded tokens to compute the softmax
        mask = mask.unsqueeze(dim=-1)  # B*L*1
        # Exp of negative infitnity is 0, so we ignore them during softmax computation
        attention_score = attention_score.masked_fill(mask.eq(0), float('-inf'))  # B*L*1
        attention_weights = F.softmax(attention_score, dim=1)  # B*L*1

        # B*H*L *  B*L*1 -> B*H*1 -> B*H
        context_vector = torch.bmm(H.transpose(1, 2), attention_weights).squeeze(dim=-1)  
        context_vector = self.tanh(context_vector)
        return context_vector


    def forward(self, data, labels):
        # B*L
        word_indices = data[:, 0, :].view(-1, self.max_length)
        # B*L
        mask = data[:, 1, :].view(-1, self.max_length)

        # Boolean mask separating the padded tokens        
        attention_mask = mask.gt(0).float()
        # Boolean mask separating the 2 sentences tokens (not the case here)
        token_type_ids = mask.gt(-1).long()

        # B*L*word_dim
        sequence_hidden_states = self.__bert_layer(word_indices, attention_mask, token_type_ids)
        embeddings = self.embed_dropout(sequence_hidden_states)
        
        # B*L*hidden_dim
        hidden_states = self.__lstm_layer(embeddings, mask)  
        hidden_states = self.lstm_dropout(hidden_states)

        # B*hidden_dim
        sentence_representation = self.__attention_layer(hidden_states, mask)
        sentence_representation = self.linear_dropout(sentence_representation)

        logits = self.dense_output(sentence_representation)
        loss = self.loss(logits, labels.type(torch.LongTensor).to(device))
        return loss, logits


    def save_model(self, save_path):
        torch.save(self.state_dict(), join_path(save_path, WEIGHTS_NAME))
        self.bert.config.to_json_file(join_path(save_path, CONFIG_NAME))