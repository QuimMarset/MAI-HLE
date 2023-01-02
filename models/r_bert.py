import torch
import torch.nn as nn
from torch.nn import init
from transformers import BertConfig
from transformers import BertModel
from transformers import WEIGHTS_NAME, CONFIG_NAME
from utils.constants_paths import bert_path
from utils.path_utils import join_path

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class RBERT(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()
        self.num_classes = num_classes

        bert_config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_hidden_size = bert_config.hidden_size

        self.max_length = config.max_length
        self.dropout_value = config.dropout

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_value)

        self.cls_dense = nn.Linear(self.bert_hidden_size, out_features=self.bert_hidden_size)
        self.entity_dense = nn.Linear(self.bert_hidden_size, self.bert_hidden_size)
        self.output_dense = nn.Linear(self.bert_hidden_size*3, self.num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.__initialize_dense_layers()


    def __initialize_dense_layers(self):
        init.xavier_uniform_(self.cls_dense.weight)
        init.constant_(self.cls_dense.bias, 0)

        init.xavier_uniform_(self.entity_dense.weight)
        init.constant_(self.entity_dense.bias, 0)

        init.xavier_uniform_(self.output_dense.weight)
        init.constant_(self.output_dense.bias, 0)


    def __bert_layer(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, 
            attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_hidden_states = outputs[0] # B*L*H
        cls_hidden_state = outputs[1] # B*H
        return sequence_hidden_states, cls_hidden_state


    def __get_entity_hidden_state(self, sequence_hidden_states, entity_mask):
        entity_mask = entity_mask.unsqueeze(dim=1).float()  # B*1*L

        # B*1*L * B*L*H -> B*1*H
        sum_entity_hidden_states = torch.bmm(entity_mask, sequence_hidden_states)
        # B*1*H -> B*H
        sum_entity_hidden_states = sum_entity_hidden_states.squeeze(dim=1) 

        # B*1
        entity_length = entity_mask.sum(dim=-1).float()
        
        # B*H
        average_entity_hidden_state = torch.div(sum_entity_hidden_states, entity_length)
        return average_entity_hidden_state


    def forward(self, data, labels):
        input_indices = data[:, 0, :].view(-1, self.max_length)
        mask = data[:, 1, :].view(-1, self.max_length)

        # Boolean mask separating the padded tokens        
        attention_mask = mask.gt(0).float()
        # Boolean mask separating the 2 sentences tokens (not the case here)
        token_type_ids = mask.gt(-1).long()

        sequence_hidden_states, cls_hidden_state = self.__bert_layer(
            input_indices, attention_mask, token_type_ids)

        cls_hidden_state = self.dropout(cls_hidden_state)
        cls_dense_out = self.tanh(self.cls_dense(cls_hidden_state))

        # Boolean mask separating the first entity
        e1_mask = mask.eq(4)
        e1_hidden_state = self.__get_entity_hidden_state(sequence_hidden_states, e1_mask)
        e1_hidden_state = self.dropout(e1_hidden_state)
        e1_dense_out = self.tanh(self.entity_dense(e1_hidden_state))

        # Boolean mask separating the second entity
        e2_mask = mask.eq(5)
        e2_hidden_state = self.__get_entity_hidden_state(sequence_hidden_states, e2_mask)
        e2_hidden_state = self.dropout(e2_hidden_state)
        e2_dense_out = self.tanh(self.entity_dense(e2_hidden_state))

        # Concatenate the 3 outputs into a single vector
        final_representation = torch.cat([cls_dense_out, e1_dense_out, e2_dense_out], dim=-1)
        final_representation = self.dropout(final_representation)

        logits = self.output_dense(final_representation)
        loss = self.loss(logits, labels.type(torch.LongTensor).to(device))
        return loss, logits


    def save_model(self, save_path):
        torch.save(self.state_dict(), join_path(save_path, WEIGHTS_NAME))
        self.bert.config.to_json_file(join_path(save_path, CONFIG_NAME))
