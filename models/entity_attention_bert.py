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


class EntityAttentionBERT(nn.Module):

    def __init__(self, num_classes, num_positions, config):
        super().__init__()
        self.num_classes = num_classes
        self.config = config

        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.hidden_dim = config.hidden_dim

        bert_config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.word_dim = bert_config.hidden_size

        self.pos1_embed = nn.Embedding(num_positions+1, config.pos_dim)
        self.pos2_embed = nn.Embedding(num_positions+1, config.pos_dim)

        self.mhsa = nn.MultiheadAttention(self.word_dim, config.num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.word_dim)

        self.lstm = nn.LSTM(self.word_dim, config.hidden_dim, batch_first=True, bidirectional=True)

        self.latent_weights = nn.Parameter(torch.randn(config.num_latents, 2*config.hidden_dim))
        self.pos_dense = nn.Linear(2*config.hidden_dim + 2*config.pos_dim, config.attention_dim, bias=False)
        self.entity_dense = nn.Linear(8*config.hidden_dim, config.attention_dim, bias=False)
        self.v = nn.Parameter(torch.randn(config.attention_dim))

        self.tanh = nn.Tanh()
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        self.lstm_dropout = nn.Dropout(config.lstm_dropout)
        self.linear_dropout = nn.Dropout(config.linear_dropout)

        self.dense_output = nn.Linear(2*config.hidden_dim, self.num_classes)
        self.__initialize_layers()
        self.loss = nn.CrossEntropyLoss()


    def __initialize_layers(self):
        init.xavier_normal_(self.dense_output.weight)
        init.constant_(self.dense_output.bias, 0.)
        init.xavier_normal_(self.pos_dense.weight)
        init.xavier_normal_(self.entity_dense.weight)
        init.xavier_normal_(self.pos1_embed.weight)
        init.xavier_normal_(self.pos2_embed.weight)

    
    def __bert_layer(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, 
            attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_hidden_states = outputs[0] # B*L*H
        #cls_hidden_state = outputs[1] # B*H
        return sequence_hidden_states


    def __multi_head_attention(self, embeddings, mask):
        key_attention_mask = mask.eq(0) # B*L
        # B*L*word_dim
        improved_embeddings, _ = self.mhsa(embeddings, embeddings, embeddings, key_attention_mask)
        improved_embeddings = self.layer_norm(improved_embeddings)
        return improved_embeddings


    def __lstm_layer(self, embeddings, mask):
        non_padded_lengths = torch.sum(mask.gt(0), dim=-1).to(torch.device('cpu'))
        # Optimize computations removing padding
        embeddings = pack_padded_sequence(embeddings, non_padded_lengths, batch_first=True, enforce_sorted=False)
        hidden_states, (_, _) = self.lstm(embeddings)
        # Restore padding to keep the same dimensions
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, padding_value=0.0, total_length=self.max_length)
        # B*L*(2*H)
        return hidden_states


    def __get_latent_type(self, entity_hidden):
        # B*(2*H) * (2*H)*M -> B*M
        logits = torch.matmul(entity_hidden, self.latent_weights.transpose(1, 0))
        probs = F.softmax(logits, dim=-1) # B*M
        # B*M * M*(2*H) -> B*(2*H)
        latent_type = torch.matmul(probs, self.latent_weights)
        return latent_type


    def __get_entity_last_hidden_state(self, hidden_states, end_index):
        end_index = end_index.type(torch.LongTensor)
        temp = torch.arange(hidden_states.size(0))
        return hidden_states[temp, end_index]


    def __entity_attention_layer(self, hidden_states, rel_pos_e1_embed, rel_pos_e2_embed, e1_end, e2_end, mask):
        # B*(2*H)
        e1_hidden = self.__get_entity_last_hidden_state(hidden_states, e1_end)
        e2_hidden = self.__get_entity_last_hidden_state(hidden_states, e2_end)
        # B*(2*H)
        e1_type = self.__get_latent_type(e1_hidden)
        e2_type = self.__get_latent_type(e2_hidden)
        # B*(8*H)
        entity_features = torch.cat([e1_hidden, e1_type,e2_hidden, e2_type], dim=-1)

        # B*L*(2*H + 2*pos_dim)
        pos_features = torch.cat([hidden_states, rel_pos_e1_embed, rel_pos_e2_embed], dim=-1)

        # B*A
        entity_dense_out = self.entity_dense(entity_features)
        # B*1*A
        entity_dense_out = entity_dense_out.unsqueeze(dim=1)
        # B*L*A
        entity_dense_out = entity_dense_out.expand(-1, self.max_length, -1)
        # B*L*A
        pos_dense_out = self.pos_dense(pos_features)
        # B*L*A
        u = self.tanh(pos_dense_out + entity_dense_out)
        # B*L
        vu = torch.matmul(u, self.v)

        # Remove the padded tokens to compute the softmax
        # Exp of negative infitnity is 0, so we ignore them during softmax computation
        vu = vu.masked_fill(mask.eq(0), float('-inf'))  # B*L
        alphas = F.softmax(vu, dim=1).unsqueeze(dim=-1)  # B*L*1

        # B*H*L *  B*L*1 -> B*H*1 -> B*H
        z = torch.bmm(hidden_states.transpose(1, 2), alphas).squeeze(dim=-1)  
        return z


    def forward(self, data, labels):
        word_indices = data[:, 0, :].view(-1, self.max_length)
        pos_e1 = data[:, 1, :].view(-1, self.max_length)
        pos_e2 = data[:, 2, :].view(-1, self.max_length)
        e1_end = data[:, 3, 0]
        e2_end = data[:, 3, 1]
        mask = data[:, 4, :].view(-1, self.max_length)

        # Boolean mask separating the padded tokens        
        attention_mask = mask.gt(0).float()
        # Boolean mask separating the 2 sentences tokens (not the case here)
        token_type_ids = mask.gt(-1).long()

        # B*L*word_dim
        sequence_hidden_states = self.__bert_layer(word_indices, attention_mask, token_type_ids)
        word_embed = self.embed_dropout(sequence_hidden_states)

        # B*L*pos_dim
        pos_e1_embed = self.pos1_embed(pos_e1)
        pos_e2_embed = self.pos2_embed(pos_e2)
        
        # B*L*word_dim
        #improved_word_embed = self.__multi_head_attention(word_embed, mask)
        
        # B*L*(2*H)
        hidden_states = self.__lstm_layer(word_embed, mask)
        hidden_states = self.lstm_dropout(hidden_states)

        # B*hidden_dim
        sentence_representation = self.__entity_attention_layer(hidden_states, pos_e1_embed, pos_e2_embed, 
            e1_end, e2_end, mask)
        sentence_representation = self.linear_dropout(sentence_representation)

        logits = self.dense_output(sentence_representation)
        loss = self.loss(logits, labels.type(torch.LongTensor).to(device))
        return loss, logits


    def save_model(self, save_path):
        torch.save(self.state_dict(), join_path(save_path, WEIGHTS_NAME))
        self.bert.config.to_json_file(join_path(save_path, CONFIG_NAME))