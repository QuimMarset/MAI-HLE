import torch
import torch.nn as nn
from torch.nn import init
from utils.path_utils import join_path
from transformers import WEIGHTS_NAME

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')



class CNN(nn.Module):
    def __init__(self, num_classes, config, word_embedding_matrix):
        super().__init__()
        
        self.num_classes = num_classes
        self.word_embed_matrix = torch.from_numpy(word_embedding_matrix)
        self.max_length = config.max_length
        self.num_filters = config.num_filters

        self.word_embedding = nn.Embedding.from_pretrained(self.word_embed_matrix, freeze=False)
        # The layer needs an extra position
        num_positions = 2 * config.max_distance + 3
        self.pos1_embedding = nn.Embedding(num_embeddings=num_positions, embedding_dim=config.pos_dim)
        self.pos2_embedding = nn.Embedding(num_embeddings=num_positions, embedding_dim=config.pos_dim)

        num_channels = config.word_dim + 2 * config.pos_dim
        self.conv = nn.Conv2d(1, self.num_filters, (3, num_channels), stride=(1, 1), padding=(1, 0))
        self.maxpool = nn.MaxPool2d((self.max_length, 1))

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)
        
        self.dense = nn.Linear(self.num_filters, config.dense_units)
        self.output = nn.Linear(config.dense_units, self.num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.__initialize_weights()


    def __initialize_weights(self):
        init.xavier_normal_(self.pos1_embedding.weight)
        init.xavier_normal_(self.pos2_embedding.weight)

        init.xavier_normal_(self.conv.weight)
        init.constant_(self.conv.bias, 0.)

        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

        init.xavier_normal_(self.output.weight)
        init.constant_(self.output.bias, 0.)


    def __encode(self, word_indices, pos1, pos2):
        word_embed = self.word_embedding(word_indices)  # B*L*word_dim
        pos1_embed = self.pos1_embedding(pos1)  # B*L*pos_dim
        pos2_embed = self.pos2_embedding(pos2)  # B*L*pos_dim
        embed = torch.cat(tensors=[word_embed, pos1_embed, pos2_embed], dim=-1)
        return embed  # B*L*(word_dim + 2*pos_dim)


    def __convolution(self, embed, mask):
        embed = embed.unsqueeze(dim=1)  # B*1*L*D
        conv_output = self.conv(embed)  # B*C*L*1

        # Use the mask to remove the padded entries before computing the max_pooling
        conv_output = conv_output.view(-1, self.num_filters, self.max_length)  # B*C*L
        mask = mask.unsqueeze(dim=1)  # B*1*L
        mask = mask.expand(-1, self.num_filters, -1)  # B*C*L
        conv_output = conv_output.masked_fill_(mask.eq(0), float('-inf'))  # B*C*L
        conv_output = conv_output.unsqueeze(dim=-1)  # B*C*L*1
        return conv_output


    def __max_pooling(self, conv_output):
        max_pool_output = self.maxpool(conv_output)  # B*C*1*1
        max_pool_output = max_pool_output.view(-1, self.num_filters)  # B*C
        return max_pool_output


    def forward(self, data, labels):
        word_indices = data[:, 0, :].view(-1, self.max_length)
        pos1 = data[:, 1, :].view(-1, self.max_length)
        pos2 = data[:, 2, :].view(-1, self.max_length)
        mask = data[:, 3, :].view(-1, self.max_length)

        embed = self.__encode(word_indices, pos1, pos2)
        embed = self.dropout(embed)

        conv_output = self.__convolution(embed, mask)
        pool_output = self.__max_pooling(conv_output)
        
        sentence_features = self.dense(pool_output)
        sentence_features = self.tanh(sentence_features)
        sentence_features = self.dropout(sentence_features)
        
        logits = self.output(sentence_features)
        loss = self.loss(logits, labels.type(torch.LongTensor).to(device))
        return loss, logits

    
    def save_model(self, save_path):
        torch.save(self.state_dict(), join_path(save_path, WEIGHTS_NAME))