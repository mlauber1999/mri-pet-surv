# network models for vision transformer
# Created: 6/16/2021
# Status: in progress

import sys
import math

import torch.nn as nn

import torch

class _Pre_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        t_layer = nn.TransformerEncoderLayer(d_model=config['emsize'], nhead=config['nhead'], dim_feedforward=config['dim_feedforward'], dropout=config['dropout'])
        self.t_encoder = nn.TransformerEncoder(encoder_layer=t_layer, num_layers=config['nlayers'])
        
        if self.config['mapping'] == 'Linear':
            self.map1 = nn.Linear(60*145*121, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
            self.map2 = nn.Linear(61*145*121, config['emsize'])
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        fh, fw, fd = config['patch_size']
        self.embedding_cnn = nn.Conv3d(1, config['emsize'], kernel_size=(fh, fw, fd), stride=(fh, fw, fd))

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['emsize']))# ?????

        # self.embed3 = nn.Embedding(100, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
        # Note seq len need to update if input size is different or patch size is different
        # self.pos_encoder1 = PositionalEncoding1(config['seq_len']+1, config['emsize']) #plus one for class token
        # self.pos_encoder1 = PositionalEncoding1(config['seq_len'], config['emsize']) #plus one for class token
        self.pos_encoder2 = PositionalEncoding2(config['emsize'])
        self.t_decoder = nn.Linear(config['emsize'], config['out_dim'])
        # self.t_decoder2 = nn.Linear(config['seq_len']+1, config['out_dim'])
        # self.a = nn.GELU()
        self.da = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        
        if self.config['mapping'] != 'Linear':
            x = self.embedding_cnn(x)
            (n, e, h, w, d) = x.shape
            x = x.reshape(n, e, h*w*d)
            x = x.permute(2, 0, 1)
            # print(x.shape)
            # sys.exit()
        
        else:
            n = x.shape[0]
            x1 = x[:,:,:60].flatten().view(n, -1) #in here shape[0] is the batch size
            x2 = x[:,:,60:].flatten().view(n, -1) #in here shape[0] is the batch size
            x1 = self.map1(x1)
            x2 = self.map2(x2)
            x = torch.stack((x1, x2), dim=0)
            # print(x.shape)
            # sys.exit()
        
        # prepend class token
        cls_token = self.cls_token.repeat(1, n, 1)
        x = torch.cat([cls_token, x], dim=0)

        # x = self.pos_encoder1(x)
        x = self.pos_encoder2(x)
        # print(x.shape)
        # sys.exit()
        x = self.t_encoder(x)
        # print(x.shape)
        # sys.exit()
        # x = x.view(b_size, -1)
        # print('before linear', x.shape)
        x = self.t_decoder(x)
        # x = self.a(x)
        # x = self.t_decoder2(x.squeeze().permute(1,0))
        x = self.da(x)
        # print('last', x.shape)
        # print(x)
        # sys.exit()
        x = x[0]
        # print('last first', x.shape)
        # sys.exit()


        return x

class _ViT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        t_layer = nn.TransformerEncoderLayer(d_model=config['emsize'], nhead=config['nhead'], dim_feedforward=config['dim_feedforward'], dropout=config['dropout'])
        self.t_encoder = nn.TransformerEncoder(encoder_layer=t_layer, num_layers=config['nlayers'])
        self.map1 = nn.Linear(60*145*121, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
        self.map2 = nn.Linear(61*145*121, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        fh, fw, fd = config['patch_size']
        self.embedding_cnn = nn.Conv3d(1, config['emsize'], kernel_size=(fh, fw, fd), stride=(fh, fw, fd))

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['emsize']))

        # self.embed3 = nn.Embedding(100, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
        # Note seq len need to update if input size is different or patch size is different
        self.pos_encoder1 = PositionalEncoding1(config['seq_len']+1, config['emsize']) #plus one for class token
        # self.pos_encoder1 = PositionalEncoding1(config['seq_len'], config['emsize']) #plus one for class token
        self.pos_encoder2 = PositionalEncoding2(config['emsize'])
        self.t_decoder = nn.Linear(config['emsize'], config['out_dim'])
        self.da = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        
        if self.config['mapping'] != 'Linear':
            x = self.embedding_cnn(x)
            (n, e, h, w, d) = x.shape
            x = x.reshape(n, e, h*w*d)
            x = x.permute(2, 0, 1)
            # print(x.shape)
            # sys.exit()
        
        else:
            n = x.shape[0]
            x1 = x[:,:,:60].flatten().view(n, -1) #in here shape[0] is the batch size
            x2 = x[:,:,60:].flatten().view(n, -1) #in here shape[0] is the batch size
            x1 = self.map1(x1)
            x2 = self.map2(x2)
            x = torch.stack((x1, x2), dim=0)
            # print(x.shape)
            # sys.exit()
        
        # prepend class token
        cls_token = self.cls_token.repeat(1, n, 1)
        x = torch.cat([cls_token, x], dim=0)

        # x = self.pos_encoder1(x)
        x = self.pos_encoder2(x)
        # print(x.shape)
        # sys.exit()
        x = self.t_encoder(x)
        # print(x.shape)
        # sys.exit()
        # x = x.view(b_size, -1)
        # print('before linear', x.shape)
        x = self.t_decoder(x)
        x = self.da(x)
        # print('last', x.shape)
        x = x[0]
        # print('last first', x.shape)
        # sys.exit()


        return x

class PositionalEncoding1(nn.Module):
    def __init__(self, seq_len, emb_dim, dropout_rate=0.1):
        super().__init__()
        # super(PositionalEncoding, self).__init__()
        # print(1, num_patches + 1, emb_dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        # print('self.pos_embedding')
        # print(self.pos_embedding.shape)
        # print(x.shape)
        # sys.exit()
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)
        return out

class PositionalEncoding2(nn.Module):
    #need update
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        # super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('pos')
        # print(self.pe.shape)
        # print(self.pe[:x.size(0), :].shape)
        # print(self.pe[:x.size(0), :])
        x = x + self.pe[:x.size(0), :]
        # print(x.shape)
        # sys.exit()
        return self.dropout(x)
