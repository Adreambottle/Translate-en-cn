#!/user/bin/env python
# coding=utf-8
'''
@project : Translate-en-cn
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : test.py
@ide     : PyCharm
@time    : 2022-06-23

@Description:
'''
UNK = 0           # The id of unknown word in the vocabulary
PAD = 1           # The id of padding word in the vocabulary
BATCH_SIZE = 64   # Batch size, data number in a data
EPOCHS = 20       # Epochs
LAYERS = 6        # encoder and decoder blocks number in the transformer
H_NUM = 8         # multihead attention hidden个数
D_MODEL = 256     # embedding维数
D_FF = 1024       # feed forward第一个全连接层维数
DROPOUT = 0.1     # dropout比例
MAX_LENGTH = 60   # 最大句子长度

TRAIN_FILE = 'nmt/en-cn/train.txt'    # 训练集数据文件
DEV_FILE = "nmt/en-cn/dev.txt"        # 验证(开发)集数据文件
SAVE_FILE = 'save/model.pt'           # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


src_vocab
tgt_vocab
N = 6
d_model = 512
d_ff = 2048
h = 8
dropout = 0.1
import torch
import copy
c = copy.deepcopy

# instantiate the Attention module
attn = MultiHeadedAttention(h, d_model).to(DEVICE)
attn(ta, ta, ta, mask)

# instantiate the Feed Forward module
ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
ff(ta)

# instantiate the PositionalEncoding module
position = PositionalEncoding(d_model, dropout).to(DEVICE)




N = 4
L = 20
D = 512
ta = torch.randn((N, L, D))

ta.shape

AAD = AddAndNormLayer(D, 0.1, c(attn))
AAD(ta, ta, ta, ta, mask)
MHT(ta, ta)
AAD(ta, MHT)

mask = subsequent_mask(L)
mask = torch.concat([mask]*N)

attention(ta, ta, ta, mask, dropout)

EL = EncoderLayer(d_model, c(attn), c(ff), dropout)
DL = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
EL(ta, mask)
DL(ta, ta, mask, mask)
MHT = MultiHeadedAttention(2, 8, 0.1)
MHT2 = MultiHeadedAttention_v2(2, 8, 0.1)

MHT(ta, ta, ta, mask)
x = MHT2(ta, ta, ta, mask_MHA)
x = MHT(ta, ta, ta, mask_MHA)
mask.shape



mask_MHA = torch.concat([mask]*2).view(N, -1, L, L)
# instantiate the Transformer module


x
model = Transformer(
    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
    Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
    nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
    nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
    Generator(d_model, tgt_vocab)).to(DEVICE)



def l1(x_1, k, b):
    return k * x_1 + b

def l2(x, b):
    return x**2 +b

def func_r(x, function, **kwargs):
    for k, v in kwargs.items():
        print(f"key:{k}, value:{v}")
    return x + function(**kwargs)

func_r(x=1, function=l1, x_1=1, k=1, b=1)