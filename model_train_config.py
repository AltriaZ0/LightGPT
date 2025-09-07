import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import math
from torch.autograd import Variable

# Transformer模型参数设置

# 轻小说文本文件名
txt_name = 'FullNovel.txt' 

# 训练与验证时使用的字符串长度
block_size = 512

#分批的规模，代表同时处理多少条独立数据
batch_size = 16

Embedding_dim = 512 # 嵌入层的维度

max_new_tokens = 100 # 生成文本的最大新令牌数

learning_rate = 3e-4 #学习率

iters = 20000 # 训练迭代次数
eval_iters = 1 # 评估迭代次数

eval_interval = int( iters / eval_iters  ) # 评估间隔

dropout = 0.2 # dropout比例

num_heads = 16
head_size = Embedding_dim // num_heads

numOfLayers = 8 # 多级残差网络的层数

