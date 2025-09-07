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
block_size = 256

Embedding_dim = 512 # 嵌入层的维度

max_new_tokens = 100 # 生成文本的最大新令牌数

dropout = 0 # dropout比例

num_heads = 8
head_size = Embedding_dim // num_heads

numOfLayers = 8 # 多级残差网络的层数

