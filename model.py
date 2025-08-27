
from model_config import *
from data import TextDataProcessor

processor = TextDataProcessor(txt_name="ymsh.txt")

# 从处理器获取必要数据
train_data = processor.get_train_data()
val_data = processor.get_val_data()
vocab_size = processor.get_vocab_size()
idx_to_char = processor.idx_to_char  # 获取索引到字符的映射

# Batching
def get_batch(split, batch_size, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).cuda()
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).cuda()
    return x, y

# 位置编码
class PositionalEncoding(nn.Module):
    """
    实现Transformer中的固定位置编码
    生成后在整个训练过程中保持不变
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        :param d_model: 嵌入维度
        :param dropout: dropout概率
        :param max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).cuda()
        
        # 计算频率项 (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model)).cuda()
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term).cuda()  # 偶数索引使用sin
        pe[:, 1::2] = torch.cos(position * div_term).cuda()  # 奇数索引使用cos
        
        # 增加batch维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :return: 添加位置编码后的张量
        """
        # 添加位置编码（只取前x.size(1)个位置）
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, Embedding_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
#
#  前馈网络
class FeedForward(nn.Module):
    def __init__(self, Embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Embedding_dim, Embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(Embedding_dim * 4, Embedding_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, Embedding_dim, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size) # 自注意力
        self.ffwd = FeedForward(Embedding_dim) # 前向传播
        self.ln1 = nn.LayerNorm(Embedding_dim) # 层归一化
        self.ln2 = nn.LayerNorm(Embedding_dim)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Pre-Norm: 残差多头注意力网络
        x = x + self.ffwd(self.ln2(x))  # Pre-Norm：残差前向传播层
        return x

# 语言模型
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, Embedding_dim)
        self.positional_encoding = PositionalEncoding(Embedding_dim, dropout=0.1, max_len=block_size)
        self.blocks = nn.Sequential(*[Block(Embedding_dim, num_heads) for _ in range(numOfLayers)]) # 脚注⑤
        self.ln_f = nn.LayerNorm(Embedding_dim) # 最后的层归一化
        self.lm_head = nn.Linear(Embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd = self.token_embedding_table(idx)  # (B, T, Embedding_dim)
        position_idx = torch.arange(T).cuda() 
        position_embd = self.positional_encoding(token_embd)  # (T, Embedding_dim)
        x = token_embd + position_embd  # (B, T, Embedding_dim)
        x = self.blocks(x)  # (B, T, Embedding_dim)
        x = self.ln_f(x)  # (B, T, Embedding_dim)
        logits = self.lm_head(x)
        if targets is not None:
            logits = logits.view(B * T, vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets) #交叉熵
        else:
            loss = None
        return logits, loss
    
    def generate(self, token_seq, max_new_tokens=100):
        for _ in range(max_new_tokens):
            tokens_input = token_seq[:, -block_size:]
            logits, loss = self.forward(tokens_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            token_next = torch.multinomial(probs, num_samples=1).cuda() # 把概率分布向量变成one-hot向量，再变成整数token
            token_seq = torch.cat((token_seq, token_next), dim=1)
            new_tokens = token_seq[:, -max_new_tokens:]
        print("生成token长度:", new_tokens.shape[1])
        return new_tokens

@torch.no_grad() # 不做梯度计算的decorator,作用域为整个函数
def estimate_loss(model):
    out = {}
    model.eval() # 把模型转化为evaluate模式（默认模式是train）
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # 建立一个初始值为0的容器，用于储存loss值
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size, block_size) # split是一个字符串，用来控制get_batch()函数的行为
            logits, loss = model(X, Y) # model的输入值一个是index（以每个字符的序号表示的序列），一个是target
            losses[k] = loss.item()
        out[split] = losses.mean() # out是含有两个元素的字典，一个是train，一个是val，每个元素对应一个loss的平均值
    model.train() # 再转化为训练模式（如果之前没有转为evaluate模式，则不需要这一步，因为模型建立后默认为训练模式）
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(Embedding_dim, head_size, bias = False)
        self.query = nn.Linear(Embedding_dim, head_size, bias = False)
        self.value = nn.Linear(Embedding_dim, head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # 上三角矩阵：不参与训练
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        att = q @ k.transpose(-2,-1) * k.shape[-1] **-0.5  # KV运算：注意力方阵
        att = att.masked_fill(self.tril[:T, :T]  == 0, float('-inf')) # 用负无穷填充上三角
        att = F.softmax(att, dim=-1) # 按行做softmax
        att = self.dropout(att)
        v = self.value(x) # (B,T,head_size)
        out = att @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

