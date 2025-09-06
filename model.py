
from model_config import *
from data import TextDataProcessor
from nvtx import nvtx_range

processor = TextDataProcessor(txt_name="ymsh.txt")

# 从处理器获取必要数据
train_data = processor.get_train_data()
val_data = processor.get_val_data()
vocab_size = processor.get_vocab_size()
idx_to_char = processor.idx_to_char  # 获取索引到字符的映射

# Batching
@nvtx_range()
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
        pe[:, 0::2] = torch.sin(position * div_term).cuda() # 偶数索引使用sin
        pe[:, 1::2] = torch.cos(position * div_term).cuda() # 奇数索引使用cos
        
        # 增加batch维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)

    @nvtx_range()
    def forward(self, x):
        """
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :return: 添加位置编码后的张量
        """
        # 添加位置编码（只取前x.size(1)个位置）
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x)
    
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(Embedding_dim, head_size, bias = False)
        self.query = nn.Linear(Embedding_dim, head_size, bias = False)
        self.value = nn.Linear(Embedding_dim, head_size, bias = False)
        with torch.cuda.nvtx.range("Tril"):
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # 上三角矩阵：不参与训练
        self.dropout = nn.Dropout(dropout)

    @nvtx_range()
    def forward(self,x):
        with torch.cuda.nvtx.range("reshape"):
            B,T,C = x.shape
        with torch.cuda.nvtx.range("Linear:Key"):
            k = self.key(x) # (B,T,head_size)
        with torch.cuda.nvtx.range("Linear:Query"):
            q = self.query(x) # (B,T,head_size)
        with torch.cuda.nvtx.range("Attention computation"):
            with torch.cuda.nvtx.range("q@kT"):
                att = q @ k.transpose(-2,-1) * k.shape[-1] **-0.5  # KV运算：注意力方阵
                att = att.masked_fill(self.tril[:T, :T]  == 0, float('-inf')) # 用负无穷填充上三角
            with torch.cuda.nvtx.range("softmax"):
                att = F.softmax(att, dim=-1) # 按行做softmax
            with torch.cuda.nvtx.range("Dropout"):
                att = self.dropout(att)
            with torch.cuda.nvtx.range("Linear:Value"):
                v = self.value(x) # (B,T,head_size)
            with torch.cuda.nvtx.range("att@v"):
                out = att @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out
    

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, Embedding_dim)
        self.dropout = nn.Dropout(dropout)
    @nvtx_range()
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


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

    @nvtx_range()
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size) # 自注意力
        self.ffwd = FeedForward(Embedding_dim) # 前向传播
        self.ln1 = nn.LayerNorm(Embedding_dim) # 层归一化
        self.ln2 = nn.LayerNorm(Embedding_dim)

    @nvtx_range()
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
        self.blocks = nn.Sequential(*[Block() for _ in range(numOfLayers)]) # 脚注⑤
        self.ln_f = nn.LayerNorm(Embedding_dim) # 最后的层归一化
        self.lm_head = nn.Linear(Embedding_dim, vocab_size)
        
    @nvtx_range()
    def forward(self, idx, targets=None):
        B, T = idx.shape
        with torch.cuda.nvtx.range("词嵌入"):
            token_embd = self.token_embedding_table(idx)  # (B, T, Embedding_dim)
        with torch.cuda.nvtx.range("位置编码"):
            x = self.positional_encoding(token_embd)  # (T, Embedding_dim)
       
        x = self.blocks(x)  # (B, T, Embedding_dim)

        with torch.cuda.nvtx.range("Final    and Linear"):
            x = self.ln_f(x)  # (B, T, Embedding_dim)
            logits = self.lm_head(x)

        if targets is not None:
            logits = logits.view(B * T, vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets) #交叉熵
        else:
            loss = None
        return logits, loss
    
    @torch.no_grad()     
    @nvtx_range()
    def generate(self, idx, max_new_tokens=100):
        """
        输入: idx (B, T) 的起始序列
        输出: 生成后的序列 (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 截取上下文窗口
            idx_cond = idx[:, -block_size:]
            # 前向传播
            logits, _ = self(idx_cond)
            # 取最后一个时间步
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # 采样
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 拼接
            idx = torch.cat((idx, next_token), dim=1)
        return idx

@torch.no_grad() # 不做梯度计算的decorator,作用域为整个函数
@nvtx_range()
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



